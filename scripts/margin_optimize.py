"""광교 안전마진 최적화 — 분위수 × 정률곱 × 정률합, 과적합에 강한 설계.

발주 order = predict_q(quantile) × mult + add.
목적: cost = w · Σ shortfall + Σ surplus  (shortfall=부족/매진분, surplus=잉여/폐기분), w = 5·10.

과적합 방지 3중 장치:
  (1) Rolling-origin(prequential) 평가 — 시간순 "과거로 선택 → 다음 블록 평가" 반복.
      마진 선택 '절차'의 일반화 성능 측정(= conformal 배포 방식, look-ahead 0).
  (2) One-SE + 파시모니 — calib 최소비용 부트스트랩 SE 안(min+1SE) config 중 가장 단순(mult≈1·add≈0·낮은q).
  (3) 절차 비교(q0.85 / 분위수만 / 3손잡이 / 3손잡이+1SE) + 블록별 선택 안정성 + 개선의 부트스트랩 CI.

방법: 목적함수 평가가 초저비용이라 전역 그리드(정확·투명). GA/베이지안 불필요.

2단계:
  collect — 분위수 그리드 windowed_backtest → reports/margin_qmat.parquet (비쌈, 1회)
  opt     — parquet 로드해 rolling-origin 최적화 (쌈, 반복)
  all     — collect + opt (기본)

실행: PYTHONPATH=src:scripts uv run --with matplotlib python scripts/margin_optimize.py [collect|opt|all]
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

import json
from pathlib import Path

import numpy as np
import pandas as pd

import store_predictive_power as sp

Q_GRID = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
STORES = [
    ("1000000047", "store_gw01", "광교"),
    ("1000000009", "store_ss01", "삼성타운"),
    ("1000000029", "store_mp01", "메세나폴리스"),
    ("1000000485", "store_gh01", "광화문"),
]
STORE = STORES[0]   # 단일 매장 opt(rolling-origin) 기본 대상
WEIGHTS = (5.0, 10.0)

Q_FINE = np.round(np.arange(0.50, 0.951, 0.01), 3)
MULT_GRID = np.round(np.arange(1.00, 1.301, 0.02), 3)
ADD_GRID = np.arange(0, 31, 2, dtype=float)

N_EVAL_BLOCKS = 6          # rolling-origin 평가 블록 수 (후반부)
CALIB_FRAC = 0.45          # 최초 calibration 비율(앞부분)
N_BOOT = 300               # 부트스트랩 (SE / CI)
SEED = 20260712

def qmat_path(store_id: str) -> Path:
    return Path(f"reports/margin_qmat_{store_id}.parquet")


QMAT_PARQUET = qmat_path(STORE[1])   # 광교 (opt 기본)
OUT = Path("reports/margin_optimize_result.json")


# ---------------------------------------------------------------------------
# stage 1: collect (expensive) — 매장별 qmat parquet 저장
# ---------------------------------------------------------------------------
def collect_store(cd: str, sid: str, label: str) -> pd.DataFrame:
    print(f"[{label}] build features ...")
    sd = sp.build_store_data(cd, sid, label)
    print(f"[{label}] collect quantile preds ({len(Q_GRID)} quantiles × {sp.MAIN_FOLDS} folds) ...")
    base = None
    for q in Q_GRID:
        res = sp.windowed_backtest(sd.feat, window_days=sp.DEFAULT_WINDOW_DAYS,
                                   n_folds=sp.MAIN_FOLDS, production_q=float(q))
        p = res.predictions[["date", "actual", "production"]].copy()
        p["date"] = pd.to_datetime(p["date"])
        p = p.rename(columns={"production": f"q{q:.2f}"})
        base = p if base is None else base.merge(p[["date", f"q{q:.2f}"]], on="date", how="inner")
        print(f"  q={q:.2f} done ({len(p)} days)")
    preds = base.sort_values("date").reset_index(drop=True)
    pw = sp.load_prod_waste(cd)
    b = preds.merge(sd.raw_incl_bulk, on="date", how="left").merge(
        pw[["date", "production"]].rename(columns={"production": "qt_made"}), on="date", how="left")
    preds["baseline_adj"] = b["qt_made"] - (b["sold_units_incl_bulk"] - b["actual"])
    out = qmat_path(sid)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    print(f"saved {out} ({len(preds)} days)")
    return preds


def run_collect() -> pd.DataFrame:
    """4매장 전부 qmat 수집·저장. 광교 프레임 반환(opt 기본용)."""
    first = None
    for cd, sid, label in STORES:
        p = collect_store(cd, sid, label)
        if first is None:
            first = p
    return first


# ---------------------------------------------------------------------------
# stage 2: robust optimize (cheap)
# ---------------------------------------------------------------------------
def _pred_at_q(qmat: np.ndarray, q: float) -> np.ndarray:
    """일자별 분위수 매트릭스(단조)에서 연속 q 선형보간 → [n_days]."""
    idx = np.interp(q, Q_GRID, np.arange(len(Q_GRID)))
    lo, hi = int(np.floor(idx)), int(np.ceil(idx))
    frac = idx - lo
    return qmat[:, lo] * (1 - frac) + qmat[:, hi] * frac


def _cost(order: np.ndarray, actual: np.ndarray, w: float) -> tuple[float, float, float]:
    short = float(np.clip(actual - order, 0, None).sum())
    surp = float(np.clip(order - actual, 0, None).sum())
    return w * short + surp, short, surp


def _all_configs():
    """(q, mult, add) 그리드 enumerate."""
    for q in Q_FINE:
        for mult in MULT_GRID:
            for add in ADD_GRID:
                yield float(q), float(mult), float(add)


def select_margin(qmat_cal: np.ndarray, act_cal: np.ndarray, w: float, *,
                  parsimony: bool, three_knob: bool, rng: np.random.Generator) -> dict:
    """calib 에서 마진 선택. three_knob=False면 mult=1·add=0 고정(분위수만).
    parsimony=True면 min+1SE 안에서 가장 단순한 config."""
    # q별 예측 미리 계산
    predq = {q: _pred_at_q(qmat_cal, q) for q in Q_FINE}
    rows = []
    mults = MULT_GRID if three_knob else np.array([1.0])
    adds = ADD_GRID if three_knob else np.array([0.0])
    for q in Q_FINE:
        pr = predq[q]
        for mult in mults:
            pm = pr * mult
            for add in adds:
                order = pm + add
                cost, short, surp = _cost(order, act_cal, w)
                rows.append((q, float(mult), float(add), cost, short, surp))
    arr = pd.DataFrame(rows, columns=["q", "mult", "add", "cost", "short", "surp"])
    best_i = int(arr["cost"].idxmin())
    best = arr.loc[best_i]
    # 주의: pandas Series 의 .add/.q 등은 메서드/속성 충돌 → 반드시 bracket 접근.
    bq, bm, ba, bc = float(best["q"]), float(best["mult"]), float(best["add"]), float(best["cost"])
    if not parsimony:
        return {"q": bq, "mult": bm, "add": ba}
    # bootstrap SE of the best config's cost
    n = len(act_cal)
    order_best = _pred_at_q(qmat_cal, bq) * bm + ba
    per_day = w * np.clip(act_cal - order_best, 0, None) + np.clip(order_best - act_cal, 0, None)
    boot = [per_day[rng.integers(0, n, n)].sum() for _ in range(N_BOOT)]
    se = float(np.std(boot))
    within = arr[arr["cost"] <= bc + se].copy()
    # 파시모니 penalty: mult≈1·add≈0·낮은 q 선호
    within["pen"] = (within["mult"] - 1.0).abs() * 60 + within["add"] * 1.0 + (within["q"] - 0.5) * 20
    pick = within.loc[within["pen"].idxmin()]
    return {"q": float(pick["q"]), "mult": float(pick["mult"]), "add": float(pick["add"])}


def rolling_origin(preds: pd.DataFrame, w: float, *, parsimony: bool, three_knob: bool,
                   fixed: dict | None, rng: np.random.Generator) -> dict:
    """시간순 블록별: 과거로 선택 → 블록 평가. pooled OOS 결과 반환.
    fixed 가 주어지면 선택 없이 그 config 고정(예: q0.85)."""
    qcols = [f"q{q:.2f}" for q in Q_GRID]
    qmat = np.maximum.accumulate(preds[qcols].to_numpy(), axis=1)   # 단조성
    actual = preds["actual"].to_numpy()
    n = len(preds)
    cal0 = int(n * CALIB_FRAC)
    edges = np.linspace(cal0, n, N_EVAL_BLOCKS + 1, dtype=int)
    oos_order, oos_act, picks = [], [], []
    for bi in range(N_EVAL_BLOCKS):
        b_start, b_end = edges[bi], edges[bi + 1]
        if b_end <= b_start:
            continue
        cfg = fixed if fixed is not None else select_margin(
            qmat[:b_start], actual[:b_start], w,
            parsimony=parsimony, three_knob=three_knob, rng=rng)
        order = _pred_at_q(qmat[b_start:b_end], cfg["q"]) * cfg["mult"] + cfg["add"]
        oos_order.append(order)
        oos_act.append(actual[b_start:b_end])
        picks.append(cfg)
    order = np.concatenate(oos_order)
    act = np.concatenate(oos_act)
    cost, short, surp = _cost(order, act, w)
    return {
        "cost": cost, "shortfall": short, "surplus": surp,
        "stockout_rate": float((act > order).mean()), "mean_order": float(order.mean()),
        "mean_actual": float(act.mean()), "n_oos": int(len(act)),
        "picks": picks, "_order": order, "_act": act,
    }


def boot_ci_diff(a_order, base_order, act, w, rng) -> dict:
    """개선(base_cost - a_cost)의 부트스트랩 CI. >0이면 a가 더 좋음."""
    n = len(act)
    per_a = w * np.clip(act - a_order, 0, None) + np.clip(a_order - act, 0, None)
    per_b = w * np.clip(act - base_order, 0, None) + np.clip(base_order - act, 0, None)
    diff = per_b - per_a
    boots = [diff[rng.integers(0, n, n)].sum() for _ in range(N_BOOT)]
    return {"improve_mean": float(np.mean(boots)),
            "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5))}


def _stability(picks: list[dict]) -> dict:
    qs = [p["q"] for p in picks]; ms = [p["mult"] for p in picks]; ads = [p["add"] for p in picks]
    return {"q": [round(x, 2) for x in qs], "mult": [round(x, 2) for x in ms], "add": ads,
            "q_std": float(np.std(qs)), "mult_std": float(np.std(ms)), "add_std": float(np.std(ads))}


def run_opt(preds: pd.DataFrame) -> None:
    label = STORE[2]
    res = {"store": label, "n_days": int(len(preds)), "config": {
        "calib_frac": CALIB_FRAC, "n_eval_blocks": N_EVAL_BLOCKS,
        "q_fine": [float(Q_FINE.min()), float(Q_FINE.max())],
        "mult": [float(MULT_GRID.min()), float(MULT_GRID.max())],
        "add": [float(ADD_GRID.min()), float(ADD_GRID.max())]}, "by_weight": {}}
    # baseline 기존 생산량 (rolling-origin 동일 OOS 구간에서)
    qcols = [f"q{q:.2f}" for q in Q_GRID]
    qmat = np.maximum.accumulate(preds[qcols].to_numpy(), axis=1)
    actual = preds["actual"].to_numpy()
    base_adj = preds["baseline_adj"].to_numpy()
    n = len(preds); cal0 = int(n * CALIB_FRAC)

    for w in WEIGHTS:
        rng = np.random.default_rng(SEED)
        procs = {
            "q085_fixed": rolling_origin(preds, w, parsimony=False, three_knob=False,
                                         fixed={"q": 0.85, "mult": 1.0, "add": 0.0}, rng=rng),
            "qonly": rolling_origin(preds, w, parsimony=False, three_knob=False, fixed=None, rng=rng),
            "three_knob": rolling_origin(preds, w, parsimony=False, three_knob=True, fixed=None, rng=rng),
            "three_knob_1se": rolling_origin(preds, w, parsimony=True, three_knob=True, fixed=None, rng=rng),
        }
        base_order = base_adj[cal0:n]; base_act = actual[cal0:n]
        bmask = ~np.isnan(base_order)
        b_cost, b_short, b_surp = _cost(base_order[bmask], base_act[bmask], w)
        base_existing = {"cost": b_cost, "shortfall": b_short, "surplus": b_surp,
                         "stockout_rate": float((base_act[bmask] > base_order[bmask]).mean()),
                         "mean_order": float(base_order[bmask].mean()), "n_oos": int(bmask.sum())}
        # 개선 CI: 각 절차 vs q085_fixed (동일 OOS)
        q085 = procs["q085_fixed"]
        out = {}
        for name, r in procs.items():
            entry = {k: v for k, v in r.items() if not k.startswith("_")}
            entry["stability"] = _stability(r["picks"])
            entry.pop("picks", None)
            if name != "q085_fixed":
                rng2 = np.random.default_rng(SEED + 1)
                entry["ci_vs_q085"] = boot_ci_diff(r["_order"], q085["_order"], r["_act"], w, rng2)
            out[name] = entry
        out["baseline_existing"] = base_existing
        res["by_weight"][str(int(w))] = out

        print(f"\n=== w={w:.0f} (부족분 가중) — rolling-origin OOS ({q085['n_oos']}일) ===")
        for name in ["q085_fixed", "qonly", "three_knob", "three_knob_1se", "baseline_existing"]:
            e = out[name]
            extra = ""
            if "stability" in e:
                st = e["stability"]
                extra = f" | 선택 q={st['q']} mult={st['mult']} add={st['add']}"
            ci = ""
            if "ci_vs_q085" in e:
                c = e["ci_vs_q085"]
                sig = "유의" if c["ci_lo"] > 0 else ("악화" if c["ci_hi"] < 0 else "무차이")
                ci = f" | vs q085 개선 {c['improve_mean']:+.0f} [{c['ci_lo']:+.0f},{c['ci_hi']:+.0f}] {sig}"
            print(f"  {name:20s} cost={e['cost']:8.0f} short={e['shortfall']:6.0f} surp={e['surplus']:6.0f} "
                  f"매진율={e['stockout_rate']*100:4.0f}% 평균발주={e.get('mean_order',0):5.0f}{ci}{extra}")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {OUT}")


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "collect":
        run_collect()
    elif stage == "opt":
        if not QMAT_PARQUET.exists():
            raise SystemExit(f"{QMAT_PARQUET} 없음 — 먼저 collect 실행")
        run_opt(pd.read_parquet(QMAT_PARQUET))
    elif stage == "all":
        run_opt(run_collect())
    else:
        raise SystemExit("usage: margin_optimize.py [collect|opt|all]")


if __name__ == "__main__":
    main()
