"""4매장 카테고리-합 수요예측 예측력 리포트 (FINALIZED regime).

regime (고정):
- target = adjusted_demand_unit, α=0.8 (마감할인 실수요 비율)
- bulk 제외 = bakery.data.bulk.flag_bulk_lines (store_daily.build_store_daily exclude_bulk=True)
- 카테고리 = bread/pastry/sandwich 합 (TARGET_CATEGORIES)
- 예측 cadence = 목요일 anchor, 다음 7일 / 주간 fold (horizon_days=7)
- 발주 quantile = 0.85, 기본 학습창 = rolling 2Y (730일)

산출물:
- reports/store_predictive_power.html  (self-contained, base64 PNG)
- reports/store_predictive_power_summary.json

leakage-safe: 모든 fold 의 train slice 는 test 시작일 이전 데이터만.
windowed backtest 는 category_total.expanding_window_backtest 를 그대로 복제하되
train slice 만 날짜 기반 rolling window 로 교체한다 (test slicing 은 동일).
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

import base64
import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bakery.analysis.seasonal import filter_seasonal
from bakery.data.calendar import LUNAR_EVENT_DATES
from bakery.features.category_aggregate import (
    TARGET_CATEGORIES,
    build_category_daily,
    build_features,
)
from bakery.models.category_total import BacktestResult, fit_category_total
from bakery.models.event_prior import EventLevelPrior
from store_daily import build_store_closing_rows, build_store_daily, item_category_map
from v4_new_data_backtest import V2

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ALPHA = 0.8
PROD_Q = 0.85
TARGET = "adjusted_demand_unit"
DEFAULT_WINDOW_DAYS = 730
MAIN_FOLDS = 52          # 메인 백테스트/헤드라인/variant (약 12개월 test span)
SENS_FOLDS = 26          # window sensitivity 스윕 전용 (약 6개월; 대형창 재적합 비용 절반)
HORIZON = 7
MIN_TRAIN_ROWS = 60
OVERLAY_DAYS = 120
WINDOWS = (90, 180, 365, 730, 1095, 1825)

OUT_HTML = Path("reports/store_predictive_power.html")
OUT_JSON = Path("reports/store_predictive_power_summary.json")
CACHE = Path("reports/store_predictive_power_cache.pkl")   # compute 단계 결과 (html 단계가 로드)

# (store_cd, store_id, label, 기상관측소)  — store_daily.STORE_MAP 기준
STORES = [
    ("1000000047", "store_gw01", "광교", 119),
    ("1000000009", "store_ss01", "삼성타운", 108),
    ("1000000029", "store_mp01", "메세나폴리스", 108),
    ("1000000485", "store_gh01", "광화문", 108),
]
STORE_COLORS = {
    "광교": "#1f77b4",
    "삼성타운": "#d62728",
    "메세나폴리스": "#2ca02c",
    "광화문": "#ff7f0e",
}
LABEL_BY_SID = {sid: label for _, sid, label, _ in STORES}
SID_BY_LABEL = {label: sid for _, sid, label, _ in STORES}

XMAS = {"xmas": (12, 25)}
SEOLLAL = {"seollal": LUNAR_EVENT_DATES["days_to_seollal"]}
CHUSEOK = {"chuseok": LUNAR_EVENT_DATES["days_to_chuseok"]}
# 매장×이벤트 opt-in. 배포 코드(median+min_events=2) OOS 순개선 확인된 것만 등록:
#   광교 추석 0.214→0.145, 메세나 설 0.179→0.101. (광화문 설·메세나 추석=악화라 미등록)
STORE_EVENT_PRIORS: dict[str, dict[str, dict]] = {
    "광교":       {"events": dict(XMAS), "lunar_events": dict(CHUSEOK)},
    "삼성타운":   {"events": dict(XMAS), "lunar_events": {}},
    "메세나폴리스": {"events": dict(XMAS), "lunar_events": dict(SEOLLAL)},
    "광화문":     {"events": dict(XMAS), "lunar_events": {}},
}


# ---------------------------------------------------------------------------
# leakage-safe windowed backtest (expanding_window_backtest 복제 + train slice 교체)
# ---------------------------------------------------------------------------

def windowed_backtest(
    df: pd.DataFrame,
    *,
    window_days: int,
    target_col: str = TARGET,
    n_folds: int = MAIN_FOLDS,
    horizon_days: int = HORIZON,
    production_q: float = PROD_Q,
    events: dict[str, tuple[int, int]] | None = None,
    lunar_events: dict | None = None,
) -> BacktestResult:
    """category_total.expanding_window_backtest 와 동일하되 train slice 만 교체.

    train = df[(date < test_start_date) & (date >= test_start_date - window_days)].
    test slicing (row 기반 iloc) 은 원본과 동일 → anchor 재현 가능.
    """
    df = df.sort_values("date").reset_index(drop=True).dropna(subset=[target_col]).copy()
    df = df.dropna().reset_index(drop=True)
    total = len(df)
    test_size = horizon_days
    if total <= n_folds * test_size + MIN_TRAIN_ROWS:
        raise ValueError(f"Not enough data: total={total}, folds={n_folds}")

    window = pd.Timedelta(days=window_days)
    folds, preds = [], []
    for k in range(n_folds):
        test_end = total - k * test_size
        test_start = test_end - test_size
        test_df = df.iloc[test_start:test_end]
        test_start_date = test_df["date"].iloc[0]
        # === 유일한 변경점: train slice 를 날짜 기반 rolling window 로 ===
        train_df = df[(df["date"] < test_start_date) & (df["date"] >= test_start_date - window)]
        if len(train_df) < MIN_TRAIN_ROWS:
            continue
        model = fit_category_total(
            train_df, target_col=target_col,
            alpha_demand=ALPHA, production_q=production_q,
        )
        exp_pred = model.predict_expected(test_df)
        prod_pred = model.predict_production(test_df)
        # 특수일 레벨-앵커 prior: pre-test 전체 history로 fit (train window보다 길게, leakage-safe)
        hist = df[df["date"] < test_start_date]
        prior = EventLevelPrior(events=events, lunar_events=lunar_events).fit(hist, target_col=target_col)
        exp_pred, prod_pred = prior.blend(test_df["date"].values, exp_pred, prod_pred)
        actual = test_df[target_col].values
        wape = np.abs(actual - exp_pred).sum() / max(np.abs(actual).sum(), 1)
        folds.append(dict(
            fold=k, n_train=len(train_df), n_test=len(test_df),
            test_start=test_start_date, test_end=test_df["date"].iloc[-1],
            wape=wape,
            wpe=(exp_pred - actual).sum() / max(actual.sum(), 1),
            prod_pct_under=(prod_pred < actual).mean(),
        ))
        preds.append(pd.DataFrame({
            "date": test_df["date"].values, "fold": k,
            "actual": actual, "expected": exp_pred, "production": prod_pred,
        }))
    return BacktestResult(
        folds=pd.DataFrame(folds).sort_values("fold").reset_index(drop=True),
        predictions=pd.concat(preds, ignore_index=True),
    )


def metrics_from_preds(p: pd.DataFrame) -> dict:
    actual, expected, prod = p["actual"], p["expected"], p["production"]
    surplus = (prod - actual).clip(lower=0)
    return {
        "n_test": int(len(p)),
        "wape": float(np.abs(actual - expected).sum() / max(np.abs(actual).sum(), 1)),
        "wpe": float((expected - actual).sum() / max(actual.sum(), 1)),
        "stockout_risk": float((prod < actual).mean()),
        "surplus_mean_units": float(surplus.mean()),
        "surplus_rate": float(surplus.sum() / max(actual.sum(), 1)),
    }


# ---------------------------------------------------------------------------
# per-store data build
# ---------------------------------------------------------------------------

@dataclass
class StoreData:
    label: str
    cd_code: str
    store_id: str
    cd_df: pd.DataFrame          # build_category_daily .df (adjusted_demand_unit 등)
    feat: pd.DataFrame           # baseline features
    feat_variant: pd.DataFrame   # baseline + trend_ratio
    raw_incl_bulk: pd.DataFrame  # date, sold_units_incl_bulk (bulk 포함)


def _category_total_sold(daily_raw: pd.DataFrame) -> pd.DataFrame:
    """daily_raw → filter_seasonal + TARGET_CATEGORIES 합 (date, sold_units)."""
    d = filter_seasonal(daily_raw)
    d = d[d["category_id"].isin(TARGET_CATEGORIES)].copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.groupby("date")["sold_units"].sum().reset_index()


def load_prod_waste(cd_code: str) -> pd.DataFrame:
    """inventory.parquet → 해당 매장 TARGET_CATEGORIES 일별 실제 생산량(QT_MADE)·폐기량(QT_OUT).

    기존 운영 baseline(사람이 실제로 만든 양·버린 양) — 우리 모델 발주와 비교용. 값싼 조회라
    compute 캐시에 안 넣고 render 단계에서 직접 로드한다.
    """
    inv = pd.read_parquet(V2 / "inventory.parquet")
    inv = inv[inv["CD_PARTNER"].astype(str) == cd_code].copy()
    inv["date"] = pd.to_datetime(inv["DT_SALE"].astype(str))
    inv["item_id"] = inv["CD_ITEM"].astype(str)
    inv["QT_MADE"] = pd.to_numeric(inv["QT_MADE"], errors="coerce").fillna(0)
    inv["QT_OUT"] = pd.to_numeric(inv["QT_OUT"], errors="coerce").fillna(0)
    cat = item_category_map()
    inv["category_id"] = inv["item_id"].map(cat).fillna("etc")
    inv = inv[inv["category_id"].isin(TARGET_CATEGORIES)]
    g = inv.groupby("date").agg(production=("QT_MADE", "sum"), waste=("QT_OUT", "sum")).reset_index()
    return g


def add_trend_ratio(feat: pd.DataFrame) -> pd.DataFrame:
    """trend_ratio = roll_mean_28 / roll_mean_180 (shift(1), leakage-safe).

    div-by-zero / inf → 1.0. front warmup(180) 만 NaN 로 남겨 dropna 가 앞만 자름.
    """
    d = feat.sort_values("date").reset_index(drop=True).copy()
    shifted = d[TARGET].shift(1)
    rm28 = shifted.rolling(28, min_periods=28).mean()
    rm180 = shifted.rolling(180, min_periods=180).mean()
    tr = rm28 / rm180
    tr = tr.replace([np.inf, -np.inf], 1.0)
    d["trend_ratio"] = tr
    return d


def build_store_data(cd_code: str, store_id: str, label: str) -> StoreData:
    print(f"  [{label}] build_store_daily / category_daily / features ...")
    daily_excl = build_store_daily(cd_code, store_id, exclude_bulk=True)
    daily_incl = build_store_daily(cd_code, store_id, exclude_bulk=False)
    closing = build_store_closing_rows(cd_code)
    cd = build_category_daily(daily_raw=daily_excl, discount_rows=closing, alpha=ALPHA)
    feat = build_features(cd, target_col=TARGET)
    feat_variant = add_trend_ratio(feat)

    # consistency: bulk-excluded category total == cd.df.sold_total_unit (동일 구성)
    excl_total = _category_total_sold(daily_excl).rename(columns={"sold_units": "b"})
    chk = cd.df[["date", "sold_total_unit"]].merge(excl_total, on="date", how="inner")
    assert np.allclose(chk["sold_total_unit"], chk["b"]), f"{label}: layer(b) != sold_total_unit"

    raw_incl = _category_total_sold(daily_incl).rename(columns={"sold_units": "sold_units_incl_bulk"})
    return StoreData(label, cd_code, store_id, cd.df, feat, feat_variant, raw_incl)


def feasible_folds(feat: pd.DataFrame) -> int:
    total = len(feat.dropna())
    return int(min(MAIN_FOLDS, max(0, (total - MIN_TRAIN_ROWS) // HORIZON)))


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------

def fig_to_img(fig, height: int = 420) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:100%;height:auto;"/>'


# ---------------------------------------------------------------------------
# 안전마진·버퍼 분석 (margin_qmat_{sid}.parquet 사용 — margin_optimize.py collect 산출)
# ---------------------------------------------------------------------------
Q_GRID_MARGIN = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
ADD_SWEEP = [0, 5, 10, 15, 20, 25, 30]
MULT_SWEEP = [1.00, 1.05, 1.10, 1.15, 1.20]
BASE_QS = [(0.85, 7), (0.95, 9)]   # (분위수, Q_GRID_MARGIN 열 index)
W_PRIMARY = 10.0                    # 부족(매진) 가중 — 매진 회피 선호 반영
Q_FINE_OPT = np.round(np.arange(0.50, 0.951, 0.01), 3)
MULT_OPT = np.round(np.arange(1.00, 1.201, 0.02), 3)
ADD_OPT = np.arange(0, 31, 2, dtype=float)


def _margin_qmat(store_id: str):
    """저장된 qmat parquet 로드 → (단조 qmat, actual, baseline_adj) 또는 None."""
    path = Path(f"reports/margin_qmat_{store_id}.parquet")
    if not path.exists():
        return None
    p = pd.read_parquet(path)
    qcols = [f"q{q:.2f}" for q in Q_GRID_MARGIN]
    qmat = np.maximum.accumulate(p[qcols].to_numpy(), axis=1)
    return qmat, p["actual"].to_numpy(), p["baseline_adj"].to_numpy()


def _margin_full(store_id: str):
    """qmat + dates → dict(dates, qmat[단조], actual, base) 또는 None."""
    path = Path(f"reports/margin_qmat_{store_id}.parquet")
    if not path.exists():
        return None
    p = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
    qcols = [f"q{q:.2f}" for q in Q_GRID_MARGIN]
    return {"dates": pd.to_datetime(p["date"]).to_numpy(),
            "qmat": np.maximum.accumulate(p[qcols].to_numpy(), axis=1),
            "actual": p["actual"].to_numpy(), "base": p["baseline_adj"].to_numpy()}


def _pred_at_q_m(qmat: np.ndarray, q: float) -> np.ndarray:
    idx = np.interp(q, Q_GRID_MARGIN, np.arange(len(Q_GRID_MARGIN)))
    lo, hi = int(np.floor(idx)), int(np.ceil(idx))
    frac = idx - lo
    return qmat[:, lo] * (1 - frac) + qmat[:, hi] * frac


def _order(qmat: np.ndarray, cfg: dict) -> np.ndarray:
    return _pred_at_q_m(qmat, cfg["q"]) * cfg["mult"] + cfg["add"]


def _max_streak(mask) -> int:
    best = cur = 0
    for x in mask:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return int(best)


# 연속 매진 제약: "일수"가 아니라 연속 구간의 누적 부족량 기준 (이월재고 흡수 관점).
# 임계 = 일평균 수요 × 이 비율 (가용 이월재고 데이터가 없어 비즈니스 허용치로 설정).
MAX_RUN_SHORTFALL_FRAC = 0.33


def _max_run_shortfall(short, so) -> float:
    """연속 매진 구간(so=True 연속)의 누적 부족량 중 최대값."""
    best = cur = 0.0
    for x, f in zip(short, so):
        cur = cur + float(x) if f else 0.0
        best = max(best, cur)
    return float(best)


def find_optimal_q_daily(store_id: str, w: float = W_PRIMARY) -> dict | None:
    """daily 생산/폐기 관점 — 일별 가중비용(w·부족+잉여) 최소 q. q만(mult=1,add=0). calib 선택."""
    m = _margin_full(store_id)
    if m is None:
        return None
    qmat, actual = m["qmat"], m["actual"]
    mid = len(actual) // 2
    qc, ac = qmat[:mid], actual[:mid]
    best = None
    for q in Q_FINE_OPT:
        o = _pred_at_q_m(qc, q)
        cost = w * np.clip(ac - o, 0, None).sum() + np.clip(o - ac, 0, None).sum()
        if best is None or cost < best[0]:
            best = (cost, float(q))
    return {"q": best[1], "mult": 1.0, "add": 0.0, "kind": "daily", "w": w}


def find_optimal_q_weekly(store_id: str, w: float = W_PRIMARY, *, max_streak: int = 2) -> dict | None:
    """weekly 관점 — '주간 생산만 맞추면 된다' 가정. 주간 합산 가중비용(w·주간부족+주간잉여)
    최소 q 를, 연속매진≤max_streak 제약 안에서 선택(폐기 최소 지향). 제약 만족 q 없으면
    제약 무시하고 weekly 비용 최소 q + flag. calib 선택."""
    m = _margin_full(store_id)
    if m is None:
        return None
    qmat, actual, dates = m["qmat"], m["actual"], pd.to_datetime(m["dates"])
    mid = len(actual) // 2
    qc, ac, dc = qmat[:mid], actual[:mid], dates[:mid]
    rows = []
    for q in Q_FINE_OPT:
        o = _pred_at_q_m(qc, q)
        st = _max_streak(ac > o)
        wk = pd.DataFrame({"date": dc, "d": ac, "o": o}).set_index("date").resample(
            "W-MON").sum(min_count=1).dropna()
        wcost = w * np.clip(wk["d"] - wk["o"], 0, None).sum() + np.clip(wk["o"] - wk["d"], 0, None).sum()
        rows.append((float(q), float(wcost), st))
    safe = [(q, wc) for q, wc, st in rows if st <= max_streak]
    if safe:
        # 제약 만족 q 중 weekly 비용 최소 (= 폐기 최소 지향, 연속매진 안전선 내)
        q, ok = min(safe, key=lambda x: x[1])[0], True
    else:
        # 완화 불가 — 제약 못 지키면 낮추면 안 됨. 가장 안전(최소 streak, 동률이면 최고 q)으로.
        q = float(min(rows, key=lambda r: (r[2], -r[0]))[0])
        ok = False
    return {"q": q, "mult": 1.0, "add": 0.0, "kind": "weekly", "w": w,
            "max_streak": max_streak, "feasible": ok}


# q=0.5(median) 고정 + N·K 버퍼 최적화 그리드 (q0.95 spread 병리 회피 — median 기반)
QMED = 0.50
K_GRID_NK = np.round(np.arange(1.00, 1.41, 0.02), 2)   # 정률곱 ×K
N_GRID_NK = np.arange(0, 41, 2, dtype=float)           # 정률합 +N
WEEKLY_WS = (5.0, 6.0, 7.0, 8.0, 10.0)


def find_opt_NK_daily(store_id: str, w: float = W_PRIMARY) -> dict | None:
    """q=0.5 고정, daily 가중비용(w·부족+잉여) 최소 (K,N). calib 선택."""
    m = _margin_full(store_id)
    if m is None:
        return None
    qmat, actual = m["qmat"], m["actual"]
    mid = len(actual) // 2
    pred = _pred_at_q_m(qmat[:mid], QMED); ac = actual[:mid]
    best = None
    for k in K_GRID_NK:
        pm = pred * k
        for nn in N_GRID_NK:
            o = pm + nn
            cost = w * np.clip(ac - o, 0, None).sum() + np.clip(o - ac, 0, None).sum()
            if best is None or cost < best[0]:
                best = (cost, float(k), float(nn))
    return {"q": QMED, "mult": best[1], "add": best[2], "kind": "daily", "w": w}


def find_opt_NK_weekly(store_id: str, *, run_frac: float = MAX_RUN_SHORTFALL_FRAC,
                       min_cover: float = 1.0) -> dict | None:
    """q=0.5 고정. '주간 생산만 맞추면 된다' → 폐기 최소(발주 최저)를 목표로 하되
    '연속 매진 구간 누적부족 ≤ run_frac×일평균수요' & 주간커버≥min_cover 제약을
    (K,N) 전그리드에서 직접 탐색. (연속 '일수'가 아니라 누적 '수량' 기준 — 이월재고 흡수 관점.)
    안전 후보 없으면 최소 누적부족 fallback. calib 선택."""
    m = _margin_full(store_id)
    if m is None:
        return None
    qmat, actual, dates = m["qmat"], m["actual"], pd.to_datetime(m["dates"])
    mid = len(actual) // 2
    pred, ac, dc = _pred_at_q_m(qmat[:mid], QMED), actual[:mid], dates[:mid]
    cap = run_frac * float(ac.mean())   # 누적부족 임계 (개)
    g = pd.DataFrame({"d": dc, "pred": pred, "act": ac}).groupby(
        pd.PeriodIndex(dc, freq="W-MON")).agg(sp=("pred", "sum"), sa=("act", "sum"), nd=("pred", "size"))
    wps, was, wnd = g["sp"].to_numpy(), g["sa"].to_numpy(), g["nd"].to_numpy()

    best_safe = None   # (mean_order, k, n) — 제약 만족 중 발주 최저
    fallback = None    # ((max_run_shortfall, mean_order), k, n)
    for k in K_GRID_NK:
        pm = pred * k; wpk = k * wps
        for nn in N_GRID_NK:
            o = pm + nn
            mo = float(o.mean())
            mrs = _max_run_shortfall(np.clip(ac - o, 0, None), ac > o)
            cover = float(((wpk + nn * wnd) >= was).mean())
            if mrs <= cap and cover >= min_cover:
                if best_safe is None or mo < best_safe[0]:
                    best_safe = (mo, float(k), float(nn))
            key = (mrs, mo)
            if fallback is None or key < fallback[0]:
                fallback = (key, float(k), float(nn))
    if best_safe is not None:
        k, nn, ok = best_safe[1], best_safe[2], True
    else:
        k, nn, ok = fallback[1], fallback[2], False
    return {"q": QMED, "mult": k, "add": nn, "kind": "weekly", "feasible": ok,
            "run_cap": cap, "run_frac": run_frac}


def _cfg_holdout_stats(store_id: str, cfg: dict) -> dict:
    m = _margin_full(store_id)
    qmat, actual, dates, base = m["qmat"], m["actual"], pd.to_datetime(m["dates"]), m["base"]
    n = len(actual); mid = n // 2; hold = slice(mid, n)
    o = _order(qmat, cfg)[hold]; a = actual[hold]; d = dates[hold]; b = base[hold]
    so = a > o
    wk = pd.DataFrame({"date": d, "d": a, "o": o}).set_index("date").resample(
        "W-MON").sum(min_count=1).dropna()
    mb = ~np.isnan(b)
    b_surp = float(np.clip(b[mb] - a[mb], 0, None).sum())
    surp = float(np.clip(o[mb] - a[mb], 0, None).sum())
    return {"q": cfg["q"], "so_rate": float(so.mean()), "max_streak": _max_streak(so),
            "max_run_short": _max_run_shortfall(np.clip(a - o, 0, None), so),
            "cover": float((wk["o"] >= wk["d"]).mean()),
            "waste_red_pct": (b_surp - surp) / b_surp * 100 if b_surp else 0.0,
            "tot_short": float(np.clip(a - o, 0, None).sum())}


def find_optimal_config(store_id: str, w: float = W_PRIMARY) -> dict | None:
    """calib(앞 절반)서 가중비용 최소 (q,mult,add) 선택 → holdout 성능 병기(정직한 OOS).
    3손잡이 joint 그리드. w=부족 가중."""
    m = _margin_full(store_id)
    if m is None:
        return None
    qmat, actual = m["qmat"], m["actual"]
    n = len(actual); mid = n // 2
    qc, ac = qmat[:mid], actual[:mid]
    best = None
    predq = {q: _pred_at_q_m(qc, q) for q in Q_FINE_OPT}
    for q in Q_FINE_OPT:
        for mult in MULT_OPT:
            pm = predq[q] * mult
            for add in ADD_OPT:
                order = pm + add
                cost = w * np.clip(ac - order, 0, None).sum() + np.clip(order - ac, 0, None).sum()
                if best is None or cost < best[0]:
                    best = (cost, {"q": float(q), "mult": float(mult), "add": float(add)})
    cfg = best[1]
    return {"cfg": cfg, "w": w, "n_calib": mid, "n_holdout": n - mid}


def _buffer_row(order, actual):
    short = np.clip(actual - order, 0, None)
    surp = np.clip(order - actual, 0, None)
    so = actual > order
    return {"so_rate": float(so.mean()), "n_so": int(so.sum()),
            "tot_short": float(short.sum()), "tot_surp": float(surp.sum())}


def buffer_analysis(store_id: str) -> dict | None:
    m = _margin_qmat(store_id)
    if m is None:
        return None
    qmat, actual, base = m
    mask = ~np.isnan(base)
    b_short = float(np.clip(actual[mask] - base[mask], 0, None).sum())
    b_surp = float(np.clip(base[mask] - actual[mask], 0, None).sum())
    existing = {"so_rate": float((actual[mask] > base[mask]).mean()),
                "tot_short": b_short, "tot_surp": b_surp,
                "mean_order": float(base[mask].mean()), "mean_actual": float(actual[mask].mean())}
    curves, curves_mult = {}, {}
    for q, qi in BASE_QS:
        pred = qmat[:, qi]
        rows = []
        for add in ADD_SWEEP:
            r = _buffer_row(pred + add, actual)
            r["add"] = add
            r["waste_red_pct"] = (b_surp - r["tot_surp"]) / b_surp * 100 if b_surp else 0
            r["short_inc"] = r["tot_short"] - b_short
            rows.append(r)
        curves[f"{q:.2f}"] = rows
        rowsm = []
        for k in MULT_SWEEP:
            r = _buffer_row(pred * k, actual)
            r["mult"] = k
            r["waste_red_pct"] = (b_surp - r["tot_surp"]) / b_surp * 100 if b_surp else 0
            r["short_inc"] = r["tot_short"] - b_short
            rowsm.append(r)
        curves_mult[f"{q:.2f}"] = rowsm

    # 진단: 예측(q0.5) 수준별 매진 발생 — ×K(레벨비례) 정당성 확인
    pred50 = qmat[:, 0]
    order085 = qmat[:, 7]
    t = np.quantile(pred50, [1 / 3, 2 / 3])
    grp = np.digitize(pred50, t)   # 0=낮음,1=중간,2=높음
    diag = []
    for g, name in [(0, "예측 낮음"), (1, "예측 중간"), (2, "예측 높음")]:
        sel = grp == g
        so = actual[sel] > order085[sel]
        sf = np.clip(actual[sel] - order085[sel], 0, None)
        diag.append({"grp": name, "n": int(sel.sum()), "so_rate": float(so.mean()),
                     "mean_sf_on_so": float(sf[so].mean()) if so.any() else 0.0,
                     "mean_pred": float(pred50[sel].mean())})
    return {"n_days": int(len(actual)), "existing": existing,
            "curves": curves, "curves_mult": curves_mult, "diag": diag}


def _cfg_label(cfg: dict) -> str:
    s = f"q{cfg['q']:.2f}"
    if abs(cfg["mult"] - 1.0) > 1e-9:
        s += f"×{cfg['mult']:.2f}"
    if cfg["add"] > 0:
        s += f"+{cfg['add']:.0f}"
    return s


def _biz_box(biz: dict, ndays: int) -> str:
    """비즈니스 요약 박스 문구 — 폐기 절감/증가·부족 기존→현재·매진율 명확 표기."""
    wr = biz["waste_red_pct"]
    waste_txt = f"{wr:.0f}% 절감" if wr >= 0 else f"{abs(wr):.0f}% 증가"
    exist_short = biz["short"] - biz["short_inc"]
    return (f"[홀드아웃 {ndays}일] 폐기 {biz['surp']:.0f} (기존比 {waste_txt}) · "
            f"부족 {exist_short:.0f}→{biz['short']:.0f} (+{biz['short_inc']:.0f}) · "
            f"매진율 {biz['existing_so']*100:.0f}%→{biz['so_rate']*100:.0f}%")


def _biz_metrics(order, actual, base, sl) -> dict:
    """slice sl 구간에서 order 정책의 매진/폐기 + 기존 발주 대비."""
    o, a, b = order[sl], actual[sl], base[sl]
    short = float(np.clip(a - o, 0, None).sum()); surp = float(np.clip(o - a, 0, None).sum())
    so_rate = float((a > o).mean())
    mb = ~np.isnan(b)
    b_short = float(np.clip(a[mb] - b[mb], 0, None).sum()); b_surp = float(np.clip(b[mb] - a[mb], 0, None).sum())
    b_so = float((a[mb] > b[mb]).mean())
    return {"short": short, "surp": surp, "so_rate": so_rate,
            "waste_red_pct": (b_surp - surp) / b_surp * 100 if b_surp else 0.0,
            "short_inc": short - b_short, "so_inc_pp": (so_rate - b_so) * 100,
            "existing_so": b_so, "existing_surp": b_surp}


def plot_order_daily(store_id: str, cfg: dict, ref_cfg: dict | None, tag: str) -> str:
    """일별 실수요 vs 발주(cfg) vs 기존 발주. ref_cfg 있으면 점선으로 병기. 비즈니스 요약 박스."""
    m = _margin_full(store_id)
    label = LABEL_BY_SID[store_id]; color = STORE_COLORS[label]
    dates, qmat, actual, base = m["dates"], m["qmat"], m["actual"], m["base"]
    n = len(actual); mid = n // 2
    order = _order(qmat, cfg)
    win = slice(max(0, n - OVERLAY_DAYS), n)
    hold = slice(mid, n)
    biz = _biz_metrics(order, actual, base, hold)

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(dates[win], actual[win], color="#2A2A2A", lw=1.7, label="실수요 (adjusted)")
    ax.plot(dates[win], order[win], color=color, lw=1.9, label=f"발주 {_cfg_label(cfg)}")
    if ref_cfg is not None:
        oref = _order(qmat, ref_cfg)
        ax.plot(dates[win], oref[win], color=color, lw=1.2, ls=":", alpha=0.8,
                label=f"발주 {_cfg_label(ref_cfg)} (버퍼 미적용)")
    ax.plot(dates[win], base[win], color="#d62728", lw=1.2, ls="--", alpha=0.75, label="기존 아띠제 발주")
    ax.text(0.01, 0.98, _biz_box(biz, n - mid), transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#ccc"))
    ax.set_title(f"{label} — 일별 발주 {_cfg_label(cfg)} vs 실수요 vs 기존 [{tag}]", fontsize=12)
    ax.set_ylabel("일 수량")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    return fig_to_img(fig)


def plot_short_surp(store_id: str, cfg: dict, ref_cfg: dict | None, tag: str) -> str:
    """발주(cfg) 대비 실현 잉여(위)/부족(아래). ref_cfg 점선 병기. 비즈니스 요약."""
    m = _margin_full(store_id)
    label = LABEL_BY_SID[store_id]; color = STORE_COLORS[label]
    dates, qmat, actual, base = m["dates"], m["qmat"], m["actual"], m["base"]
    n = len(actual); mid = n // 2
    order = _order(qmat, cfg)
    win = slice(max(0, n - OVERLAY_DAYS), n)
    hold = slice(mid, n)
    surp = np.clip(order - actual, 0, None); short = np.clip(actual - order, 0, None)
    biz = _biz_metrics(order, actual, base, hold)

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(dates[win], surp[win], color=color, alpha=0.55, width=1.0, label="실현 잉여/폐기 (위)")
    ax.bar(dates[win], -short[win], color="#d62728", alpha=0.60, width=1.0, label="실현 부족/매진분 (아래)")
    if ref_cfg is not None:
        oref = _order(qmat, ref_cfg)
        sref = np.clip(oref - actual, 0, None); shref = np.clip(actual - oref, 0, None)
        ax.plot(dates[win], sref[win], color="#555", lw=1.0, ls=":", label=f"{_cfg_label(ref_cfg)} 잉여")
        ax.plot(dates[win], -shref[win], color="#900", lw=1.0, ls=":", label=f"{_cfg_label(ref_cfg)} 부족")
    ax.axhline(0, color="#888", lw=0.8)
    ax.text(0.01, 0.98, _biz_box(biz, n - mid), transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#ccc"))
    ax.set_title(f"{label} — 발주 {_cfg_label(cfg)} 매진/폐기 [{tag}]", fontsize=12)
    ax.set_ylabel("발주 대비 (위=폐기, 아래=매진)")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    return fig_to_img(fig)


def plot_weekly_supply_demand(store_id: str, cfg: dict) -> str:
    """연간(전 OOS) 주단위: 기존 생산량 vs 기존 수요량 vs 우리 최적 발주.

    주단위로 합쳐 이월재고 효과를 흡수 — 하루 매진 위험이 있어도 주 총량이 수요를 덮으면
    전날/주중 재고로 커버 가능. 스케일 통일 위해 기존 생산량은 adjusted 스케일(baseline_adj)."""
    m = _margin_full(store_id)
    label = LABEL_BY_SID[store_id]; color = STORE_COLORS[label]
    df = pd.DataFrame({"date": pd.to_datetime(m["dates"]), "demand": m["actual"],
                       "existing": m["base"], "ours": _order(m["qmat"], cfg)})
    wk = df.set_index("date").resample("W-MON").sum(min_count=1).dropna()

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(wk.index, wk["existing"], color="#d62728", lw=1.6, ls="--", marker="o", ms=3,
            label="기존 생산량 (adjusted 스케일)")
    ax.plot(wk.index, wk["demand"], color="#2A2A2A", lw=2.0, marker="o", ms=3, label="기존 수요량 (실측)")
    ax.plot(wk.index, wk["ours"], color=color, lw=1.9, marker="o", ms=3,
            label=f"우리 최적 발주 {_cfg_label(cfg)}")
    cover = float((wk["ours"] >= wk["demand"]).mean())
    prod_cut = (wk["existing"].sum() - wk["ours"].sum()) / wk["existing"].sum() * 100
    ax.text(0.01, 0.98,
            f"주간 우리발주 ≥ 주간수요: {cover*100:.0f}% 주 · 기존 대비 생산 {prod_cut:+.0f}% · "
            f"주 평균 수요 {wk['demand'].mean():.0f} / 우리 {wk['ours'].mean():.0f} / 기존 {wk['existing'].mean():.0f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#ccc"))
    ax.set_title(f"{label} — 연간 주단위: 기존 생산 vs 수요 vs 우리 최적 발주 ({len(wk)}주)", fontsize=12)
    ax.set_ylabel("주 수량")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    return fig_to_img(fig)


def plot_buffer_frontier(store_id: str, ba: dict) -> str:
    color = STORE_COLORS[LABEL_BY_SID[store_id]]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    styles = {"0.85": ("-", color), "0.95": ("--", "#8e44ad")}
    for qk, rows in ba["curves"].items():
        ls, c = styles[qk]
        xs = [r["so_rate"] * 100 for r in rows]
        ys = [r["waste_red_pct"] for r in rows]
        ax.plot(xs, ys, ls, color=c, marker="o", lw=1.8, label=f"base q{qk}")
        for r in rows:
            if r["add"] in (0, 10, 20, 30):
                ax.annotate(f"+{r['add']}", (r["so_rate"] * 100, r["waste_red_pct"]),
                            fontsize=7, ha="left", va="bottom")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("매진일 비율 % (낮을수록 안전)")
    ax.set_ylabel("기존 대비 폐기 절감 % (높을수록 이득)")
    ax.set_title(f"{LABEL_BY_SID[store_id]} — 안전마진 frontier (매진↓ ↔ 폐기절감↓ trade-off)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    return fig_to_img(fig)


def buffer_table_html(ba: dict) -> str:
    ex = ba["existing"]
    h = (f'<p class="desc">기존 아띠제 발주(adjusted): 매진일 {ex["so_rate"]*100:.0f}% · '
         f'총폐기 {ex["tot_surp"]:.0f} · 평균발주 {ex["mean_order"]:.0f} vs 평균실수요 {ex["mean_actual"]:.0f} '
         f'(과잉생산). 아래는 우리 발주 = q예측 + 버퍼(add).</p>')
    for qk, rows in ba["curves"].items():
        h += (f'<b>base q{qk} · 정률합(+N)</b><table class="mtab"><tr>'
              '<th>버퍼(+N)</th><th>매진일%</th><th>매진일수</th><th>총부족</th><th>총폐기</th>'
              '<th>기존대비 폐기절감</th><th>기존대비 부족증가</th></tr>')
        for r in rows:
            h += (f'<tr><td>+{r["add"]}</td><td>{r["so_rate"]*100:.0f}%</td><td>{r["n_so"]}</td>'
                  f'<td>{r["tot_short"]:.0f}</td><td>{r["tot_surp"]:.0f}</td>'
                  f'<td>{r["waste_red_pct"]:.0f}%</td><td>+{r["short_inc"]:.0f}</td></tr>')
        h += "</table>"
    for qk, rows in ba.get("curves_mult", {}).items():
        h += (f'<b>base q{qk} · 정률곱(×K)</b><table class="mtab"><tr>'
              '<th>버퍼(×K)</th><th>매진일%</th><th>매진일수</th><th>총부족</th><th>총폐기</th>'
              '<th>기존대비 폐기절감</th><th>기존대비 부족증가</th></tr>')
        for r in rows:
            h += (f'<tr><td>×{r["mult"]:.2f}</td><td>{r["so_rate"]*100:.0f}%</td><td>{r["n_so"]}</td>'
                  f'<td>{r["tot_short"]:.0f}</td><td>{r["tot_surp"]:.0f}</td>'
                  f'<td>{r["waste_red_pct"]:.0f}%</td><td>+{r["short_inc"]:.0f}</td></tr>')
        h += "</table>"
    if ba.get("diag"):
        h += ('<p class="desc"><b>진단 — ×K(레벨비례) vs +N(정액) 어느 버퍼가 맞나</b>: '
              '예측(q0.5) 수준별 q0.85 발주의 매진 발생. 예측 높은 날 매진율/부족이 더 크면 ×K가 유리.</p>'
              '<table class="mtab"><tr><th>예측 그룹</th><th>일수</th><th>평균예측</th>'
              '<th>매진일%</th><th>매진일 평균부족</th></tr>')
        for d in ba["diag"]:
            h += (f'<tr><td>{d["grp"]}</td><td>{d["n"]}</td><td>{d["mean_pred"]:.0f}</td>'
                  f'<td>{d["so_rate"]*100:.0f}%</td><td>{d["mean_sf_on_so"]:.1f}</td></tr>')
        h += "</table>"
    return h


def plot_four_layer(sd: StoreData, main_preds: pd.DataFrame, pw: pd.DataFrame) -> str:
    test_end = main_preds["date"].max()
    start = test_end - pd.Timedelta(days=OVERLAY_DAYS)
    color = STORE_COLORS[sd.label]

    a = sd.raw_incl_bulk
    b = sd.cd_df[["date", "sold_total_unit"]]
    c = sd.cd_df[["date", TARGET]]
    d = main_preds[["date", "expected"]]
    e = pw[["date", "production"]]   # 기존 실제 생산량(QT_MADE)

    def win(df, col):
        # 반드시 날짜순 정렬 — main_preds는 fold 순서(최근→과거)라 정렬 안 하면 선이 꼬인다.
        m = (df["date"] >= start) & (df["date"] <= test_end)
        sub = df.loc[m, ["date", col]].sort_values("date")
        return sub["date"], sub[col]

    fig, ax = plt.subplots(figsize=(13, 4.6))
    # 겹침 완화: 실수요(c)·모델예측(d)만 선, 원판매(a)·bulk제외(b)는 마커로.
    ax.scatter(*win(a, "sold_units_incl_bulk"), color="#bdbdbd", s=13, alpha=0.55,
               marker="o", label="(a) 원판매 (bulk 포함)")
    ax.scatter(*win(b, "sold_total_unit"), color="#7f8c8d", s=22, alpha=0.75,
               marker="x", label="(b) bulk 제외 판매")
    ax.plot(*win(c, TARGET), color=color, lw=1.9, label="(c) 실수요 adjusted (α=0.8)")
    ax.plot(*win(d, "expected"), color="#000000", lw=1.7, ls="--", label="(d) 모델 예측 (q0.5)")
    ax.scatter(*win(e, "production"), color="#d62728", s=20, alpha=0.7, marker="D",
               label="(e) 기존 실제 생산량 (QT_MADE)")
    ax.set_title(f"{sd.label} — 4-layer 카테고리 합 일별 (최근 {OVERLAY_DAYS}일)", fontsize=12)
    ax.set_ylabel("일 수량")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    return fig_to_img(fig)


def plot_risk(sd: StoreData, main_preds: pd.DataFrame, pw: pd.DataFrame) -> str:
    p = main_preds.sort_values("date")
    resid = (p["actual"] - p["expected"]).values  # pooled residuals
    test_end = p["date"].max()
    start = test_end - pd.Timedelta(days=OVERLAY_DAYS)
    w = p[(p["date"] >= start) & (p["date"] <= test_end)].copy()
    w = w.merge(pw[["date", "waste"]], on="date", how="left")   # 기존 실제 폐기량(QT_OUT)

    exp = w["expected"].values[:, None]
    prod = w["production"].values[:, None]
    draws = exp + resid[None, :]                       # actual ~ expected + pooled residual
    w["stockout_risk"] = (draws > prod).mean(axis=1)
    w["expected_waste"] = np.clip(prod - draws, 0, None).mean(axis=1)
    # 실현 결과: 잉여(발주>실측, 폐기) 위쪽 / 부족(실측>발주, 매진분) 아래쪽
    w["realized_waste"] = (w["production"] - w["actual"]).clip(lower=0)
    w["realized_shortfall"] = (w["actual"] - w["production"]).clip(lower=0)

    color = STORE_COLORS[sd.label]
    fig, ax1 = plt.subplots(figsize=(13, 4.6))
    ax1.bar(w["date"], w["realized_waste"], color=color, alpha=0.55, width=1.0,
            label="실현 잉여/폐기 (발주-실측, 위)")
    ax1.bar(w["date"], -w["realized_shortfall"], color="#d62728", alpha=0.60, width=1.0,
            label="실현 부족/매진분 (실측-발주, 아래)")
    ax1.plot(w["date"], w["expected_waste"], color=color, lw=1.0, ls=":", alpha=0.8,
             label="기대 잉여(폐기) 수량 (q0.85)")
    ax1.plot(w["date"], w["waste"], color="#8888aa", lw=1.3, alpha=0.9,
             label="기존 실제 폐기량 (QT_OUT)")
    ax1.axhline(0, color="#888", lw=0.8)
    ax1.set_ylabel("발주 대비 수량 (위=잉여/폐기, 아래=부족)")

    ax2 = ax1.twinx()
    ax2.plot(w["date"], w["stockout_risk"], color="#d62728", lw=1.6, alpha=0.85,
             label="매진 위험 예측 P(실수요>발주)")
    ax2.set_ylabel("매진 위험 확률", color="#d62728")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, ncol=2, loc="upper left")
    ax1.set_title(f"{sd.label} — 날짜별 매진위험 / 기대 잉여 (q0.85 발주)", fontsize=12)
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    return fig_to_img(fig)


def plot_vs_baseline(sd: StoreData, main_preds: pd.DataFrame, pw: pd.DataFrame) -> str:
    """기존 대비 — 실수요(adjusted) vs 우리 발주×1.1 vs 기존 생산량(adjusted 스케일 보정).

    기존 생산량을 adjusted 스케일로 맞추기 위해, 원판매→adjusted 로 줄인 양(bulk+마감할인
    비실수요분)을 기존 QT_MADE 에서 뺀다. 실판매만 보면 우리 발주가 리스키해 보이지만,
    실제 비교 대상인 '기존 생산량'과 나란히 두면 과잉생산 절감 효과가 드러난다.
    """
    p = main_preds.sort_values("date")
    test_end = p["date"].max()
    start = test_end - pd.Timedelta(days=OVERLAY_DAYS)
    w = p[(p["date"] >= start) & (p["date"] <= test_end)][["date", "actual", "production"]].copy()
    w = w.merge(sd.raw_incl_bulk, on="date", how="left")
    w = w.merge(pw[["date", "production"]].rename(columns={"production": "qt_made"}),
                on="date", how="left")
    w["our_x11"] = w["production"] * 1.1
    w["baseline_adj"] = w["qt_made"] - (w["sold_units_incl_bulk"] - w["actual"])

    color = STORE_COLORS[sd.label]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(w["date"], w["actual"], color="#2A2A2A", lw=1.7, label="실수요 (adjusted, 실측)")
    ax.plot(w["date"], w["our_x11"], color=color, lw=1.8, label="우리 발주 제안 × 1.1")
    ax.plot(w["date"], w["baseline_adj"], color="#d62728", lw=1.4, ls="--", alpha=0.85,
            label="기존 생산량 (adjusted 스케일 보정)")
    ax.fill_between(w["date"], w["actual"], w["our_x11"],
                    where=(w["our_x11"] >= w["actual"]), color=color, alpha=0.10)

    d_sum, o_sum, b_sum = w["actual"].sum(), w["our_x11"].sum(), w["baseline_adj"].sum()
    over_our = o_sum / d_sum - 1 if d_sum else 0
    over_base = b_sum / d_sum - 1 if d_sum else 0
    cut = (b_sum - o_sum) / b_sum if b_sum else 0
    ax.text(0.01, 0.97,
            f"기간합계 — 실수요 {d_sum:,.0f} / 우리×1.1 {o_sum:,.0f} (+{over_our*100:.0f}%) / "
            f"기존 {b_sum:,.0f} (+{over_base*100:.0f}%)\n"
            f"→ 기존 대비 생산량 {cut*100:.0f}% 절감하면서 실수요 커버",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#ccc"))
    ax.set_title(f"{sd.label} — 기존 대비: 실수요 vs 우리 발주×1.1 vs 기존 생산", fontsize=12)
    ax.set_ylabel("일 수량")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    return fig_to_img(fig)


def plot_actual_vs_pred(sd: StoreData, main_preds: pd.DataFrame) -> str:
    p = main_preds.sort_values("date")
    color = STORE_COLORS[sd.label]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(p["date"], p["actual"], color="#2A2A2A", lw=1.3, label="실측 실수요")
    ax.plot(p["date"], p["expected"], color=color, lw=1.6, label="예측 (q0.5)")
    ax.plot(p["date"], p["production"], color="#d62728", lw=1.0, ls="--", alpha=0.7,
            label="발주점 (q0.85)")
    under = p[p["production"] < p["actual"]]
    ax.scatter(under["date"], under["actual"], color="#d62728", marker="x", s=28, zorder=5,
               label="매진 위험일 (실측>발주)")
    ax.set_title(f"{sd.label} — OOS 실측 vs 예측 (rolling 2Y, {len(p)}일)", fontsize=12)
    ax.set_ylabel("일 수량")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    return fig_to_img(fig)


def plot_wpe_over_time(sd: StoreData, main_preds: pd.DataFrame) -> str:
    p = main_preds.copy()
    p["ym"] = p["date"].dt.to_period("M").dt.to_timestamp()
    m = p.groupby("ym").apply(
        lambda g: (g["expected"] - g["actual"]).sum() / max(g["actual"].sum(), 1),
        include_groups=False,
    ).reset_index(name="wpe")
    color = STORE_COLORS[sd.label]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    bars = ax.bar(m["ym"], m["wpe"] * 100, width=20,
                  color=[color if v >= 0 else "#d62728" for v in m["wpe"]], alpha=0.75)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(f"{sd.label} — 월별 signed WPE (예측-실측)/실측, 추세 pinning 진단", fontsize=12)
    ax.set_ylabel("WPE %  (+과대 / -과소)")
    ax.grid(True, alpha=0.2, axis="y")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    return fig_to_img(fig)


def plot_window_sensitivity(window_table: dict) -> tuple[str, str]:
    fig1, ax1 = plt.subplots(figsize=(11, 4.6))
    fig2, ax2 = plt.subplots(figsize=(11, 4.6))
    for label, rows in window_table.items():
        color = STORE_COLORS[label]
        xs = [r["window_days"] for r in rows]
        ax1.plot(xs, [r["wape"] * 100 for r in rows], "o-", color=color, label=label, lw=1.8)
        ax2.plot(xs, [r["stockout_risk"] * 100 for r in rows], "o-", color=color, label=label, lw=1.8)
    for ax, ttl, ylab in [
        (ax1, "학습창(일) × WAPE — 매장별", "WAPE %"),
        (ax2, "학습창(일) × 매진위험 — 매장별", "매진위험 % (prod<actual)"),
    ]:
        ax.set_title(ttl, fontsize=12)
        ax.set_xlabel("rolling window (일, 1825≈expanding)")
        ax.set_ylabel(ylab)
        ax.set_xscale("log")
        ax.set_xticks(list(WINDOWS))
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    return fig_to_img(fig1), fig_to_img(fig2)


def plot_variant(variant_table: dict) -> str:
    labels = list(variant_table.keys())
    x = np.arange(len(labels))
    base_wape = [variant_table[l]["baseline"]["wape"] * 100 for l in labels]
    var_wape = [variant_table[l]["variant"]["wape"] * 100 for l in labels]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(x - 0.2, base_wape, 0.4, label="baseline", color="#7f8c8d")
    ax.bar(x + 0.2, var_wape, 0.4, label="+ trend_ratio", color="#1abc9c")
    for i, l in enumerate(labels):
        delta = var_wape[i] - base_wape[i]
        ax.annotate(f"{delta:+.2f}pp", (x[i], max(base_wape[i], var_wape[i]) + 0.1),
                    ha="center", fontsize=8, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WAPE %")
    ax.set_title("feature variant: baseline vs + trend_ratio (rolling 2Y)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    return fig_to_img(fig)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def metric_table(m: dict) -> str:
    return (
        '<table class="mtab"><tr>'
        "<th>WAPE</th><th>WPE(signed)</th><th>매진위험</th>"
        "<th>기대잉여/일</th><th>잉여율</th><th>N(test일)</th></tr>"
        f"<tr><td>{m['wape']*100:.2f}%</td><td>{m['wpe']*100:+.2f}%</td>"
        f"<td>{m['stockout_risk']*100:.1f}%</td><td>{m['surplus_mean_units']:.0f}개</td>"
        f"<td>{m['surplus_rate']*100:.1f}%</td><td>{m['n_test']}</td></tr></table>"
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>4매장 카테고리 수요 예측력 리포트</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          margin: 0; padding: 0; background: #f5f5f5; color: #333; }}
  header {{ background: #2c3e50; color: white; padding: 20px 30px; }}
  header h1 {{ margin: 0; font-size: 22px; }}
  header .meta {{ font-size: 14px; opacity: 0.85; margin-top: 5px; }}
  .tabs {{ display: flex; background: #34495e; padding: 0 20px; position: sticky; top: 0; z-index: 100; flex-wrap: wrap; }}
  .tabs button {{ background: none; border: none; color: white; padding: 14px 22px;
                  cursor: pointer; font-size: 15px; font-weight: 500; }}
  .tabs button.active {{ background: #1abc9c; }}
  .tabs button:hover {{ background: #16a085; }}
  .tab-content {{ display: none; padding: 30px; max-width: 1300px; margin: 0 auto; }}
  .tab-content.active {{ display: block; }}
  .chart-block {{ background: white; padding: 20px; margin-bottom: 25px; border-radius: 8px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .chart-block h2 {{ margin: 0 0 8px 0; font-size: 18px; color: #2c3e50; }}
  .chart-block .desc {{ color: #666; font-size: 13px; margin-bottom: 15px; line-height: 1.5; }}
  .intro {{ background: white; padding: 25px; border-radius: 8px; margin-bottom: 25px; }}
  .intro h2 {{ margin-top: 0; color: #2c3e50; }}
  .regime {{ background: #eef7f5; border-left: 4px solid #1abc9c; padding: 16px 20px;
             border-radius: 6px; margin-bottom: 22px; line-height: 1.8; font-size: 14px; }}
  .regime b {{ color: #16a085; }}
  .findings {{ background: #fff8ec; border-left: 4px solid #e67e22; padding: 18px 22px;
               border-radius: 6px; margin-bottom: 22px; line-height: 1.7; font-size: 14px; }}
  .findings h2 {{ margin: 0 0 10px 0; color: #b9660b; font-size: 18px; }}
  .findings ol {{ margin: 0 0 8px 0; padding-left: 20px; }}
  .findings li {{ margin-bottom: 8px; }}
  .findings li.risk {{ background: #fdecea; border-radius: 4px; padding: 8px 10px; }}
  .findings .note {{ color: #555; font-size: 13px; margin: 6px 0 0 0; }}
  table.mtab {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; font-size: 14px; }}
  table.mtab th, table.mtab td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: center; }}
  table.mtab th {{ background: #2c3e50; color: white; }}
  table.mtab td {{ background: #fafafa; font-weight: 600; }}
</style></head><body>
<header><h1>4매장 카테고리-합 수요 예측력 리포트</h1>
<div class="meta">FINALIZED regime · adjusted_demand α=0.8 · bulk 제외 · bread/pastry/sandwich 합 · 목요일→다음주 · 발주=매장별 최적 q (daily / weekly)</div></header>
<div class="tabs">{tab_buttons}</div>
{tab_contents}
<script>
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(e => e.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}
</script></body></html>
"""

REGIME_BOX = """<div class="regime">
<b>예측 대상</b>: bread/pastry/sandwich 카테고리 <b>합계</b>의 일 수요.
<b>target</b> = adjusted_demand_unit (정상판매 + 마감할인판매 × α, <b>α=0.8</b>) ·
<b>bulk 제외</b> = flag_bulk_lines (예약/대량 line 제거) ·
<b>학습창</b> = rolling 2Y(730일) ·
<b>cadence</b> = 목요일 anchor → 다음 7일, 주간 fold expanding backtest.<br>
<b>발주</b> = <b>q0.5(median) + 매장별 N·K 버퍼</b> (median×K+N). 고분위(q0.95) 직접발주는 저수요일 spread 병리로 폐기. ·
<b>제약</b> = 발주 주1회 → <b>주간커버 100% hard</b> + 주중 연속 누적부족 ≤ 33%×일평균수요.<br>
모든 fold 의 학습 데이터는 test 시작일 <b>이전</b>만 사용 — time leakage 없음.
</div>"""


ASSUMPTIONS_BOX = """<div class="chart-block">
<h2>이 모델이 깔고 있는 가정 · 전처리 기준</h2>
<div class="desc">예측 결과를 읽기 전에, 어떤 정의·가정 위에서 만들어졌는지 먼저 확인해야 합니다.</div>
<table class="mtab">
<tr><th>항목</th><th>기준 / 가정</th></tr>
<tr><td><b>카테고리 정의</b></td><td>POS 메뉴명 <b>키워드 매핑</b>으로 분류. 대상 = <b>bread / pastry / sandwich</b> 3종.
 bread=식사빵(식빵·바게트·베이글 등), pastry=단과자빵+페이스트리(크루아상·단팥·스콘·머핀 등), sandwich=완제품(토스트·파니니 등).
 <b>cake는 제외</b>(사전예약+시즌 특수라 일수요 예측 대상 아님), beverage/sweets도 제외. ※ 아띠제 공식 카테고리 미제공 → 임의 분류(추후 교체 대상).</td></tr>
<tr><td><b>모델링 단위(grain)</b></td><td>품목별이 아니라 <b>3개 카테고리를 합친 하루 단일 총량(EA 합계)</b>을 예측. 개별 품목은 이 총량을 비율 배분(별도 단계).
 근거 = 카테고리 내 수요흡수(품목 품절 시 다른 품목으로 이전)로 <b>총량은 보존</b>되므로 총량이 안정적으로 예측 가능(W0 게이트 통과).</td></tr>
<tr><td><b>예약(bulk) 제외</b></td><td>실수요 아닌 사전예약/대량주문 line을 제거(예측 불가·불필요). (영수증,품목) 단위 판정:
 <b>①단일품목형</b> = 한 영수증 한 품목 수량 ≥ 10 <b>이고</b> 그 품목 일판매 <b>중앙값의 3배</b> 이상 (품목 active days ≥ 14) ·
 <b>②다품목 event</b> = 영수증 총량 ≥ 30 <b>이고</b> 최대 단일품목 ≥ 5 → 그 영수증의 5개+ line만 제거.
 line-level 제거(소량 애드온·영수증 수는 보존). ※ POS에 예약 플래그 없어 <b>휴리스틱</b>, 정밀도 우선(애매지대는 남김).</td></tr>
<tr><td><b>마감할인 실수요(α)</b></td><td>마감 떨이 판매를 그대로 실수요로 보면 과대추정 → <b>adjusted = 정상판매 + α × 마감할인판매</b>, <b>α=0.8</b> 적용.
 마감할인 = 할인코드 <b>0069·0077·320</b>(마감세일 코드, 음료/제휴 할인 제외)로 식별. α는 점식별 불가라 가정값(높은-α 방향 증거) + 민감도.</td></tr>
<tr><td><b>발주점(생산량)</b></td><td><b>q0.5(median) + 버퍼(×K+N)</b>. 고분위 q0.95 직접발주는 저수요일(명절)에 상단 마진이 오히려 넓어지는 spread 병리 → median 기반+통제 버퍼로 회피. 버퍼는 매장별 최적화(daily=부족가중 w10, weekly=주간커버 100%+연속 누적부족 cap 하 폐기최소). 매진위험 = 실수요가 발주 초과한 비율.</td></tr>
<tr><td><b>학습·평가</b></td><td>학습창 = 최근 <b>2년(rolling)</b>, 목요일 기준 다음 7일 예측, 주간 fold. 지표 = WAPE(정확도)·WPE(편향)·매진위험(calibration). 검증 매장 = 광교·삼성·메세나·광화문 4곳.</td></tr>
</table>
<div class="desc" style="margin-top:12px;">⚠️ <b>주의</b>: 위 가정 중 <b>카테고리 분류·α·bulk 임계값</b>은 아띠제 실무 정보(공식 카테고리·원가율·실제 발주 로직) 수령 시 교체/재검증 대상입니다. 현재 예측은 이 가정들 위에서의 결과입니다.</div>
</div>"""


# 특수일 레벨-앵커 prior 설명 (verify_event_prior.py 재측정 결과 정적 반영).
EVENT_PRIOR_BOX = """<div class="chart-block">
<h2>특수일 sharp 당일 — 레벨-앵커 prior</h2>
<div class="desc">캘린더 lead-up 피처(<code>days_to_*</code>, ±14일)는 <b>뭉툭한 완만 상승만</b> 잡고 <b>당일 sharp 피크는 못 잡습니다</b>.
 예: 광교 크리스마스 당일 5년 실측이 ~330에 고정인데, 모델은 <b>주말 크리스마스를 평소 바쁜 주말로(→과대), 평일 크리스마스를 평소 한산한 평일로(→과소)</b> 예측합니다.
 이벤트가 요일 계절성을 덮어써 <b>고유 레벨로 끌어당기는</b> 현상(부호 = 상권: 주거/복합↑ · 오피스↓)입니다.</div>
<div class="desc" style="margin-top:8px;">트리(LightGBM)로는 못 배웁니다 — 카테고리 총량은 하루 1행이라 2년 학습창 안에 이벤트당 <b>2샘플</b>뿐,
 <code>min_child_samples</code>가 그 소수 샘플의 분리(split)를 막습니다(실증: 값을 낮춰도 해당 dummy의 gain=0). 당일 dummy·<code>요일×이벤트</code> interaction 모두 사망.</div>
<div class="desc" style="margin-top:8px;">해법 = <b>EventLevelPrior</b>(예측 이후 post-model 블렌드, 트리 밖): 과거 <b>같은 이벤트</b> 실측의 <b>중앙값(median, anomaly-robust)</b>에 shrink로 앵커합니다.
 <code>min_events=2</code>(단일 샘플 prior 차단) · <b>이벤트별 분리</b>(한 매장에 여러 이벤트가 등록돼도 안 섞임) · leakage-safe(예측 시점 이전 이벤트만).</div>
<table class="mtab">
<tr><th>매장 · 이벤트</th><th>기존 WAPE</th><th>+ prior</th></tr>
<tr><td>광교 · 크리스마스</td><td>0.231</td><td><b>0.102</b></td></tr>
<tr><td>광교 · 추석</td><td>0.214</td><td><b>0.145</b></td></tr>
<tr><td>삼성타운 · 크리스마스</td><td>0.827</td><td><b>0.421</b></td></tr>
<tr><td>메세나폴리스 · 크리스마스</td><td>0.174</td><td><b>0.100</b></td></tr>
<tr><td>메세나폴리스 · 설</td><td>0.179</td><td><b>0.101</b></td></tr>
<tr><td>광화문 · 크리스마스</td><td>0.151</td><td><b>0.085</b></td></tr>
</table>
<div class="desc" style="margin-top:12px;"><b>등록</b>(매장×이벤트 OOS 순개선 확인된 것만): 광교[크리스마스·추석] · 메세나[크리스마스·설] · 삼성·광화문[크리스마스].
 <b>미등록</b>: 광화문 설·메세나 추석(OOS 악화), 어린이날(기존 예측이 이미 정확한 매장에선 역효과 → 보류).</div>
<div class="desc" style="margin-top:8px;">⚠️ <b>한계</b>: 이벤트당 과거 3~5샘플로 OOS 증거가 얇고, 첫 발생은 교정 불가(과거 없음), median도 완만한 추세는 지연됩니다.
 삼성은 하향 보정(매진위험 높은 날 생산 축소)이라 운영 주의. 오피스(삼성·광화문)는 설/추석에 사실상 휴무.</div>
</div>"""


def _best_window(rows: list[dict]) -> int:
    return min(rows, key=lambda r: r["wape"])["window_days"]


def build_findings(summary: dict, stores: list) -> str:
    """예측력 핵심 결론 (동적) — q0.5 정확도 + median+버퍼 발주 정책 + 매장별 리스크."""
    wape_lo = min(summary[s.label]["headline_2Y"]["wape"] for s in stores) * 100
    wape_hi = max(summary[s.label]["headline_2Y"]["wape"] for s in stores) * 100
    # weekly 폐기절감 범위
    wr = []
    for s in stores:
        sid = SID_BY_LABEL.get(s.label)
        if sid:
            b = _cfg_holdout_stats(sid, find_opt_NK_weekly(sid))
            wr.append(b["waste_red_pct"])
    wr_lo, wr_hi = (min(wr), max(wr)) if wr else (0, 0)

    items = [
        f'<li><b>① 점추정(q0.5 median) 예측력 양호.</b> 4매장 카테고리-합 WAPE {wape_lo:.1f}~{wape_hi:.1f}%, '
        'signed WPE |≤1.1%| — 체계적 추세편향(trend-pinning) 거의 없음. (rolling 2Y, N=364일/매장)</li>',

        '<li><b>② 발주 = median + 매장별 N·K 버퍼.</b> 고분위(q0.95) 직접발주는 저수요일(명절)에 상단 마진이 '
        '오히려 넓어지는 <b>spread 병리</b>(광화문 마진-레벨 상관 −0.81)로 과잉발주 → <b>median 기반 + 통제 버퍼</b>로 회피. '
        'N·K를 backtest로 과최적화하진 않고(에러구조 손튜닝=과적합), 구조만 median+버퍼로 채택.</li>',

        '<li class="risk"><b>③ 발주 주1회 → 주간커버 100% hard 제약.</b> 주중 회복 불가라 한 주라도 못 덮으면 구조적 매진. '
        '그 안에서 <b>주중 연속 누적부족 ≤ 33%×일평균수요</b>(연속 일수 아닌 누적 수량 기준)로 배분 리스크 억제, '
        f'남는 여유로 <b>기존 발주 대비 폐기 {wr_lo:.0f}~{wr_hi:.0f}%↓</b>(weekly). daily(w10)/weekly 2안 제시.</li>',

        '<li><b>④ 매장별 지배 리스크가 다름 (단일 정책 아님).</b> '
        '광교=<b>이벤트 단일피크</b>(크리스마스 하루 45개 부족, 연속 아님) · '
        '광화문=<b>다일 연속 streak</b>(누적 큼) · 삼성=<b>변동 불안정</b>(홀드아웃 취약) · 메세나=안정(최대 절감). '
        '리스크는 ①매진율 ②연속 누적부족 ③단일일 최대부족 3축으로 봐야 함.</li>',

        '<li><b>⑤ 다음 단계(정석화): 분포 부스팅(σ(x) 학습) + conformal.</b> median+버퍼의 통계적 완성형 — '
        'feature가 레벨·마진을 동시 주도(명절엔 σ도 축소), 커버리지 보장. rare 이벤트 피크는 버퍼가 아니라 '
        '<b>feature 강화</b>(명절 강도 등)로 해결.</li>',
    ]
    return '<div class="findings"><h2>핵심 결론 — 예측력 & 발주 정책</h2><ol>' + "".join(items) + '</ol></div>'


def safe_id(label: str) -> str:
    return {"광교": "gw", "삼성타운": "ss", "메세나폴리스": "mp", "광화문": "gh"}[label]


def block(title: str, desc: str, img: str) -> str:
    return f'<div class="chart-block"><h2>{title}</h2><div class="desc">{desc}</div>{img}</div>\n'


def render_html(overview_blocks: str, store_sections: dict) -> str:
    buttons = '<button class="active" onclick="showTab(\'overview\')">종합</button>\n'
    contents = f'<div id="tab-overview" class="tab-content active">{overview_blocks}</div>\n'
    for label, body in store_sections.items():
        sid = safe_id(label)
        buttons += f'<button onclick="showTab(\'{sid}\')">{label}</button>\n'
        contents += f'<div id="tab-{sid}" class="tab-content">{body}</div>\n'
    return HTML_TEMPLATE.format(tab_buttons=buttons, tab_contents=contents)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_store(sd: StoreData, n_folds: int, sens_folds: int, *, full: bool = False) -> dict:
    """단일 매장 분석. 기본은 메인 2Y backtest만(빠름).

    `full=True`일 때만 window sensitivity(6창) + trend_ratio variant를 함께 계산한다 —
    window=2Y 고정·추세 피쳐 폐기가 확정된 뒤엔 재계산 불필요한 탐색 분석이라 기본 off.
    """
    cfg = STORE_EVENT_PRIORS.get(sd.label, {})
    print(f"  [{sd.label}] main backtest (2Y, {n_folds} folds) ...")
    main = windowed_backtest(sd.feat, window_days=DEFAULT_WINDOW_DAYS, n_folds=n_folds,
                              events=cfg.get("events"), lunar_events=cfg.get("lunar_events"))
    main_preds = main.predictions.copy()
    main_preds["date"] = pd.to_datetime(main_preds["date"])
    headline = metrics_from_preds(main_preds)

    window_rows: list[dict] = []
    variant = None
    if full:
        print(f"  [{sd.label}] window sensitivity ({sens_folds} folds) ...")
        for wd in WINDOWS:
            res = windowed_backtest(sd.feat, window_days=wd, n_folds=sens_folds,
                                     events=cfg.get("events"), lunar_events=cfg.get("lunar_events"))
            m = metrics_from_preds(res.predictions)
            m["window_days"] = wd
            window_rows.append(m)

        print(f"  [{sd.label}] variant (trend_ratio) ...")
        var_res = windowed_backtest(sd.feat_variant, window_days=DEFAULT_WINDOW_DAYS, n_folds=n_folds,
                                     events=cfg.get("events"), lunar_events=cfg.get("lunar_events"))
        assert list(main_preds["date"]) == list(pd.to_datetime(var_res.predictions["date"])), \
            f"{sd.label}: baseline/variant date misalignment"
        variant = {"baseline": headline, "variant": metrics_from_preds(var_res.predictions)}

    return {
        "main_preds": main_preds, "headline": headline,
        "window_rows": window_rows, "variant": variant,
    }


def build_store_body(sd: StoreData, r: dict) -> str:
    m = r["headline"]
    sid = sd.store_id
    pw = load_prod_waste(sd.cd_code)   # 기존 실제 생산량·폐기량 (비교 baseline)
    body = f'<div class="intro"><h2>{sd.label} 예측력</h2></div>\n'

    # headline — 예측 정확도(q0.5) + 발주 운영점(q0.5 고정 + N·K 버퍼)
    qd = find_opt_NK_daily(sid, w=W_PRIMARY)
    qw = find_opt_NK_weekly(sid)
    hl = f'<div class="chart-block"><h2>headline (rolling 2Y)</h2>'
    hl += ('<p class="desc"><b>① 예측 정확도 (q0.5 점추정, 발주버퍼와 무관)</b> — '
           f'WAPE <b>{m["wape"]*100:.1f}%</b> · WPE {m["wpe"]*100:+.1f}% (N={m["n_test"]}일)</p>')
    if qd is not None and qw is not None:
        sdl, swk = _cfg_holdout_stats(sid, qd), _cfg_holdout_stats(sid, qw)
        feas = "" if qw.get("feasible", True) else " ⚠️완화불가"
        hl += ('<p class="desc"><b>② 발주 운영점 (q0.5 median + 버퍼, 홀드아웃)</b> — '
               '발주 = median×K + N. q0.95 대신 median 기반으로 명절 등 저수요일 과잉발주 회피.</p>'
               '<table class="mtab"><tr><th>정책</th><th>발주(median×K+N)</th><th>매진율</th>'
               '<th>최대 연속매진 (일 / 누적부족)</th><th>주간커버</th><th>기존比 폐기절감</th></tr>'
               f'<tr><td>daily (w=10)</td><td>{_cfg_label(qd)}</td><td>{sdl["so_rate"]*100:.0f}%</td>'
               f'<td>{sdl["max_streak"]}일 / {sdl["max_run_short"]:.0f}개</td><td>{sdl["cover"]*100:.0f}%</td><td>{sdl["waste_red_pct"]:.0f}%</td></tr>'
               f'<tr><td>weekly{feas}</td><td>{_cfg_label(qw)}</td><td>{swk["so_rate"]*100:.0f}%</td>'
               f'<td>{swk["max_streak"]}일 / {swk["max_run_short"]:.0f}개</td><td>{swk["cover"]*100:.0f}%</td><td>{swk["waste_red_pct"]:.0f}%</td></tr>'
               '</table>')
    hl += "</div>\n"
    body += hl
    body += block("① 4-layer 일별 overlay",
                  "원판매(bulk 포함) → bulk 제외 → 실수요(α=0.8) → 모델 예측. 빨간 마름모=기존 실제 생산량(QT_MADE). "
                  "각 보정 단계가 target 을 어떻게 바꾸는지 + 기존 생산이 실수요 대비 얼마나 위였는지.",
                  plot_four_layer(sd, r["main_preds"], pw))

    if qd is not None and qw is not None:
        sd_stat = _cfg_holdout_stats(sid, qd)
        sw_stat = _cfg_holdout_stats(sid, qw)
        dl, wl = _cfg_label(qd), _cfg_label(qw)
        # === daily: q0.5 median + N·K (w=10) ===
        body += block(f"② daily 최적 발주 {dl} — 일별 발주 vs 실수요 vs 기존",
                      f"q=0.5(median) 고정 + 버퍼 K·N 최적화(부족 가중 w={int(W_PRIMARY)}). median 기반이라 명절 등 "
                      f"저수요일 과잉발주 회피. holdout 매진율 {sd_stat['so_rate']*100:.0f}% · "
                      f"최대 연속매진 {sd_stat['max_streak']}일(누적부족 {sd_stat['max_run_short']:.0f}개) · "
                      f"주간커버 {sd_stat['cover']*100:.0f}%.",
                      plot_order_daily(sid, qd, None, f"daily {dl}"))
        body += block(f"③ daily {dl} — 매진/폐기",
                      "위=폐기(발주-실측), 아래=매진분(실측-발주). 박스=기존 아띠제 발주 대비.",
                      plot_short_surp(sid, qd, None, f"daily {dl}"))
        # === weekly: q0.5 + N·K, w sweep ===
        feas = "" if qw.get("feasible", True) else " ⚠️연속매진 제약 미충족(최소 streak fallback)"
        body += block(f"④ weekly 최적 발주 {wl} — 일별 (점선=daily {dl} 대조)",
                      f"발주가 주 1회라 주간커버는 hard 제약(주중 회복 불가). (K,N) 전그리드에서 주간커버 100% & "
                      f"'연속 매진구간 누적부족 ≤ {int(MAX_RUN_SHORTFALL_FRAC*100)}%×일평균수요'(연속 일수 아닌 누적 수량 기준) "
                      f"만족하며 폐기 최소인 버퍼. holdout 매진율 {sw_stat['so_rate']*100:.0f}% · "
                      f"최대 연속매진 {sw_stat['max_streak']}일(누적부족 {sw_stat['max_run_short']:.0f}개) · "
                      f"주간커버 {sw_stat['cover']*100:.0f}%.{feas}",
                      plot_order_daily(sid, qw, qd, f"weekly {wl}"))
        body += block(f"⑤ weekly {wl} — 매진/폐기 (점선=daily {dl} 대조)",
                      "weekly 버퍼의 일별 실현 매진/폐기. daily보다 발주 낮춰 폐기↓·매진↑, 단 연속매진 제한 내. 박스=기존 대비.",
                      plot_short_surp(sid, qw, qd, f"weekly {wl}"))
        # === 연간 주단위 공급·수요 (weekly 버퍼 기준) ===
        body += block(f"⑥ 연간 주단위 — 기존 생산 vs 수요 vs weekly 최적 발주 {wl}",
                      "주단위 합산 관점(이월재고 흡수). 주 총량이 수요를 덮으면 하루 매진은 전날/주중 재고로 커버 "
                      "(며칠 연속 매진만 아니면). weekly 버퍼 발주 사용. 기존 생산량=adjusted 스케일 보정.",
                      plot_weekly_supply_demand(sid, qw))
    else:
        body += block("② 날짜별 매진위험 / 실현 잉여·부족",
                      "막대 위=폐기, 아래=매진분. (margin qmat 미생성 매장)",
                      plot_risk(sd, r["main_preds"], pw))

    body += block("⑦ 월별 signed WPE (추세 pinning 진단)",
                  "(예측-실측)/실측 월별 집계. 부호가 한쪽으로 쏠리면 추세 지연(pinning) — 성장/하락 구간 체계적 편향.",
                  plot_wpe_over_time(sd, r["main_preds"]))
    return body


def run_compute(full: bool = False) -> dict:
    """메인 2Y backtest 실행 → 캐시(pickle) + summary JSON 저장. HTML은 만들지 않는다.

    기본은 간소화(매장별 메인 2Y backtest만) — window 2Y 고정·추세 피쳐 폐기가 확정돼
    window sweep·variant는 재계산 불필요. `full=True`면 그 탐색 분석까지 포함(느림).
    HTML만 반복 수정할 땐 이 단계를 건너뛰고 run_render()가 캐시를 로드한다.
    """
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    stores = [build_store_data(cd, sid, label) for cd, sid, label, _ in STORES]

    results, window_table, variant_table, summary = {}, {}, {}, {}
    for sd in stores:
        n_folds = feasible_folds(sd.feat)
        sens_folds = min(SENS_FOLDS, n_folds)
        print(f"[{sd.label}] feasible folds = {n_folds} (sens={sens_folds}), N(rows)={len(sd.feat.dropna())}")
        r = run_store(sd, n_folds, sens_folds, full=full)
        results[sd.label] = r
        window_table[sd.label] = r["window_rows"]
        variant_table[sd.label] = r["variant"]
        summary[sd.label] = {
            "n_folds": n_folds,
            "sens_folds": sens_folds,
            "n_rows_usable": int(len(sd.feat.dropna())),
            "headline_2Y": r["headline"],
            "window_sensitivity": r["window_rows"],
            "variant": r["variant"],
        }

    # anchor 재검증: 새 windowed_backtest 로 광교 window=1825, n_folds=26 → ≈expanding
    print("\n[anchor] 광교 window=1825 n_folds=26 (via windowed_backtest) ...")
    gw = next(s for s in stores if s.label == "광교")
    gw_cfg = STORE_EVENT_PRIORS.get("광교", {})
    ares = windowed_backtest(gw.feat, window_days=1825, n_folds=26,
                              events=gw_cfg.get("events"), lunar_events=gw_cfg.get("lunar_events"))
    am = metrics_from_preds(ares.predictions)
    anchor_result = {"wape": am["wape"], "stockout_risk": am["stockout_risk"], "n_test": am["n_test"]}
    anchor_ok = abs(am["wape"] - 0.077) <= 0.005 and abs(am["stockout_risk"] - 0.15) <= 0.03
    summary["_anchor"] = {**anchor_result, "passed": bool(anchor_ok)}

    bundle = {
        "stores": stores, "results": results, "window_table": window_table,
        "variant_table": variant_table, "summary": summary,
        "anchor": anchor_result, "anchor_ok": bool(anchor_ok),
    }
    with open(CACHE, "wb") as f:
        pickle.dump(bundle, f)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"\nsaved cache {CACHE} ({CACHE.stat().st_size/1024/1024:.1f} MB)")
    print(f"saved {OUT_JSON}")
    print(f"\n[ANCHOR] window=1825 n_folds=26: WAPE={am['wape']:.4f} "
          f"risk={am['stockout_risk']:.4f} n={am['n_test']} → {'PASS' if anchor_ok else 'FAIL'}")
    print("\n[2Y headline per store]")
    for sd in stores:
        h = summary[sd.label]["headline_2Y"]
        print(f"  {sd.label}: WAPE={h['wape']*100:.2f}% WPE={h['wpe']*100:+.2f}% "
              f"매진위험={h['stockout_risk']*100:.1f}% (N={h['n_test']}, folds={summary[sd.label]['n_folds']})")
    if not anchor_ok:
        raise SystemExit("ANCHOR FAILED — 결과 신뢰 불가, 중단")
    return bundle


def run_render(bundle: dict | None = None) -> None:
    """캐시(compute 결과)를 로드해 플롯 + HTML만 생성한다 (backtest 재계산 없음 → 빠름)."""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    if bundle is None:
        if not CACHE.exists():
            raise SystemExit(f"캐시 없음: {CACHE} — 먼저 `compute` 단계를 실행하세요.")
        print(f"[render] 캐시 로드 {CACHE} ...")
        with open(CACHE, "rb") as f:
            bundle = pickle.load(f)
    stores = bundle["stores"]
    results = bundle["results"]
    window_table = bundle["window_table"]
    variant_table = bundle["variant_table"]
    summary = bundle["summary"]
    am = bundle["anchor"]

    # ----- charts -----
    print("\n[charts] cross-store panels ...")
    has_window = any(window_table.get(l) for l in window_table)
    has_variant = any(variant_table.get(l) for l in variant_table)

    overview = REGIME_BOX
    overview += build_findings(summary, stores)
    overview += ASSUMPTIONS_BOX
    overview += EVENT_PRIOR_BOX
    overview += '<div class="intro"><h2>종합 — 매장 간 예측력 비교</h2></div>\n'
    # headline 요약 — 예측 정확도(q0.5) + daily/weekly 최적 q 운영점
    rows = ""
    for s in stores:
        h = summary[s.label]["headline_2Y"]
        sid = SID_BY_LABEL.get(s.label)
        qd = find_opt_NK_daily(sid, w=W_PRIMARY) if sid else None
        qw = find_opt_NK_weekly(sid) if sid else None
        if qd and qw:
            a = _cfg_holdout_stats(sid, qd); b = _cfg_holdout_stats(sid, qw)
            dq = (f"{_cfg_label(qd)} (매진{a['so_rate']*100:.0f}%·연속{a['max_streak']}일/누적{a['max_run_short']:.0f}개·폐기{a['waste_red_pct']:.0f}%↓)")
            wqf = "" if qw.get("feasible", True) else "⚠️"
            wq = (f"{_cfg_label(qw)}{wqf} (매진{b['so_rate']*100:.0f}%·연속{b['max_streak']}일/누적{b['max_run_short']:.0f}개·폐기{b['waste_red_pct']:.0f}%↓)")
        else:
            dq = wq = "—"
        rows += (f"<tr><td>{s.label}</td><td>{h['wape']*100:.2f}%</td>"
                 f"<td>{h['wpe']*100:+.2f}%</td><td>{dq}</td><td>{wq}</td></tr>")
    overview += (
        '<div class="chart-block"><h2>매장별 headline (rolling 2Y)</h2>'
        '<p class="desc">WAPE·WPE = q0.5 점추정 정확도(발주버퍼 무관). daily/weekly = q0.5 median + 최적 버퍼(median×K+N) '
        '+ 홀드아웃 매진율·최대연속매진·기존比 폐기절감. daily=w10 고정, weekly=w sweep+연속매진≤2.</p>'
        '<table class="mtab"><tr><th>매장</th><th>WAPE(q0.5)</th><th>WPE(q0.5)</th>'
        '<th>daily 발주(median×K+N)</th><th>weekly 발주</th></tr>' + rows + "</table></div>\n"
    )
    if has_window:
        win_wape_img, win_risk_img = plot_window_sensitivity(window_table)
        overview += block("⑤ 학습창 × WAPE (window sensitivity)",
                          "rolling window 90~1825일. WAPE 최소화 지점이 매장별 최적 학습창. 1825≈expanding. "
                          "스윕은 대형창 재적합 비용 절감을 위해 26 folds(약 6개월)로 계산(헤드라인은 52 folds).",
                          win_wape_img)
        overview += block("⑤ 학습창 × 매진위험",
                          "학습창이 매진위험(q0.85 발주 대비 실측 초과율)에 주는 영향.", win_risk_img)
    if has_variant:
        variant_img = plot_variant(variant_table)
        overview += block("⑥ feature variant: baseline vs + trend_ratio",
                          "장/단기 추세비(roll_mean_28/roll_mean_180) 추가가 WAPE 를 낮추는가. Δpp<0 이면 도움.",
                          variant_img)
    if not (has_window or has_variant):
        overview += block("⑤·⑥ 학습창·feature 탐색 (확정 → 생략)",
                          "window=2Y 고정·trend_ratio 미채택이 이전 full 분석에서 확정돼 재계산 생략. "
                          "다시 보려면 `compute --full`.", "")

    store_sections = {sd.label: build_store_body(sd, results[sd.label]) for sd in stores}
    html = render_html(overview, store_sections)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nsaved {OUT_HTML} ({OUT_HTML.stat().st_size/1024:.0f} KB)")
    print(f"[anchor(cache)] WAPE={am['wape']:.4f} risk={am['stockout_risk']:.4f} "
          f"passed={summary.get('_anchor', {}).get('passed')}")


def main() -> None:
    """단계 분리 실행:
      compute — 무거운 backtest 실행 후 캐시 저장 (HTML 안 만듦, ~38분)
      html    — 캐시 로드해 HTML만 생성 (빠름, 반복 수정용)
      all     — compute 후 바로 html (기본)
    """
    stage = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "all"
    full = "--full" in sys.argv   # window sweep + variant 탐색 분석 포함 (느림, 기본 off)
    if stage == "compute":
        run_compute(full=full)
    elif stage == "html":
        run_render()
    elif stage == "all":
        run_render(run_compute(full=full))
    else:
        raise SystemExit("usage: store_predictive_power.py [compute|html|all] [--full]")


if __name__ == "__main__":
    main()
