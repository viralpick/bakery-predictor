"""[일회성 탐색] expected 모델 하이퍼파라미터 2목표 스윕 — WAPE ↔ 요일 편향.

★일회성임을 명시한다(라우팅 표 escape hatch). 승자가 나오면 즉시 `ExperimentSpec` 에
모델 파라미터 블록을 추가해 백본으로 승격하고 YAML로 재현 가능하게 만든다. 승자가 없으면
이 스크립트와 결과 문서만 남는다.

왜 필요한가: 요일별 편향이 **조건부 중앙값 자체의 miscalibration**(실제 모델 결함)으로
규명됐다(docs/expected_objective_result.md). 지표 아티팩트도 정규화 부산물도 아니다.
과거 튜닝은 WAPE 단일 축이었고 두 목표를 **동시에** 본 적이 없다.

★★설계 핵심 = selection overfitting 방어. 이 프로젝트는 여기서 반복해 당했다
(NK 과적합 폐기 / customization=noise-driven / convex loss 전부 negative). 같은 fold에
N개 config를 돌려 최고를 고르면 노이즈를 고르는 것이고 그 수치는 인용 불가다. 그래서:
  - fold를 **시간으로 분할**: 오래된 절반 = 탐색(select), 최근 절반 = 확인(confirm)
  - 최종 수치는 confirm에서 **한 번만** 낸다
  - **기본 설정을 반드시 후보에 포함** — 현행이 파레토 위에 있으면 "튜닝 여지 없음"이
    결론이고 그것도 유효한 답이다
  - 스칼라 하나로 합치지 않고 **파레토 프론티어**로 보고(가중치를 임의로 정하면
    architect 판단을 뺏는다)

실행:
    uv run python scripts/tune_expected_two_objective.py            # 기본(탐색+확인)
    uv run python scripts/tune_expected_two_objective.py --quick    # 축소 그리드 smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.models.category_total import (
    EXPECTED_OBJECTIVE_L1,
    EXPECTED_OBJECTIVE_L2,
    select_feature_cols,
)

sys.stdout.reconfigure(line_buffering=True)   # 밤샘 실행: 진행 로그를 즉시 흘린다

TARGET = "adjusted_demand_unit"
HORIZON_DAYS = 7
WINDOW_DAYS = 730
MIN_TRAIN_ROWS = 60
N_FOLDS_TOTAL = 52
DOW_NAMES = ("월", "화", "수", "목", "금", "토", "일")

# 현행 헤드라인 설정 — 반드시 후보에 포함한다(비교 기준선).
BASELINE = dict(
    objective=EXPECTED_OBJECTIVE_L1, num_leaves=31, max_depth=6,
    min_child_samples=20, learning_rate=0.05, n_estimators=400,
    feature_fraction=1.0, lambda_l2=0.0,
)

# 탐색 축. learning_rate×n_estimators는 등가 교환이라 쌍으로 묶는다.
GRID = dict(
    objective=[EXPECTED_OBJECTIVE_L1, EXPECTED_OBJECTIVE_L2],
    leaves_depth=[(15, 4), (31, 6), (63, 8)],
    min_child_samples=[10, 20, 40],
    lr_trees=[(0.05, 400), (0.03, 700)],
    feature_fraction=[0.7, 1.0],
    lambda_l2=[0.0, 5.0],
)
QUICK_GRID = dict(
    objective=[EXPECTED_OBJECTIVE_L1, EXPECTED_OBJECTIVE_L2],
    leaves_depth=[(31, 6), (63, 8)],
    min_child_samples=[20],
    lr_trees=[(0.05, 400)],
    feature_fraction=[1.0],
    lambda_l2=[0.0],
)


def build_folds(feat: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """헤드라인과 같은 fold 규약(연속 7일 블록, train = date < test_start, 730일 창).

    fold 0 = 가장 최근. windowed_backtest 산술을 그대로 따른다(비교 가능성 유지).
    """
    total = len(feat)
    folds = []
    for k in range(N_FOLDS_TOTAL):
        end = total - k * HORIZON_DAYS
        test = feat.iloc[end - HORIZON_DAYS:end]
        start_date = test["date"].iloc[0]
        train = feat[(feat["date"] < start_date)
                     & (feat["date"] >= start_date - pd.Timedelta(days=WINDOW_DAYS))]
        if len(train) < MIN_TRAIN_ROWS:
            continue
        folds.append((train, test))
    return folds


def run_config(cfg: dict, folds, cols: list[str]) -> pd.DataFrame:
    """config 하나를 fold 목록에 돌려 예측 프레임 반환. event_prior는 적용하지 않는다.

    ★event_prior를 빼는 이유: 이벤트일 보정이 목적함수·하이퍼파라미터 효과와 섞이면
    무엇이 편향을 움직였는지 식별할 수 없다. 헤드라인 절대값과는 비교하지 않고
    **config 간 상대 비교**에만 쓴다.
    """
    leaves, depth = cfg["leaves_depth"]
    lr, trees = cfg["lr_trees"]
    params = dict(
        objective=cfg["objective"], num_leaves=leaves, max_depth=depth,
        min_child_samples=cfg["min_child_samples"], learning_rate=lr,
        n_estimators=trees, colsample_bytree=cfg["feature_fraction"],
        reg_lambda=cfg["lambda_l2"], random_state=42, verbosity=-1,
    )
    rows = []
    for train, test in folds:
        model = lgb.LGBMRegressor(**params).fit(train[cols], train[TARGET])
        rows.append(pd.DataFrame({
            "date": test["date"].to_numpy(),
            "actual": test[TARGET].to_numpy(),
            "expected": np.clip(model.predict(test[cols]), 0.0, None),
        }))
    return pd.concat(rows, ignore_index=True)


def score(pred: pd.DataFrame) -> dict:
    """2목표 + 진단. wape = 정확도, dow_abs_wpe = 요일 편향(작을수록 좋음)."""
    actual, expected = pred["actual"], pred["expected"]
    dow = pred["date"].dt.dayofweek
    per_dow = {}
    for k, g in pred.groupby(dow):
        per_dow[DOW_NAMES[k]] = float((g.expected - g.actual).sum() / max(g.actual.sum(), 1))
    return {
        "wape": float(np.abs(actual - expected).sum() / max(np.abs(actual).sum(), 1)),
        "dow_abs_wpe": float(sum(abs(v) for v in per_dow.values())),
        "wpe": float((expected - actual).sum() / max(actual.sum(), 1)),
        "std_ratio": float(actual.std() / expected.std()),
        **{f"wpe_{k}": v for k, v in per_dow.items()},
    }


def pareto_front(table: pd.DataFrame, axes=("wape", "dow_abs_wpe")) -> pd.DataFrame:
    """두 축 모두 작을수록 좋음 — 지배당하지 않는 행만."""
    keep = []
    values = table[list(axes)].to_numpy()
    for i, row in enumerate(values):
        dominated = np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1))
        if not dominated:
            keep.append(i)
    return table.iloc[keep].sort_values(list(axes)).reset_index(drop=True)


def label(cfg: dict) -> str:
    leaves, depth = cfg["leaves_depth"]
    lr, trees = cfg["lr_trees"]
    obj = "L1" if cfg["objective"] == EXPECTED_OBJECTIVE_L1 else "L2"
    return (f"{obj}/lv{leaves}d{depth}/mcs{cfg['min_child_samples']}"
            f"/lr{lr}t{trees}/ff{cfg['feature_fraction']}/l2r{cfg['lambda_l2']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="축소 그리드 smoke")
    ap.add_argument("--out", default="reports/tune_two_objective.csv")
    args = ap.parse_args()

    grid = QUICK_GRID if args.quick else GRID
    feat = build_features(build_category_daily(), target_col=TARGET).dropna().reset_index(drop=True)
    cols = select_feature_cols(feat, TARGET)
    folds = build_folds(feat)
    # ★시간 분할: fold 0이 최신이므로 뒤쪽(오래된)이 탐색, 앞쪽(최근)이 확인.
    half = len(folds) // 2
    confirm_folds, select_folds = folds[:half], folds[half:]
    print(f"fold {len(folds)}개 → 탐색(오래된) {len(select_folds)} / 확인(최근) {len(confirm_folds)}")
    print(f"feature {len(cols)}개 | config {int(np.prod([len(v) for v in grid.values()]))}개\n")

    keys = list(grid)
    configs = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]
    if not any(all(c[k] == BASELINE_MAP[k] for k in keys) for c in configs):
        configs.insert(0, {k: BASELINE_MAP[k] for k in keys})   # 기준선 강제 포함

    rows = []
    for i, cfg in enumerate(configs, 1):
        name = label(cfg)
        s = score(run_config(cfg, select_folds, cols))
        rows.append({"config": name, "phase": "select", **s, **{f"_{k}": str(cfg[k]) for k in keys}})
        print(f"[{i}/{len(configs)}] {name:52s} wape={s['wape']:.5f} dow|wpe|={s['dow_abs_wpe']:.4f}")

    select_tbl = pd.DataFrame(rows)
    front = pareto_front(select_tbl)
    print(f"\n=== 탐색 파레토 프론티어 ({len(front)}개) ===")
    print(front[["config", "wape", "dow_abs_wpe", "wpe", "std_ratio"]].round(5).to_string(index=False))

    # 확인 단계 — 파레토 후보 + 기준선만 최근 fold에서 한 번 측정
    base_label = label({k: BASELINE_MAP[k] for k in keys})
    to_confirm = list(dict.fromkeys(list(front["config"]) + [base_label]))
    by_label = {label(c): c for c in configs}
    crows = []
    print(f"\n=== 확인(최근 {len(confirm_folds)} fold) — {len(to_confirm)}개 ===")
    for name in to_confirm:
        s = score(run_config(by_label[name], confirm_folds, cols))
        crows.append({"config": name, "phase": "confirm", **s})
        print(f"  {name:52s} wape={s['wape']:.5f} dow|wpe|={s['dow_abs_wpe']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([select_tbl, pd.DataFrame(crows)], ignore_index=True).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(json.dumps({"baseline": base_label, "pareto_n": len(front)}, ensure_ascii=False))


BASELINE_MAP = {
    "objective": BASELINE["objective"],
    "leaves_depth": (BASELINE["num_leaves"], BASELINE["max_depth"]),
    "min_child_samples": BASELINE["min_child_samples"],
    "lr_trees": (BASELINE["learning_rate"], BASELINE["n_estimators"]),
    "feature_fraction": BASELINE["feature_fraction"],
    "lambda_l2": BASELINE["lambda_l2"],
}

if __name__ == "__main__":
    main()
