"""운영 예측 패널 백테스트 — fold를 **원점**으로 자른다.

`backtest_core.windowed_backtest` 와 형제 함수다. 패널은 fold 계약이 다르므로(대상일 7일
블록 → 원점 1개) 하드 게이트가 걸린 그 함수를 더 파라미터화하지 않고 분리했다.
반환은 같은 `BacktestResult` shape이라 `metrics_from_preds`·`report.py` 가 무변경으로 소비한다.

leakage 차단은 fold 정의 한 줄이 전부다 — 평가 원점 O에 대해 **`target_date <= O` 인 행만
학습**한다. `target_date <= O` 면 `origin = target − offset <= O − min(offsets) < O` 이므로
원점도 자동으로 O 이전이다. 즉 "O 시점에 실측이 확정된 행"만 쓴다.

설계 근거는 docs/superpowers/specs/2026-07-29-operational-panel-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.features.forecast_panel import origin_feature_columns
from bakery.harness.forecasters import CategoryTotalForecaster, Forecaster
from bakery.models.category_total import BacktestResult
from bakery.models.event_prior import EventLevelPrior

MIN_TRAIN_ROWS = 60
TUESDAY = 1        # 운영 원점: 수요일 오전 발주 전달 → 화요일까지의 데이터


def _fit_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """forecaster 입력용 프레임. origin_date를 떼고 target_date를 date로 바꾼다.

    `select_feature_cols` 가 date와 타깃을 제외하므로, 날짜 컬럼이 feature로 새는 것을 막는다.
    """
    return rows.drop(columns=["origin_date"]).rename(columns={"target_date": "date"})


def _eval_origins(panel: pd.DataFrame, *, origin_dow: int, n_offsets: int, n_folds: int) -> list[pd.Timestamp]:
    """평가 원점(최신 → 과거). offset 전부가 갖춰진 완전한 블록만 쓴다."""
    counts = panel.groupby("origin_date").size()
    complete = counts[counts == n_offsets].index
    origins = sorted(o for o in complete if pd.Timestamp(o).dayofweek == origin_dow)
    return [pd.Timestamp(o) for o in reversed(origins)][:n_folds]


def panel_backtest(
    panel: pd.DataFrame, *, window_days: int,
    target_col: str = "adjusted_demand_unit", n_folds: int = 52,
    production_q: float = 0.85, alpha: float = 0.8,
    events: dict | None = None, lunar_events: dict | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
    forecaster: Forecaster | None = None,
    origin_dow: int = TUESDAY,
) -> BacktestResult:
    """원점 기준 fold로 패널을 백테스트한다.

    n_folds 는 **평가 원점 수**다(원점당 offset 수만큼 test 행이 나온다).
    """
    fc = forecaster if forecaster is not None else CategoryTotalForecaster()
    ar_cols = origin_feature_columns(panel)
    panel = panel.dropna(subset=[target_col, *ar_cols]).reset_index(drop=True)
    panel = panel.sort_values(["origin_date", "horizon_offset"]).reset_index(drop=True)

    offsets = sorted(panel["horizon_offset"].unique())
    origins = _eval_origins(
        panel, origin_dow=origin_dow, n_offsets=len(offsets), n_folds=n_folds,
    )
    if not origins:
        raise ValueError(
            f"평가 원점이 없다 — origin_dow={origin_dow}, offsets={offsets}. "
            "패널 범위와 요일을 확인하라."
        )

    window = pd.Timedelta(days=window_days)
    folds, preds = [], []
    for k, origin in enumerate(origins):
        test_df = panel[panel["origin_date"] == origin]
        # ★leakage 차단: O 시점에 실측이 확정된 행만 학습(원점도 자동으로 O 이전).
        train_df = panel[(panel["target_date"] <= origin)
                         & (panel["target_date"] >= origin - window)]
        if len(train_df) < min_train_rows:
            continue
        model = fc.fit(
            _fit_frame(train_df), target_col=target_col, alpha=alpha, production_q=production_q,
        )
        test_fit = _fit_frame(test_df)
        exp_pred = model.predict_expected(test_fit)
        prod_pred = model.predict_production(test_fit)
        if events or lunar_events:
            # prior history도 동일 cutoff — 대상일 실측이 확정된 구간만.
            hist = (panel.loc[panel["target_date"] <= origin, ["target_date", target_col]]
                    .drop_duplicates("target_date")
                    .rename(columns={"target_date": "date"}))
            prior = EventLevelPrior(events=events, lunar_events=lunar_events).fit(
                hist, target_col=target_col,
            )
            exp_pred, prod_pred = prior.blend(
                test_df["target_date"].values, exp_pred, prod_pred,
            )
        actual = test_df[target_col].to_numpy()
        folds.append(dict(
            fold=k, n_train=len(train_df), n_test=len(test_df),
            test_start=test_df["target_date"].iloc[0],
            test_end=test_df["target_date"].iloc[-1],
            origin_date=origin,
            wape=np.abs(actual - exp_pred).sum() / max(np.abs(actual).sum(), 1),
            wpe=(exp_pred - actual).sum() / max(actual.sum(), 1),
            prod_pct_under=(prod_pred < actual).mean(),
        ))
        preds.append(pd.DataFrame({
            "date": test_df["target_date"].to_numpy(), "fold": k,
            "horizon_offset": test_df["horizon_offset"].to_numpy(),
            "actual": actual, "expected": exp_pred, "production": prod_pred,
        }))
    return BacktestResult(
        folds=pd.DataFrame(folds).sort_values("fold").reset_index(drop=True),
        predictions=pd.concat(preds, ignore_index=True),
    )


def horizon_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """offset별 WAPE·WPE — horizon_offset을 feature로 준 것만으로 offset별 편향이
    자동 해소되지는 않으므로, 체계적으로 어긋나는 offset이 있는지 반드시 본다."""
    rows = []
    for offset, group in predictions.groupby("horizon_offset"):
        actual, expected = group["actual"], group["expected"]
        denom = max(float(np.abs(actual).sum()), 1.0)
        rows.append({
            "horizon_offset": int(offset),
            "n": int(len(group)),
            "wape": float(np.abs(actual - expected).sum() / denom),
            "wpe": float((expected - actual).sum() / max(float(actual.sum()), 1.0)),
        })
    return pd.DataFrame(rows).sort_values("horizon_offset").reset_index(drop=True)
