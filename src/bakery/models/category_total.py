"""Stage 1: Category-Total Demand LightGBM (unit + revenue 병행)."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd


# 기대수요(center) 모델 목적함수. L1=조건부 중앙값(헤드라인), L2=조건부 평균(실험).
EXPECTED_OBJECTIVE_L1 = "regression_l1"
EXPECTED_OBJECTIVE_L2 = "regression"

# 학습 시 target 외에 leak 되는 컬럼 (다른 target도 leak 처리)
LEAK_COLS = (
    "sold_total_unit", "sold_total_revenue",
    "sold_normal_unit", "sold_normal_revenue",
    "sold_closing", "sold_closing_revenue",
    "adjusted_demand_unit", "adjusted_demand_revenue",
    "n_stockout_items", "n_early_stockout", "n_items_active",
)


@dataclass
class CategoryTotalModel:
    expected: lgb.LGBMRegressor
    quantile: lgb.LGBMRegressor
    feature_cols: list[str]
    alpha_demand: float
    production_q: float
    target_col: str
    production_lo: lgb.LGBMRegressor | None = None
    q_lo: float | None = None

    def predict_expected(self, X: pd.DataFrame) -> np.ndarray:
        return self.expected.predict(X[self.feature_cols])

    def predict_production(self, X: pd.DataFrame) -> np.ndarray:
        return self.quantile.predict(X[self.feature_cols])

    def predict_production_lo(self, X: pd.DataFrame) -> np.ndarray:
        if self.production_lo is None:
            raise ValueError("no q_lo model: fit_category_total was called without q_lo")
        return self.production_lo.predict(X[self.feature_cols])


def select_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    return [c for c in df.columns if c not in ("date", target_col, *LEAK_COLS)]


def fit_category_total(
    train: pd.DataFrame,
    target_col: str = "adjusted_demand_unit",
    alpha_demand: float = 0.5,
    production_q: float = 0.90,
    q_lo: float | None = None,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    random_state: int = 42,
    expected_objective: str = EXPECTED_OBJECTIVE_L1,
) -> CategoryTotalModel:
    """Fit the expected and q0.90 production models.

    When `q_lo` is given, also fits an adaptive lower-quantile model used as the
    asymmetric interval's lower anchor. Symmetric intervals don't need it, so it
    stays off by default.

    `expected_objective` 는 **기대수요(center) 모델의 목적함수**다. 기본 L1(조건부
    중앙값)이 헤드라인이며, `regression`(L2, 조건부 평균)은 실험용 대안이다.
    생산량(quantile) 모델은 이 인자와 무관하게 항상 quantile 목적함수를 쓴다.

    ★왜 인자로 뺐나: 평가 지표(WAPE/WPE)는 **합계 기준**이라 평균성 대상을 가리키는데
    L1은 중앙값을 맞춘다 — 훈련 목표와 평가 지표의 불일치가 요일별 편향·예측 압축의
    후보 원인으로 실측됐다(docs/expected_objective_result.md). 축을 실험 파라미터로
    노출해 데이터로 판정한다.
    """
    feat_cols = select_feature_cols(train, target_col)
    X = train[feat_cols]
    y = train[target_col]
    common = dict(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, num_leaves=num_leaves,
        random_state=random_state, verbosity=-1,
    )
    expected = lgb.LGBMRegressor(objective=expected_objective, **common).fit(X, y)
    quantile = lgb.LGBMRegressor(objective="quantile", alpha=production_q, **common).fit(X, y)
    production_lo = None
    if q_lo is not None:
        production_lo = lgb.LGBMRegressor(objective="quantile", alpha=q_lo, **common).fit(X, y)
    return CategoryTotalModel(
        expected=expected, quantile=quantile,
        feature_cols=feat_cols,
        alpha_demand=alpha_demand, production_q=production_q, target_col=target_col,
        production_lo=production_lo, q_lo=q_lo,
    )


@dataclass
class BacktestResult:
    folds: pd.DataFrame
    predictions: pd.DataFrame


@dataclass
class CalibrationFold:
    fold: int
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def expanding_calibration_folds(
    df: pd.DataFrame,
    *,
    target_col: str = "adjusted_demand_unit",
    n_folds: int = 4,
    min_train_days: int = 365,
    calibration_days: int = 60,
    horizon_days: int = 30,
) -> list[CalibrationFold]:
    """Time-ordered train → calibration → test folds for conformal intervals.

    Splits on unique dates (not row index) so a single date never straddles the
    calibration/test boundary — that would leak future residuals into the margin
    (CLAUDE.md absolute rules #1, #3). Calibration always sits strictly between
    train and test.
    """
    df = df.sort_values("date").dropna(subset=[target_col]).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(df["date"].unique())))
    n = len(dates)
    needed = min_train_days + calibration_days + n_folds * horizon_days
    if n < needed:
        raise ValueError(f"Not enough unique days: {n} < {needed}")

    folds: list[CalibrationFold] = []
    for k in range(n_folds):
        test_end_i = n - k * horizon_days
        test_start_i = test_end_i - horizon_days
        cal_start_i = test_start_i - calibration_days
        train_dates = dates[:cal_start_i]
        cal_dates = dates[cal_start_i:test_start_i]
        test_dates = dates[test_start_i:test_end_i]
        folds.append(
            CalibrationFold(
                fold=k,
                train=df[df["date"].isin(train_dates)].copy(),
                calibration=df[df["date"].isin(cal_dates)].copy(),
                test=df[df["date"].isin(test_dates)].copy(),
            )
        )
    return list(reversed(folds))


def expanding_window_backtest(
    df: pd.DataFrame,
    target_col: str = "adjusted_demand_unit",
    n_folds: int = 4,
    min_train_days: int = 365,
    horizon_days: int = 30,
    alpha_demand: float = 0.5,
    production_q: float = 0.90,
) -> BacktestResult:
    df = df.sort_values("date").reset_index(drop=True).dropna(subset=[target_col]).copy()
    df = df.dropna().reset_index(drop=True)
    total = len(df)
    test_size = horizon_days
    n_test_total = n_folds * test_size
    if total < min_train_days + n_test_total:
        raise ValueError(f"Not enough data: {total} < {min_train_days + n_test_total}")

    folds, preds = [], []
    for k in range(n_folds):
        test_end = total - k * test_size
        test_start = test_end - test_size
        train_df = df.iloc[:test_start]
        test_df  = df.iloc[test_start:test_end]
        model = fit_category_total(
            train_df, target_col=target_col,
            alpha_demand=alpha_demand, production_q=production_q,
        )
        exp_pred = model.predict_expected(test_df)
        prod_pred = model.predict_production(test_df)
        actual = test_df[target_col].values
        wape = np.abs(actual - exp_pred).sum() / max(np.abs(actual).sum(), 1)
        mae  = np.abs(actual - exp_pred).mean()
        rmse = np.sqrt(((actual - exp_pred)**2).mean())
        folds.append(dict(
            fold=k, target=target_col, alpha=alpha_demand, q=production_q,
            test_start=test_df["date"].iloc[0], test_end=test_df["date"].iloc[-1],
            n_train=len(train_df), n_test=len(test_df),
            wape=wape, mae=mae, rmse=rmse,
            pct_under=(exp_pred < actual).mean(),
            pct_over=(exp_pred > actual).mean(),
            prod_pct_under=(prod_pred < actual).mean(),
            prod_pct_over=(prod_pred > actual).mean(),
            mean_expected=exp_pred.mean(), mean_production=prod_pred.mean(),
            mean_actual=actual.mean(),
        ))
        preds.append(pd.DataFrame({
            "date": test_df["date"].values,
            "target": target_col, "alpha": alpha_demand, "q": production_q, "fold": k,
            "actual": actual, "expected": exp_pred, "production": prod_pred,
        }))
    return BacktestResult(
        folds=pd.DataFrame(folds).sort_values("fold").reset_index(drop=True),
        predictions=pd.concat(preds, ignore_index=True),
    )
