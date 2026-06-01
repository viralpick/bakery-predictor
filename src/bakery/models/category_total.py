"""Stage 1: Category-Total Demand LightGBM (unit + revenue 병행)."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd


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

    def predict_expected(self, X: pd.DataFrame) -> np.ndarray:
        return self.expected.predict(X[self.feature_cols])

    def predict_production(self, X: pd.DataFrame) -> np.ndarray:
        return self.quantile.predict(X[self.feature_cols])


def select_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    return [c for c in df.columns if c not in ("date", target_col, *LEAK_COLS)]


def fit_category_total(
    train: pd.DataFrame,
    target_col: str = "adjusted_demand_unit",
    alpha_demand: float = 0.5,
    production_q: float = 0.90,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    random_state: int = 42,
) -> CategoryTotalModel:
    feat_cols = select_feature_cols(train, target_col)
    X = train[feat_cols]
    y = train[target_col]
    common = dict(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, num_leaves=num_leaves,
        random_state=random_state, verbosity=-1,
    )
    expected = lgb.LGBMRegressor(objective="regression_l1", **common).fit(X, y)
    quantile = lgb.LGBMRegressor(objective="quantile", alpha=production_q, **common).fit(X, y)
    return CategoryTotalModel(
        expected=expected, quantile=quantile,
        feature_cols=feat_cols,
        alpha_demand=alpha_demand, production_q=production_q, target_col=target_col,
    )


@dataclass
class BacktestResult:
    folds: pd.DataFrame
    predictions: pd.DataFrame


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
