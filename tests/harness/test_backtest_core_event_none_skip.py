import numpy as np

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.backtest_core import windowed_backtest
from bakery.harness.registry import build_forecaster


def test_events_none_skips_prior_same_as_empty():
    """events=None은 events={}와 동일하게 event_prior 미적용(xmas fallback 금지)."""
    feat = build_features(build_category_daily(alpha=0.8), target_col="adjusted_demand_unit")
    kw = dict(window_days=730, target_col="adjusted_demand_unit", n_folds=52,
              horizon_days=7, production_q=0.85, alpha=0.8)
    fc = build_forecaster("category_total")
    r_none = windowed_backtest(feat, forecaster=fc, events=None, lunar_events=None, **kw)
    r_empty = windowed_backtest(feat, forecaster=fc, events={}, lunar_events={}, **kw)
    np.testing.assert_allclose(
        r_none.predictions["expected"].to_numpy(),
        r_empty.predictions["expected"].to_numpy(), rtol=1e-9)
