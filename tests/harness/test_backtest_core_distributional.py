import numpy as np

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.backtest_core import windowed_backtest
from bakery.harness.event_priors import resolve_event_priors
from bakery.harness.forecasters import DistributionalTotalForecaster


def _feat():
    return build_features(build_category_daily(alpha=0.8), target_col="adjusted_demand_unit")


def test_distributional_through_windowed_backtest_deterministic():
    feat = _feat()
    events, lunar = resolve_event_priors("gwangyo")
    kw = dict(window_days=730, n_folds=8, production_q=0.85, alpha=0.8,
              events=events, lunar_events=lunar)
    r1 = windowed_backtest(feat, forecaster=DistributionalTotalForecaster(), **kw)
    r2 = windowed_backtest(feat, forecaster=DistributionalTotalForecaster(), **kw)
    p1 = r1.predictions.sort_values(["fold", "date"]).reset_index(drop=True)
    p2 = r2.predictions.sort_values(["fold", "date"]).reset_index(drop=True)
    np.testing.assert_array_equal(p1["expected"].to_numpy(), p2["expected"].to_numpy())
    np.testing.assert_array_equal(p1["production"].to_numpy(), p2["production"].to_numpy())


def test_distributional_wape_sane():
    feat = _feat()
    events, lunar = resolve_event_priors("gwangyo")
    r = windowed_backtest(feat, forecaster=DistributionalTotalForecaster(),
                          window_days=730, n_folds=8, production_q=0.85, alpha=0.8,
                          events=events, lunar_events=lunar)
    p = r.predictions
    wape = np.abs(p["actual"] - p["expected"]).sum() / max(np.abs(p["actual"]).sum(), 1)
    assert 0.0 < wape < 1.0
    assert (p["production"] >= p["expected"]).mean() > 0.9   # production은 대체로 expected 이상
