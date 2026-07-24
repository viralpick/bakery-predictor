import sys
import numpy as np

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.event_priors import resolve_event_priors


def _feat():
    cd = build_category_daily(alpha=0.8)                    # None→canonical 3cat
    return build_features(cd, target_col="adjusted_demand_unit")


def test_core_matches_script_windowed_backtest():
    sys.path.insert(0, "scripts")
    import store_predictive_power as s
    from bakery.harness.backtest_core import windowed_backtest as core_wb

    feat = _feat()
    events, lunar = resolve_event_priors("gwangyo")

    legacy = s.windowed_backtest(
        feat, window_days=s.DEFAULT_WINDOW_DAYS, n_folds=s.MAIN_FOLDS,
        events=events, lunar_events=lunar,
    )
    got = core_wb(
        feat, window_days=730, n_folds=52, production_q=0.85, alpha=0.8,
        events=events, lunar_events=lunar,
    )
    lp = legacy.predictions.sort_values(["fold", "date"]).reset_index(drop=True)
    gp = got.predictions.sort_values(["fold", "date"]).reset_index(drop=True)
    assert len(lp) == len(gp)
    np.testing.assert_allclose(gp["expected"].to_numpy(), lp["expected"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(gp["production"].to_numpy(), lp["production"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(gp["actual"].to_numpy(), lp["actual"].to_numpy(), rtol=1e-9)
