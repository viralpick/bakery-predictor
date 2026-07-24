from bakery.harness.config import ExperimentSpec, DataSpec, WindowSpec
from bakery.harness.runner import run_experiment, ExperimentResult, RunResult, STAGES


def _spec(forecaster=("category_total", "distributional_total"), n_folds=8):
    return ExperimentSpec(
        name="gw_core", data=DataSpec(source="real", store="store_gw01"),
        target="adjusted_demand_unit", forecaster=list(forecaster), layers=["event_prior"],
        event_priors="gwangyo",
        window=WindowSpec(scheme="expanding", n_folds=n_folds, window_days=730, horizon_days=7),
        alpha=0.8, production_q=0.85,
    )


def test_run_returns_experiment_result(tmp_path):
    result = run_experiment(_spec(), out_dir=tmp_path / "out", cache_dir=None)
    assert isinstance(result, ExperimentResult)
    assert result.name == "gw_core"
    assert set(result.runs) == {"category_total", "distributional_total"}
    assert all(isinstance(r, RunResult) for r in result.runs.values())
    assert len(result.comparison) == 2
    assert {"forecaster", "wape"}.issubset(result.comparison.columns)


def test_run_writes_artifacts(tmp_path):
    run_experiment(_spec(), out_dir=tmp_path / "out", cache_dir=None)
    d = tmp_path / "out" / "gw_core"
    assert (d / "config_resolved.yaml").exists()
    assert (d / "comparison.csv").exists()
    for fname in ("category_total", "distributional_total"):
        assert (d / fname / "predictions.csv").exists()
        assert (d / fname / "metrics.json").exists()


def test_features_cache_hit(tmp_path):
    cache = tmp_path / "cache"
    t1, t2 = [], []
    spec = _spec(forecaster=("category_total",))   # 캐시만 검증 — 빠른 단일 forecaster
    run_experiment(spec, out_dir=tmp_path / "o1", cache_dir=cache, _trace=t1)
    run_experiment(spec, out_dir=tmp_path / "o2", cache_dir=cache, _trace=t2)
    assert ("features", "miss") in t1
    assert ("features", "hit") in t2


def test_unrunnable_skipped_with_warning(tmp_path):
    import pytest
    spec = _spec(forecaster=("category_total", "lightgbm_v2"))
    with pytest.warns(UserWarning, match="lightgbm_v2"):
        result = run_experiment(spec, out_dir=tmp_path / "out", cache_dir=None)
    assert set(result.runs) == {"category_total"}


def test_stages_constant():
    assert STAGES == ("features", "backtest", "evaluate")
