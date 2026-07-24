from bakery.harness.config import ExperimentSpec, DataSpec, WindowSpec
from bakery.harness.runner import run_experiment, RunResult, STAGES


def _spec(n_folds=8):   # 8 folds: wrapper/캐시/IO 검증엔 충분(엔진 정확성은 Task 3 equivalence가 52에서 담당)
    return ExperimentSpec(
        name="gw_core", data=DataSpec(source="real", store="store_gw01"),
        target="adjusted_demand_unit", forecaster=["category_total"], layers=["event_prior"],
        event_priors="gwangyo",
        window=WindowSpec(scheme="expanding", n_folds=n_folds, window_days=730, horizon_days=7),
        alpha=0.8, production_q=0.85,
    )


def test_run_returns_runresult(tmp_path):
    result = run_experiment(_spec(), out_dir=tmp_path / "out", cache_dir=None)
    assert isinstance(result, RunResult)
    assert result.name == "gw_core"
    assert not result.predictions.empty
    assert {"date", "expected", "production", "actual"}.issubset(result.predictions.columns)
    assert "wape" in result.metrics


def test_run_writes_artifacts(tmp_path):
    run_experiment(_spec(), out_dir=tmp_path / "out", cache_dir=None)
    d = tmp_path / "out" / "gw_core"
    assert (d / "config_resolved.yaml").exists()
    assert (d / "predictions.csv").exists()
    assert (d / "metrics.json").exists()


def test_features_cache_hit(tmp_path):
    cache = tmp_path / "cache"
    t1, t2 = [], []
    run_experiment(_spec(), out_dir=tmp_path / "o1", cache_dir=cache, _trace=t1)
    run_experiment(_spec(), out_dir=tmp_path / "o2", cache_dir=cache, _trace=t2)
    assert ("features", "miss") in t1
    assert ("features", "hit") in t2


def test_stages_constant():
    assert STAGES == ("features", "backtest", "evaluate")
