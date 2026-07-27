import pandas as pd
import pytest

from bakery.analysis.demand_absorption import AbsorptionResult
from bakery.analysis.lab.handlers.absorption import (
    absorption_verdict,
    demand_absorption,
    results_to_frame,
)
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.result import KIND_HYPOTHESIS


def _res(category, verdict, beta=-0.01):
    return AbsorptionResult(store_id="store_gw01", category_id=category, n=100,
                            beta=beta, se=0.02, ci_low=beta - 0.03, ci_high=beta + 0.03,
                            delta=0.05, verdict=verdict)


def test_results_to_frame_columns_and_arm():
    frame = results_to_frame([_res("bread", "absorb")], arm="real")
    assert frame.columns.tolist() == ["store_id", "category_id", "n", "beta", "se",
                                      "ci_low", "ci_high", "delta", "verdict", "arm"]
    assert frame["arm"].tolist() == ["real"]
    assert frame["beta"].tolist() == [-0.01]


def test_verdict_supports_when_all_gate_categories_absorb():
    results = [_res("bread", "absorb"), _res("pastry", "absorb"), _res("cake", "walkaway")]
    # cake는 게이트 대상 아님(단일품목/시즌) → 판정에 영향 없음
    assert absorption_verdict(results) == "지지 — 게이트 카테고리 2건 전부 absorb, walk-away 0건"


def test_verdict_rejects_on_any_gate_walkaway():
    results = [_res("bread", "walkaway"), _res("pastry", "absorb")]
    assert absorption_verdict(results) == (
        "기각 — walk-away 발견: [('store_gw01', 'bread')] (게이트 2건 중 absorb 1건)")


def test_verdict_inconclusive_when_mixed_without_walkaway():
    results = [_res("bread", "inconclusive"), _res("pastry", "absorb")]
    assert absorption_verdict(results) == (
        "불확실 — 게이트 2건 중 absorb 1건 / inconclusive 1건, walk-away 0건")


def test_verdict_when_no_gate_category_present():
    assert absorption_verdict([_res("cake", "absorb")]) == (
        "불확실 — 게이트 카테고리(bread/pastry) 결과 없음")


def test_horizon_days_param_reaches_placebo_only(stub_inputs):
    """horizon_days는 placebo 전용 kwarg — 공유 params를 그냥 전개하면
    run_absorption이 TypeError를 낸다. 라우팅이 되는지 정확값으로 고정한다."""
    import sys
    sys.path.insert(0, "tests")
    from test_demand_absorption_placebo import _daily

    daily = _daily()
    inputs = stub_inputs(daily=daily, params={"demand_absorption": {"horizon_days": 14}})
    result = demand_absorption(inputs)                       # TypeError 없이 통과해야 한다
    table = dict(result.tables)["results"]
    placebo_n = table[table["arm"] == "placebo"]["n"].tolist()
    real_n = table[table["arm"] == "real"]["n"].tolist()
    # horizon_days=14가 실제로 placebo에만 반영됐는지: placebo n = real n − 14
    assert placebo_n == [n - 14 for n in real_n]


@pytest.mark.slow
def test_handler_tables_equal_primitive_output():
    """핸들러는 프리미티브를 호출만 한다 — 재구현 드리프트가 있으면 이 테스트가 깨진다."""
    from bakery.analysis.demand_absorption import placebo_absorption, run_absorption
    from bakery.analysis.lab.spec import AnalysisSpec

    inputs = AnalysisInputs.from_spec(AnalysisSpec(name="t", data={"source": "real"}))
    result = demand_absorption(inputs)
    assert result.kind == KIND_HYPOTHESIS
    assert len(result.figures) == 2
    table = dict(result.tables)["results"]
    expected = pd.concat([results_to_frame(run_absorption(inputs.daily), arm="real"),
                          results_to_frame(placebo_absorption(inputs.daily), arm="placebo")],
                         ignore_index=True)
    pd.testing.assert_frame_equal(table, expected)


@pytest.mark.slow
def test_handler_gate_summary_counts_match_results():
    from bakery.analysis.demand_absorption import GATE_CATEGORIES
    from bakery.analysis.lab.spec import AnalysisSpec

    inputs = AnalysisInputs.from_spec(AnalysisSpec(name="t", data={"source": "real"}))
    result = demand_absorption(inputs)
    tables = dict(result.tables)
    real = tables["results"].query("arm == 'real'")
    gate = real[real["category_id"].isin(GATE_CATEGORIES)]
    summary = tables["gate_summary"]
    assert summary["n_gate"].iloc[0] == len(gate)
    assert summary["n_walkaway"].iloc[0] == int((gate["verdict"] == "walkaway").sum())
