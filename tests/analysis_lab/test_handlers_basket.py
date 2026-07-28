import pandas as pd
import pytest

from bakery.analysis.lab.handlers.basket import (
    ASSUMPTION_THRESHOLDS,
    modeling_v4_assumptions,
    v4_verdict,
)
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.result import KIND_HYPOTHESIS


def _table(supported, n_observations=None):
    n_observations = n_observations or {}
    return pd.DataFrame([{"assumption": key, "statistic": "s", "value": 0.1,
                          "threshold": 0.5, "is_supported": flag,
                          "n_observations": n_observations.get(key, 5)}
                         for key, flag in supported.items()])


def test_thresholds_are_declared_for_four_assumptions():
    assert sorted(ASSUMPTION_THRESHOLDS) == ["1-1-b", "2-1-a", "2-1-b", "basket"]
    assert ASSUMPTION_THRESHOLDS["1-1-b"] == 0.7
    assert ASSUMPTION_THRESHOLDS["2-1-a"] == 0.05
    assert ASSUMPTION_THRESHOLDS["2-1-b"] == 0.10
    assert ASSUMPTION_THRESHOLDS["basket"] == 0.30


def test_verdict_supports_when_all_pass():
    verdict = v4_verdict(_table({"1-1-b": True, "2-1-a": True, "2-1-b": True,
                                "basket": True}))
    assert verdict == "지지 — 4가정 전부 통과(v4 카테고리 합 → 품목 비율 설계 정당)"


def test_verdict_lists_failed_assumptions():
    verdict = v4_verdict(_table({"1-1-b": True, "2-1-a": False, "2-1-b": False,
                                "basket": True}))
    assert verdict == "부분 지지 — 4가정 중 2건 통과, 미통과: ['2-1-a', '2-1-b']"


def test_verdict_rejects_when_none_pass():
    verdict = v4_verdict(_table({"1-1-b": False, "2-1-a": False, "2-1-b": False,
                                "basket": False}))
    assert verdict == "기각 — 4가정 전부 미통과"


def test_verdict_flags_zero_observation_when_supported_assumption_has_no_data():
    """2-1-b가 통과처럼 보여도 관측 0건이면 '통과'가 아니라 '미검증'임을 verdict가 밝혀야 한다."""
    table = _table({"1-1-b": True, "2-1-a": True, "2-1-b": True, "basket": True},
                   n_observations={"2-1-b": 0})
    verdict = v4_verdict(table)
    assert verdict == ("지지 — 4가정 전부 통과(v4 카테고리 합 → 품목 비율 설계 정당) "
                       "(관측 0건: ['2-1-b'] — 통과가 아니라 미검증)")


@pytest.mark.slow
def test_handler_reports_all_four_assumptions():
    from bakery.analysis.lab.spec import AnalysisSpec

    inputs = AnalysisInputs.from_spec(AnalysisSpec(name="t", data={"source": "real"}))
    result = modeling_v4_assumptions(inputs)
    assert result.kind == KIND_HYPOTHESIS
    table = dict(result.tables)["assumptions"]
    assert sorted(table["assumption"].tolist()) == ["1-1-b", "2-1-a", "2-1-b", "basket"]
    assert table["is_supported"].dtype == bool
    assert "n_observations" in table.columns
