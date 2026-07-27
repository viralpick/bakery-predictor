import pandas as pd
import pytest

from bakery.analysis.lab.handlers.model_bias import weekday_bias, weekday_bias_verdict
from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import HYPOTHESES, load_handlers
from bakery.analysis.lab.result import KIND_HYPOTHESIS
from bakery.analysis.lab.spec import AnalysisSpec


def _grid(winners):
    return pd.DataFrame([{"w_target": 0.06, "trim": 0.02 + 0.01 * i,
                          "gap_freq": -0.001, "gap_mag": -0.0001,
                          "freq_ci_low": -0.01, "freq_median": -0.001,
                          "freq_ci_high": -0.0005 if w == "DOW 우위" else 0.01,
                          "winner": w} for i, w in enumerate(winners)])


def test_registered_as_needs_predictions():
    load_handlers()
    assert HYPOTHESES["weekday_bias"].needs_predictions is True


def test_verdict_supports_when_any_cell_favors_dow():
    # _grid()는 winners 길이만큼(3행) 만든다 — 실제 9행 격자는 아래 핸들러 통합테스트가 검증.
    verdict = weekday_bias_verdict(_grid(["DOW 우위", "0포함(무차)", "0포함(무차)"]))
    assert verdict == "지지 — 3칸 중 1칸에서 DOW 트림 우위(CI 0 배제). center 보정 가치 있음"


def test_verdict_rejects_when_all_cells_tie():
    verdict = weekday_bias_verdict(_grid(["0포함(무차)", "0포함(무차)", "0포함(무차)"]))
    assert verdict == "기각 — 전 격자에서 CI가 0을 포함. center 보정은 전역 균일 하향을 못 이김"


def test_verdict_reports_global_advantage():
    grid = _grid(["0포함(무차)", "0포함(무차)", "0포함(무차)"])
    grid.loc[0, "winner"] = "GLOBAL 우위"
    verdict = weekday_bias_verdict(grid)
    assert verdict == "기각 — 1칸에서 GLOBAL(전역 균일) 우위, DOW 우위 0칸"


@pytest.mark.slow
def test_handler_consumes_predictions_artifact(tmp_path):
    """canonical harness preds로 실행 — 수치는 동결 캐시와 다르며 방향만 본다."""
    preds_path = "reports/gwangyo_default/category_total/predictions.csv"
    spec = AnalysisSpec(name="t", data={"source": "real"}, predictions=preds_path,
                        params={"weekday_bias": {"n_boot": 50}})
    result = weekday_bias(AnalysisInputs.from_spec(spec))
    assert result.kind == KIND_HYPOTHESIS
    grid = dict(result.tables)["isowaste_grid"]
    assert len(grid) == 9
    assert set(grid["winner"]) <= {"DOW 우위", "GLOBAL 우위", "0포함(무차)"}
    assert any("엔진" in note for note in result.notes)
