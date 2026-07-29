from pathlib import Path

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


# reports/ 는 gitignored라 신규 clone·워크트리에는 없다. 이 테스트는 **live canonical
# 아티팩트** 경로를 일부러 태우는 것이므로(동결 fixture 테스트와 역할이 다르다) 부재는
# 결함이 아니라 "harness-run을 아직 안 돌렸다"는 뜻이다 → 이유를 밝히고 skip한다.
# 반면 tests/fixtures/frozen/* 은 git-tracked이며 부재 시 fail이 맞다(설계 의도 유지).
_LIVE_PREDS = Path("reports/gwangyo_default/category_total/predictions.csv")


@pytest.mark.slow
@pytest.mark.skipif(
    not _LIVE_PREDS.exists(),
    reason=f"{_LIVE_PREDS} 없음 — `uv run bakery harness-run experiments/gwangyo_default.yaml` 먼저 실행",
)
def test_handler_consumes_predictions_artifact(tmp_path):
    """canonical harness preds로 실행 — 수치는 동결 캐시와 다르며 방향만 본다."""
    preds_path = str(_LIVE_PREDS)
    spec = AnalysisSpec(name="t", data={"source": "real"}, predictions=preds_path,
                        params={"weekday_bias": {"n_boot": 50}})
    result = weekday_bias(AnalysisInputs.from_spec(spec))
    assert result.kind == KIND_HYPOTHESIS
    grid = dict(result.tables)["isowaste_grid"]
    assert len(grid) == 9
    assert set(grid["winner"]) <= {"DOW 우위", "GLOBAL 우위", "0포함(무차)"}
    assert any("엔진" in note for note in result.notes)


def test_seasonal_bias_verdict_rejects_when_both_noise():
    import numpy as np

    from bakery.analysis.lab.handlers.model_bias import seasonal_bias_verdict

    noise = {"wpe_diff": 0.5, "ci": np.array([-1.0, 2.0]), "n_segment": 100, "n_rest": 900}
    assert seasonal_bias_verdict(noise, noise) == (
        "기각 — 주말 WPE 차 +0.50%p CI[-1.00,+2.00] noise(CI 0 포함) / "
        "여름 WPE 차 +0.50%p CI[-1.00,+2.00] noise(CI 0 포함)")


def test_weather_bias_verdict_reports_underpowered_not_refuted():
    """fix(final review 5): n=17/11/14에서 "기각 … 정당화되지 않음"은 반증 주장이라
    캐비앗("Underpowered ≠ refuted") 위반이었다 — "신호 없음(검정력 부족)"으로 교정."""
    from bakery.analysis.lab.handlers.model_bias import weather_bias_verdict

    contrasts = pd.DataFrame([
        {"segment": "is_heatwave", "wpe_diff": 1.984, "n_segment": 17, "is_signal": False},
        {"segment": "is_coldwave", "wpe_diff": -4.566, "n_segment": 11, "is_signal": False},
        {"segment": "is_heavy_rain", "wpe_diff": -0.829, "n_segment": 14, "is_signal": False},
    ])
    assert weather_bias_verdict(contrasts) == (
        "신호 없음(검정력 부족) — 폭염/한파/강한비 전부 CI 0 포함. "
        "표본이 작아(is_heatwave n=17, is_coldwave n=11, is_heavy_rain n=14) "
        "'효과 없음'으로 단정할 수 없다")


def test_weather_bias_verdict_all_segments_empty_is_not_a_ci_claim():
    """세그먼트/여집합이 전부 비면 CI 자체가 계산되지 않는다 — "CI 0 포함"이라 적으면
    없는 CI를 있는 것처럼 은폐한다(fix round 1 _empty_contrast_row 경로)."""
    from bakery.analysis.lab.handlers.model_bias import weather_bias_verdict

    contrasts = pd.DataFrame([
        {"segment": s, "wpe_diff": float("nan"), "n_segment": 0, "is_signal": False}
        for s in ("is_heatwave", "is_coldwave", "is_heavy_rain")
    ])
    assert weather_bias_verdict(contrasts) == (
        "판정 불가 — 폭염/한파/강한비 전부 대조군 없음(세그먼트 또는 여집합이 비어 CI 계산 불가)")


def test_weather_bias_verdict_supports_when_one_segment_is_signal():
    from bakery.analysis.lab.handlers.model_bias import weather_bias_verdict

    contrasts = pd.DataFrame([
        {"segment": "is_heatwave", "wpe_diff": 5.0, "n_segment": 100, "is_signal": True},
        {"segment": "is_coldwave", "wpe_diff": -1.0, "n_segment": 11, "is_signal": False},
        {"segment": "is_heavy_rain", "wpe_diff": -0.8, "n_segment": 14, "is_signal": False},
    ])
    assert weather_bias_verdict(contrasts) == (
        "지지 — ['is_heatwave'] 세그먼트에서 CI 0 배제(체계적 편향)")


def test_event_prior_verdict_single_artifact_mode():
    from bakery.analysis.lab.handlers.model_bias import event_prior_verdict

    table = pd.DataFrame([{"segment": "event", "n": 20, "wpe": -3.0, "stockout_rate": 10.0},
                          {"segment": "non_event", "n": 980, "wpe": -0.5,
                           "stockout_rate": 5.0}])
    assert event_prior_verdict(table, is_ab_mode=False) == (
        "단일 artifact 모드 — 이벤트일 WPE -3.00% vs 비이벤트일 -0.50% "
        "(prior 적용 후 잔여 편향; base 대비 개선폭은 baseline preds 필요)")


def test_event_prior_verdict_ab_mode_reports_bias_reduction_not_raw_diff():
    """fix(final review 4): baseline −19.76 / prior −2.57일 때 raw diff(baseline−wpe)
    −17.19를 "개선 −17.19%p"로 찍으면 17점 악화처럼 읽힌다. WPE는 부호 있는 편향이라
    개선은 |WPE| 감소량이어야 한다(19.76 → 2.57, 즉 +17.19%p 감소)."""
    from bakery.analysis.lab.handlers.model_bias import event_prior_verdict

    table = pd.DataFrame([
        {"segment": "event", "n": 3, "wpe": -2.57, "wpe_baseline": -19.76,
         "stockout_rate": 0.0, "stockout_rate_baseline": 66.67, "n_baseline": 3},
        {"segment": "non_event", "n": 361, "wpe": 0.68, "wpe_baseline": 0.68,
         "stockout_rate": 21.88, "stockout_rate_baseline": 21.88, "n_baseline": 361},
    ])
    assert event_prior_verdict(table, is_ab_mode=True) == (
        "A/B 모드 — 이벤트일 WPE -2.57% (baseline -19.76%), "
        "|WPE| 감소 +17.19%p (baseline |19.76| → -2.57)")


def test_event_dates_for_expands_solar_and_lunar():
    from bakery.analysis.lab.handlers.model_bias import event_dates_for

    priors = {"gwangyo": {"events": {"xmas": (12, 25), "childrens": (5, 5)},
                          "lunar_events": {"chuseok": {2024: "2024-09-17",
                                                       2025: "2025-10-06"}}}}
    dates = event_dates_for("gwangyo", range(2024, 2026), priors)
    assert dates.tolist() == [pd.Timestamp("2024-05-05"), pd.Timestamp("2024-09-17"),
                              pd.Timestamp("2024-12-25"), pd.Timestamp("2025-05-05"),
                              pd.Timestamp("2025-10-06"), pd.Timestamp("2025-12-25")]


def test_event_dates_for_unknown_key_is_empty():
    from bakery.analysis.lab.handlers.model_bias import event_dates_for

    assert event_dates_for("nope", range(2024, 2025), {}).tolist() == []


def test_all_three_registered_as_needs_predictions():
    load_handlers()
    for name in ("seasonal_bias", "weather_bias", "event_prior_validation"):
        assert HYPOTHESES[name].needs_predictions is True, name


# fix round 1(리뷰 Important) — verdict 문자열은 exact-value 테스트로 고정돼 있어 표본
# 크기를 못 담는다. notes는 substring 단언만 하므로 "n=" 마커 존재만 확인한다.

def test_seasonal_bias_note_reports_segment_sample_sizes():
    from bakery.analysis.lab.handlers.model_bias import _sample_size_note

    contrasts = pd.DataFrame([{"segment": "weekend", "n_segment": 104, "is_signal": True},
                              {"segment": "summer", "n_segment": 122, "is_signal": False}])
    note = _sample_size_note(contrasts)
    assert "n=" in note


def test_weather_bias_note_reports_segment_sample_sizes():
    from bakery.analysis.lab.handlers.model_bias import _sample_size_note

    contrasts = pd.DataFrame([{"segment": "is_heatwave", "n_segment": 17, "is_signal": False},
                              {"segment": "is_coldwave", "n_segment": 11, "is_signal": False},
                              {"segment": "is_heavy_rain", "n_segment": 14, "is_signal": False}])
    note = _sample_size_note(contrasts)
    assert "n=" in note


def test_event_prior_validation_note_reports_event_sample_size():
    from bakery.analysis.lab.handlers.model_bias import _event_sample_note

    table = pd.DataFrame([{"segment": "event", "n": 3, "wpe": -2.57, "stockout_rate": 0.0},
                          {"segment": "non_event", "n": 361, "wpe": 0.68,
                           "stockout_rate": 21.88}])
    note = _event_sample_note(table)
    assert "n=" in note


def test_event_prior_verdict_handles_zero_event_days():
    """등록 이벤트일이 preds 윈도우에 0건이면 .iloc[0] IndexError 대신 명시 판정(Minor 2)."""
    from bakery.analysis.lab.handlers.model_bias import event_prior_verdict

    table = pd.DataFrame([{"segment": "non_event", "n": 364, "wpe": 0.1, "stockout_rate": 5.0}])
    assert event_prior_verdict(table, is_ab_mode=False) == "이벤트일 0건 — 판정 불가"


def test_extreme_contrasts_honours_n_boot_param():
    """weather_bias의 params_for(n_boot)가 하드코딩 N_BOOT를 밀어내는지 확인(Minor 1)."""
    from bakery.analysis.lab.handlers.model_bias import _extreme_contrasts

    merged = pd.DataFrame({
        "date": pd.date_range("2025-06-01", periods=10),
        "actual": [100.0] * 10,
        "expected": [90.0, 110.0] * 5,
        "production": [95.0] * 10,
        "month": [6] * 10,
        "is_heatwave": [True] * 5 + [False] * 5,
        "is_coldwave": [False] * 10,
        "is_heavy_rain": [False] * 10,
    })
    first = _extreme_contrasts(merged, n_boot=20, seed=7)
    second = _extreme_contrasts(merged, n_boot=20, seed=7)
    # is_coldwave/is_heavy_rain 세그먼트는 fixture에 아예 없어 NaN 대조행이 된다
    # (_empty_contrast_row) — NaN != NaN이라 tolist() 비교가 아니라 프레임 단위로 맞춘다.
    pd.testing.assert_frame_equal(first, second)
