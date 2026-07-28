"""가설 — 모델 예측 편향 진단(preds artifact 의존).

★경계: 이 레이어는 모델을 실행하지 않는다. harness-run이 남긴 predictions.csv를
읽기만 하며, spec.predictions가 없으면 runner가 preds_required로 스킵한다.

수치 게이트 주의: 출처 스크립트들은 비-canonical 엔진(store_predictive_power)의
캐시 preds를 썼다. canonical harness preds(category_total + event_prior)로 돌리면
수치가 다르므로 방향/판정만 비교 가능하다 — 동결 입력 대조는 tests/test_order_bias.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult
from bakery.analysis.order_bias import (
    TARGET_DOWS,
    WINNER_DOW,
    WINNER_GLOBAL,
    isowaste_grid,
    waste_rate_of,
)
from bakery.analysis.pred_bias import (
    EXTREME_THRESHOLDS,
    SUMMER_MONTHS,
    WEEKEND_DOW,
    WINTER_MONTHS,
    bias_by_axis,
    is_signal,
    segment_contrast,
)

_NOTE_ENGINE = ("preds는 harness-run(canonical category_total+event_prior) 산출물 — "
                "출처 스크립트가 쓴 비-canonical 엔진 캐시와 수치가 다르다. "
                "판정/방향만 비교하라.")
_NOTE_NO_REFIT = "모델 재학습 없음 — 이미 계산된 expected/actual에 발주 정책 껍질만 씌운 A/B다."
_NOTE_WPE_SIGN = "WPE 부호: (expected−actual)/Σ|actual|. 음수=과소예측(발주부족 방향)."


def weekday_bias_verdict(grid: pd.DataFrame) -> str:
    """격자에서 DOW 트림이 CI 0 배제로 이긴 칸이 하나라도 있으면 지지."""
    n_dow = int((grid["winner"] == WINNER_DOW).sum())
    n_global = int((grid["winner"] == WINNER_GLOBAL).sum())
    if n_dow > 0:
        return (f"지지 — {len(grid)}칸 중 {n_dow}칸에서 DOW 트림 우위(CI 0 배제). "
                "center 보정 가치 있음")
    if n_global > 0:
        return f"기각 — {n_global}칸에서 GLOBAL(전역 균일) 우위, DOW 우위 0칸"
    return "기각 — 전 격자에서 CI가 0을 포함. center 보정은 전역 균일 하향을 못 이김"


def _dow_bias_table(preds: pd.DataFrame) -> pd.DataFrame:
    """요일별 상대편향 — 진단 근거(음수=과대예측)."""
    frame = preds.copy()
    frame["dow"] = pd.to_datetime(frame["date"]).dt.dayofweek
    frame["rel_error"] = (frame["actual"] - frame["expected"]) / frame["actual"]
    return (frame.groupby("dow")
            .agg(n=("rel_error", "size"), rel_mean=("rel_error", "mean"),
                 rel_median=("rel_error", "median"))
            .reset_index())


def _gap_fig(grid: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for w_target, group in grid.groupby("w_target"):
        fig.add_trace(go.Scatter(
            x=group["trim"], y=group["gap_freq"], mode="lines+markers",
            name=f"waste {w_target:.0%}",
            error_y=dict(type="data", symmetric=False,
                         array=group["freq_ci_high"] - group["gap_freq"],
                         arrayminus=group["gap_freq"] - group["freq_ci_low"]),
        ))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title="iso-waste 매진빈도 gap (DOW − GLOBAL), 음수=DOW 우위",
                      xaxis_title="대상요일 트림", yaxis_title="gap (빈도 차)")
    return fig


def _dow_bias_fig(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=table["dow"], y=table["rel_mean"]))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title="요일별 상대오차 평균 (음수=과대예측)",
                      xaxis_title="요일(월=0)", yaxis_title="(actual−expected)/actual")
    return fig


@register_hypothesis("weekday_bias", "평일(월·수) 과대예측 트림의 iso-waste 가치",
                     needs_predictions=True)
def weekday_bias(inputs: AnalysisInputs) -> AnalysisResult:
    preds = inputs.predictions
    grid = isowaste_grid(preds, **inputs.params_for("weekday_bias"))
    base_waste = waste_rate_of(preds["expected"].to_numpy(), preds["actual"].to_numpy())
    return AnalysisResult(
        name="weekday_bias", kind=KIND_HYPOTHESIS,
        title="평일(월·수) 과대예측 트림의 iso-waste 가치",
        tables=[("isowaste_grid", grid), ("dow_bias", _dow_bias_table(preds))],
        figures=[_gap_fig(grid), _dow_bias_fig(_dow_bias_table(preds))],
        verdict=weekday_bias_verdict(grid),
        notes=[_NOTE_ENGINE, _NOTE_NO_REFIT,
               f"base(expected) waste={base_waste:.3f}, 대상요일={TARGET_DOWS}(월·수)"],
    )


def _with_axes(preds: pd.DataFrame) -> pd.DataFrame:
    frame = preds.copy()
    dates = pd.to_datetime(frame["date"])
    frame["dow"] = dates.dt.dayofweek
    frame["month"] = dates.dt.month
    frame["is_weekend"] = frame["dow"].isin(WEEKEND_DOW)
    return frame


def seasonal_bias_verdict(weekend: dict, summer: dict) -> str:
    parts = []
    for label, contrast in (("주말", weekend), ("여름", summer)):
        low, high = contrast["ci"]
        state = "신호" if is_signal(contrast) else "noise(CI 0 포함)"
        parts.append(f"{label} WPE 차 {contrast['wpe_diff']:+.2f}%p "
                     f"CI[{low:+.2f},{high:+.2f}] {state}")
    prefix = ("지지" if is_signal(weekend) or is_signal(summer) else "기각")
    return f"{prefix} — " + " / ".join(parts)


@register_hypothesis("seasonal_bias", "주말·여름 계절 편향(WPE 축 분해)",
                     needs_predictions=True)
def seasonal_bias(inputs: AnalysisInputs) -> AnalysisResult:
    preds = _with_axes(inputs.predictions)
    params = inputs.params_for("seasonal_bias")
    weekend = segment_contrast(preds, preds["is_weekend"], **params)
    summer = segment_contrast(preds, preds["month"].isin(SUMMER_MONTHS), **params)
    contrasts = pd.DataFrame([
        {"segment": "weekend", "wpe_diff": weekend["wpe_diff"],
         "ci_low": weekend["ci"][0], "ci_high": weekend["ci"][1],
         "n_segment": weekend["n_segment"], "is_signal": is_signal(weekend)},
        {"segment": "summer", "wpe_diff": summer["wpe_diff"],
         "ci_low": summer["ci"][0], "ci_high": summer["ci"][1],
         "n_segment": summer["n_segment"], "is_signal": is_signal(summer)},
    ])
    return AnalysisResult(
        name="seasonal_bias", kind=KIND_HYPOTHESIS, title="주말·여름 계절 편향(WPE 축 분해)",
        tables=[("by_dow", bias_by_axis(preds, "dow")),
                ("by_month", bias_by_axis(preds, "month")),
                ("contrasts", contrasts)],
        figures=[_axis_fig(bias_by_axis(preds, "dow"), "dow", "요일별 WPE"),
                 _axis_fig(bias_by_axis(preds, "month"), "month", "월별 WPE")],
        verdict=seasonal_bias_verdict(weekend, summer),
        notes=[_NOTE_ENGINE, _NOTE_WPE_SIGN, _sample_size_note(contrasts)],
    )


def _axis_fig(table: pd.DataFrame, axis: str, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=table[axis].astype(str), y=table["wpe"]))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title=title, xaxis_title=axis, yaxis_title="WPE %")
    return fig


def _sample_size_note(contrasts: pd.DataFrame) -> str:
    """세그먼트별 표본 크기(n_segment)를 note에 노출한다.

    ★fix round 1(리뷰 Important): CI가 0을 포함하는 결과를 verdict 문자열에서
    "기각"이라 적으면, n이 작아 검정력이 부족했을 뿐인 경우도 "효과 없음이 확정됐다"로
    오독된다. verdict은 exact-value 테스트로 고정돼 있어 여기서 못 고치므로, note에
    표본 크기 + 해석 주의를 명시해 다운스트림(리포트 HTML)에서 verdict만 보고
    확정 기각으로 인용하지 못하게 한다.
    """
    sizes = ", ".join(f"{row.segment} n={int(row.n_segment)}" for row in contrasts.itertuples())
    return (f"세그먼트 표본: {sizes}. CI가 0을 포함하는 것은 '효과 없음'이 아니라 "
            "이 표본에서 **검정력이 부족**하다는 뜻이다 — 확정 기각으로 인용하지 말 것.")


def _weather_station_id(inputs: AnalysisInputs, params: dict) -> int:
    """params override 없으면 매장 기본 ASOS 관측소(ingest/store_mapping.py)를 쓴다.

    ★필터 없이 merge하면 관측소별 행이 중복 매칭돼 wpe_percent 합산이 배로 부풀려진다
    (weather_observed에 station_id 108/119 두 관측소가 섞여 있음, 2026-07-28 확인) —
    반드시 단일 관측소로 좁혀야 한다.
    """
    if "station_id" in params:
        return int(params["station_id"])
    from bakery.analysis.lab.inputs import GWANGYO
    from bakery.ingest.store_mapping import load_store_mapping

    key = GWANGYO if inputs.is_multistore else inputs.store
    return load_store_mapping()[key]["station_id"]


def _weather_segments(preds: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = preds.merge(weather, on="date", how="left")
    merged["is_heatwave"] = merged["maxTa"] >= EXTREME_THRESHOLDS["heatwave_max_ta"]
    merged["is_coldwave"] = merged["minTa"] <= EXTREME_THRESHOLDS["coldwave_min_ta"]
    merged["is_heavy_rain"] = merged["sumRn"] >= EXTREME_THRESHOLDS["heavy_rain_mm"]
    return merged


_EXTREME_SEGMENTS: tuple[tuple[str, tuple[int, ...] | None], ...] = (
    ("is_heatwave", SUMMER_MONTHS),      # 비교군 = 동계절 비극한일
    ("is_coldwave", WINTER_MONTHS),
    ("is_heavy_rain", None),             # 강수는 전 계절
)


def _empty_contrast_row(segment: str, n_segment: int) -> dict:
    """세그먼트/여집합 한쪽이 비면 대조 불가 — noise로 두되 n을 남겨 은폐하지 않는다."""
    return {"segment": segment, "wpe_diff": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "n_segment": n_segment, "is_signal": False}


def _extreme_contrasts(merged: pd.DataFrame, **contrast_kwargs) -> pd.DataFrame:
    """극한날씨 3세그먼트 × 동계절 비교군 WPE 대조.

    contrast_kwargs는 segment_contrast로 그대로 전달(n_boot/seed) — fix round 1(Minor 1):
    seasonal_bias는 params_for를 통해 n_boot을 조정할 수 있는데 이 함수만 N_BOOT
    하드코딩이라 일관성이 깨져 있었다.
    """
    rows = []
    for segment, season in _EXTREME_SEGMENTS:
        scope = merged if season is None else merged[merged["month"].isin(season)]
        mask = scope[segment].fillna(False)
        if mask.sum() == 0 or (~mask).sum() == 0:
            rows.append(_empty_contrast_row(segment, int(mask.sum())))
            continue
        contrast = segment_contrast(scope, mask, **contrast_kwargs)
        rows.append({"segment": segment, "wpe_diff": contrast["wpe_diff"],
                     "ci_low": contrast["ci"][0], "ci_high": contrast["ci"][1],
                     "n_segment": contrast["n_segment"],
                     "is_signal": is_signal(contrast)})
    return pd.DataFrame(rows)


_BOOTSTRAP_PARAM_NAMES = ("n_boot", "seed")


def _bootstrap_kwargs(params: dict) -> dict:
    """params_for("weather_bias")에서 segment_contrast용 부트스트랩 인자만 걸러낸다.

    params에는 station_id도 섞여 있을 수 있어(대상은 _weather_station_id) 그대로
    **params를 넘기면 segment_contrast가 모르는 키워드 인자로 TypeError가 난다.
    """
    return {name: params[name] for name in _BOOTSTRAP_PARAM_NAMES if name in params}


def weather_bias_verdict(contrasts: pd.DataFrame) -> str:
    signals = contrasts[contrasts["is_signal"]]["segment"].tolist()
    if not signals:
        return ("기각 — 폭염/한파/강한비 전부 CI 0 포함(noise). "
                "극한날씨 전용 feature는 정당화되지 않음")
    return f"지지 — {signals} 세그먼트에서 CI 0 배제(체계적 편향)"


@register_hypothesis("weather_bias", "극한날씨(폭염·한파·강한비) 편향",
                     needs_predictions=True)
def weather_bias(inputs: AnalysisInputs) -> AnalysisResult:
    """★경계: weather_observed는 AnalysisInputs를 거치지 않고 paths.dataset()을
    직접 읽는다(다른 핸들러는 전부 inputs.* 캐시 경유) — 날씨는 preds와 별개
    소스이며 station 필터가 필요해 AnalysisInputs 공용 캐시에 넣지 않았다.
    """
    from bakery.data import paths

    params = inputs.params_for("weather_bias")
    weather = pd.read_parquet(paths.dataset("weather_observed"))
    weather["date"] = pd.to_datetime(weather["date"])
    for column in ("maxTa", "minTa", "sumRn"):
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    station = _weather_station_id(inputs, params)
    weather = weather[weather["station_id"] == station]
    merged = _weather_segments(_with_axes(inputs.predictions),
                              weather[["date", "maxTa", "minTa", "sumRn"]])
    contrasts = _extreme_contrasts(merged, **_bootstrap_kwargs(params))
    fig = go.Figure(go.Bar(
        x=contrasts["segment"], y=contrasts["wpe_diff"],
        error_y=dict(type="data", symmetric=False,
                     array=contrasts["ci_high"] - contrasts["wpe_diff"],
                     arrayminus=contrasts["wpe_diff"] - contrasts["ci_low"])))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title="극한날씨 세그먼트 WPE 차(동계절 비극한 대비)",
                      yaxis_title="WPE 차 %p")
    return AnalysisResult(
        name="weather_bias", kind=KIND_HYPOTHESIS, title="극한날씨(폭염·한파·강한비) 편향",
        tables=[("contrasts", contrasts)], figures=[fig],
        verdict=weather_bias_verdict(contrasts),
        notes=[_NOTE_ENGINE, _NOTE_WPE_SIGN,
               f"임계: {EXTREME_THRESHOLDS} / 비교군은 동계절 비극한일이다. "
               f"관측소 station_id={station}.",
               _sample_size_note(contrasts)],
    )


def _baseline_segment_table(baseline_path, event_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """prior 없는 baseline artifact를 같은 세그먼트 축으로 집계해 A/B 대조군을 만든다."""
    baseline = pd.read_csv(baseline_path)
    baseline["date"] = pd.to_datetime(baseline["date"])
    baseline["segment"] = np.where(baseline["date"].isin(event_dates), "event", "non_event")
    return bias_by_axis(baseline, "segment").rename(
        columns={"wpe": "wpe_baseline", "stockout_rate": "stockout_rate_baseline",
                 "n": "n_baseline"})


def event_prior_verdict(table: pd.DataFrame, *, is_ab_mode: bool) -> str:
    event_rows = table[table["segment"] == "event"]
    if event_rows.empty:
        # fix round 1(Minor 2): 등록 이벤트일이 preds 윈도우에 0건이면 .iloc[0]가
        # IndexError를 던지고 runner가 "error:"로 삼켜 스킵처럼 보인다 — 명시적 판정으로 대체.
        return "이벤트일 0건 — 판정 불가"
    event = event_rows.iloc[0]
    if is_ab_mode:
        return (f"A/B 모드 — 이벤트일 WPE {event['wpe']:+.2f}% "
                f"(baseline {event['wpe_baseline']:+.2f}%), "
                f"개선 {event['wpe_baseline'] - event['wpe']:+.2f}%p")
    non_event = table[table["segment"] == "non_event"].iloc[0]
    return (f"단일 artifact 모드 — 이벤트일 WPE {event['wpe']:+.2f}% vs "
            f"비이벤트일 {non_event['wpe']:+.2f}% "
            f"(prior 적용 후 잔여 편향; base 대비 개선폭은 baseline preds 필요)")


def _event_sample_note(table: pd.DataFrame) -> str:
    """이벤트일 표본 크기를 note에 노출 — fix round 1(리뷰 Important).

    n=3처럼 작은 표본에서 나온 개선폭을 verdict 문자열만 보고 정밀한 효과 크기로
    인용하면 안 된다. 방향(부호)은 신뢰할 수 있어도 크기는 아니다.
    """
    event_rows = table[table["segment"] == "event"]
    if event_rows.empty:
        return "이벤트일 n=0 — 표본 없음(등록 prior 이벤트일만 집계), 판정 불가."
    n = int(event_rows.iloc[0]["n"])
    return (f"이벤트일 n={n} — 방향은 뚜렷하나 표본이 작아 정확한 효과 크기로 인용하면 "
            "안 된다(등록 prior 이벤트일만 집계).")


@register_hypothesis("event_prior_validation", "이벤트 prior 적용 후 이벤트일 정확도",
                     needs_predictions=True)
def event_prior_validation(inputs: AnalysisInputs) -> AnalysisResult:
    """★재정의(Task 17): 출처 scripts/verify_event_prior.py는 model.predict_expected()를
    실행해 base vs prior A/B를 했다 — 이 레이어는 모델을 실행하지 않으므로 대신
    (1) baseline_predictions params가 있으면 artifact 대 artifact A/B,
    (2) 없으면 단일 artifact(이벤트일 vs 비이벤트일) 대조로 대체한다.
    """
    from bakery.harness.event_priors import STORE_EVENT_PRIORS

    params = inputs.params_for("event_prior_validation")
    preds = _with_axes(inputs.predictions)
    years = range(int(preds["date"].dt.year.min()), int(preds["date"].dt.year.max()) + 1)
    event_dates = event_dates_for(inputs.prior_key, years, STORE_EVENT_PRIORS)
    preds["segment"] = np.where(preds["date"].isin(event_dates), "event", "non_event")
    table = bias_by_axis(preds, "segment")
    baseline_path = params.get("baseline_predictions")
    is_ab_mode = baseline_path is not None and Path(baseline_path).exists()
    if is_ab_mode:
        table = table.merge(_baseline_segment_table(baseline_path, event_dates),
                            on="segment", how="left")
    fig = go.Figure(go.Bar(x=table["segment"], y=table["wpe"]))
    fig.add_hline(y=0.0, line_dash="dash")
    fig.update_layout(title="이벤트일 vs 비이벤트일 WPE", yaxis_title="WPE %")
    return AnalysisResult(
        name="event_prior_validation", kind=KIND_HYPOTHESIS,
        title="이벤트 prior 적용 후 이벤트일 정확도",
        tables=[("by_segment", table)], figures=[fig],
        verdict=event_prior_verdict(table, is_ab_mode=is_ab_mode),
        notes=[_NOTE_ENGINE, _NOTE_WPE_SIGN,
               ("모델을 실행하지 않으므로 base vs prior A/B는 baseline preds artifact"
                "(layers: [] 로 돌린 harness-run 산출)가 있을 때만 가능하다."),
               f"A/B 모드: {is_ab_mode}", _event_sample_note(table)],
    )


def _solar_event_dates(events: dict, years: range) -> list[pd.Timestamp]:
    """{이벤트명: (월, 일)} → 연도별 날짜로 전개."""
    dates = []
    for month, day in events.values():
        dates += [pd.Timestamp(year=year, month=month, day=day) for year in years]
    return dates


def _lunar_event_dates(lunar_events: dict) -> list[pd.Timestamp]:
    """{이벤트명: {연도: 'YYYY-MM-DD'}} → 날짜 리스트."""
    dates = []
    for per_year in lunar_events.values():
        dates += [pd.Timestamp(value) for value in per_year.values()]
    return dates


def event_dates_for(prior_key: str, years: range, store_event_priors: dict) -> pd.DatetimeIndex:
    """등록된 prior 이벤트일만(공휴일 전체가 아니다 — prior가 실제로 손대는 날짜)."""
    config = store_event_priors.get(prior_key, {})
    dates = _solar_event_dates(config.get("events") or {}, years)
    dates += _lunar_event_dates(config.get("lunar_events") or {})
    return pd.DatetimeIndex(sorted(set(dates)))
