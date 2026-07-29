"""2층 설명 레이어 (v7 5b) — explain_category_total / explain_item_order.

5a forecast_forward seam의 중간값을 소비해 "왜 이 수량?"을 총량·품목 두 층위로
분해 서술한다. 새 모델링 없음 — 엔진이 낸 실제 값을 충실히 분해한다.

★faithfulness: event_prior 기여는 prior_median − base_median(실제 blend 차이)이지
"크리스마스=고정 N개 룰"이 아니다(레벨-앵커 블렌드). apply_policy 퍼센트 safety를
얹지 않는다 — seam our_order가 이미 분위수 버퍼(q0.85)를 총량 레벨에 포함하므로
이중계상이 된다. 기존 explain_order(v6 apply_policy)는 다른 발주 철학, 무변경.

See docs/superpowers/specs/2026-07-27-ontology-explain-layer-design.md.
"""

from __future__ import annotations

import pandas as pd

from ..decision.policy import _round_up_to_unit
from ..forecast.forward import forecast_forward

BATCH_ROUND_UNIT = 3  # 라벨된 가정: 아띠제 배수생산(3/6/9). 품목별 배수는 후속.


def _forward_at_date(store_id, daily, date, horizon_days, use_forecast):
    """forecast_forward를 돌려 요청 date의 category_totals 1행 + item_quantities/proportions 슬라이스.

    daily가 다중 store면 store_id로 필터(forecast_forward=단일-store 가정, synthetic 안전).
    """
    if "store_id" in daily.columns and daily["store_id"].nunique() > 1:
        daily = daily[daily["store_id"] == store_id].copy()
    ff = forecast_forward(store_id, daily=daily, horizon_days=horizon_days, use_forecast=use_forecast)
    ts = pd.Timestamp(date)
    ct = ff.category_totals[pd.to_datetime(ff.category_totals["date"]) == ts]
    if ct.empty:
        raise ValueError(f"date {date} not in forward horizon for {store_id}")
    iq = ff.item_quantities[pd.to_datetime(ff.item_quantities["date"]) == ts].copy()
    iq["item_id"] = iq["item_id"].astype(str)
    props = ff.proportions[pd.to_datetime(ff.proportions["date"]) == ts].copy()
    props["item_id"] = props["item_id"].astype(str)
    return ct.iloc[0], iq, props


# 비중 인자: step 이름 → (원시 근거 컬럼, 설명). 곱셈 인자이며 base_sold에 차례로 곱해진다.
# 근거 = models/item_proportion.compute_proportions (승수 상수도 그쪽이 단일 출처).
_PROPORTION_FACTORS: tuple[tuple[str, str, str], ...] = (
    ("adj_trend", "trend_pct",
     "추세 보정 × (근거=최근 추세율; |추세|≤15%는 noise로 무시, 최대 ±20%)"),
    ("adj_stockout", "avg_stockout_h",
     "매진 회피 보정 × (근거=평균 매진시각; 카테고리 내 매진이 이른 순위일수록 상향, 최대 +20%)"),
    ("adj_closing", "closing_rate",
     "마감할인 과잉 보정 × (근거=마감할인 비율; 카테고리 내 마감이 많은 순위일수록 하향, 최대 −20%)"),
    ("adj_new", "days_since_first",
     "신제품 보정 × (근거=최초 판매 후 경과일; 90일 미만이면 1.2, 아니면 1.0)"),
)


def _factor_rows(seam_row: pd.Series, day_props: pd.DataFrame) -> list[tuple[str, float, float, str]]:
    """비중 인자 분해 행 — (step, value, evidence, detail).

    raw_weight = base_sold × 인자4, weight_sum = 그날 전 품목 raw_weight 합.
    proportion = raw_weight / weight_sum 이 정확히 성립한다(정규화 분모 노출 목적).
    """
    factor_names = [name for name, _, _ in _PROPORTION_FACTORS]
    raw_weight = float(seam_row["base_sold"])
    for name in factor_names:
        raw_weight *= float(seam_row[name])
    weight_sum = float(day_props["base_sold"].mul(
        day_props[factor_names].prod(axis=1)).sum())

    rows = [("base_sold", float(seam_row["base_sold"]), float("nan"),
             "최근 판매 합 — 비중의 출발점(정규화 전 가중치의 base)")]
    rows += [(name, float(seam_row[name]), float(seam_row[evidence_col]), detail)
             for name, evidence_col, detail in _PROPORTION_FACTORS]
    rows += [
        ("raw_weight", raw_weight, float("nan"), "정규화 전 가중치 = base_sold × 인자4"),
        ("weight_sum", weight_sum, float("nan"), "그날 전 품목 raw_weight 합 (정규화 분모)"),
    ]
    return rows


def explain_category_total(store_id, *, daily, date, horizon_days=7, use_forecast=False):
    """카테고리 생산총량 분해: base 예측 → event_prior 보정 → 분위수 버퍼 → prior_prod.

    event_prior = prior_median − base_median (실제 blend 차이, 특수일 레벨-앵커).
    """
    ct, _, _ = _forward_at_date(store_id, daily, date, horizon_days, use_forecast)
    base_median = float(ct["base_median"])
    prior_median = float(ct["prior_median"])
    prior_prod = float(ct["prior_prod"])
    event_prior = prior_median - base_median
    buffer = prior_prod - prior_median
    rows = [
        ("base_median", base_median, "Stage1 카테고리 수요 예측 (event_prior 이전)"),
        ("event_prior", event_prior, "특수일 레벨-앵커 블렌드 보정 (prior_median − base_median)"),
        ("prior_median", prior_median, "보정된 카테고리 수요 (q0.5)"),
        ("quantile_buffer", buffer,
         "생산 버퍼 (prior_prod − prior_median; 비이벤트일=순수 q0.85−q0.5 spread, "
         "이벤트일=event 보정이 spread에도 곱셈 적용됨)"),
        ("prior_prod", prior_prod, "카테고리 생산총량"),
    ]
    return pd.DataFrame(
        [{"store_id": store_id, "date": date, "step": s, "value": v,
          "evidence": float("nan"), "detail": d} for s, v, d in rows]
    )


def explain_item_order(store_id, item_id, *, daily, date, horizon_days=7,
                       round_unit=BATCH_ROUND_UNIT, use_forecast=False):
    """품목 생산량 분해(통합 단일 체인): 카테고리 총량 × 품목 비중 → 배수 라운딩.

    category_total(prior_prod) × proportion = item_order(=seam our_order) →
    ceil_round_unit = final. apply_policy 미사용(이중계상 방지).

    비중은 한 숫자로 뭉개지 않고 인자 5종(base_sold + adj_trend/stockout/closing/new)과
    정규화 분모(weight_sum)까지 노출한다 — "매진이 이른 품목이라 비중을 6% 올렸다"처럼
    근거를 말할 수 있게. 인자는 seam proportions의 실제 계산값이며 재계산하지 않는다.
    """
    ct, iq, props = _forward_at_date(store_id, daily, date, horizon_days, use_forecast)
    row = iq[iq["item_id"] == str(item_id)]
    if row.empty:
        raise ValueError(f"item {item_id} not in forward forecast for {store_id} at {date}")
    prior_prod = float(ct["prior_prod"])
    item_order = float(row["our_order"].iloc[0])
    proportion = item_order / prior_prod if prior_prod > 0 else 0.0
    final = _round_up_to_unit(item_order, round_unit)
    rows = [("category_total", prior_prod, float("nan"), "카테고리 생산총량 (prior_prod)")]
    seam_prop = props[props["item_id"] == str(item_id)]
    if not seam_prop.empty:
        rows += _factor_rows(seam_prop.iloc[0], props)
    rows += [
        ("proportion", proportion, float("nan"), "품목 비중 (= raw_weight / weight_sum)"),
        ("item_order", item_order, float("nan"), "품목 생산량 (= 총량 × 비중)"),
        ("final", final, float("nan"), f"배수 라운딩 (ceil to {round_unit})"),
    ]
    return pd.DataFrame(
        [{"store_id": store_id, "item_id": str(item_id), "date": date,
          "step": s, "value": v, "evidence": e, "detail": d} for s, v, e, d in rows]
    )


def rank_forward_items(store_id, *, daily, date, k=3, horizon_days=7, use_forecast=False):
    """forward our_order 기준 top-k 품목 (q_explain_item 진입점).

    [item_id, our_order] 내림차순. explain_item_order의 품목 선택과 동일 seam.
    """
    _, iq, _ = _forward_at_date(store_id, daily, date, horizon_days, use_forecast)
    ranked = iq.sort_values(["our_order", "item_id"], ascending=[False, True]).head(k)
    return ranked[["item_id", "our_order"]].reset_index(drop=True)
