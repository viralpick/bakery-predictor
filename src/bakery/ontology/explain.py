"""2층 설명 레이어 (v7 5b) — explain_category_total / explain_item_order.

5a forecast_forward seam의 중간값을 소비해 "왜 이 수량?"을 총량·품목 두 층위로
분해 서술한다. 새 모델링 없음 — 엔진이 낸 실제 값을 충실히 분해한다.

★faithfulness: event_prior 기여는 prior_median − base_median(실제 blend 차이)이지
"크리스마스=고정 N개 룰"이 아니다(레벨-앵커 블렌드). apply_policy 퍼센트 safety를
얹지 않는다 — seam our_order가 이미 분위수 버퍼(q0.85)를 총량 레벨에 포함하므로
이중계상이 된다. 기존 explain_order(v6 apply_policy)는 다른 발주 철학, 무변경.
"""

from __future__ import annotations

import pandas as pd

from ..decision.policy import _round_up_to_unit
from ..forecast.forward import forecast_forward

BATCH_ROUND_UNIT = 3  # 라벨된 가정: 아띠제 배수생산(3/6/9). 품목별 배수는 후속.


def _forward_at_date(store_id, daily, date, horizon_days, use_forecast):
    """forecast_forward를 돌려 요청 date의 category_totals 1행 + item_quantities 슬라이스.

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
    return ct.iloc[0], iq


def explain_category_total(store_id, *, daily, date, horizon_days=7, use_forecast=False):
    """카테고리 생산총량 분해: base 예측 → event_prior 보정 → 분위수 버퍼 → prior_prod.

    event_prior = prior_median − base_median (실제 blend 차이, 특수일 레벨-앵커).
    """
    ct, _ = _forward_at_date(store_id, daily, date, horizon_days, use_forecast)
    base_median = float(ct["base_median"])
    prior_median = float(ct["prior_median"])
    prior_prod = float(ct["prior_prod"])
    event_prior = prior_median - base_median
    buffer = prior_prod - prior_median
    rows = [
        ("base_median", base_median, "Stage1 카테고리 수요 예측 (event_prior 이전)"),
        ("event_prior", event_prior, "특수일 레벨-앵커 블렌드 보정 (prior_median − base_median)"),
        ("prior_median", prior_median, "보정된 카테고리 수요 (q0.5)"),
        ("quantile_buffer", buffer, "생산 분위수 버퍼 (q0.85 − q0.5)"),
        ("prior_prod", prior_prod, "카테고리 생산총량"),
    ]
    return pd.DataFrame(
        [{"store_id": store_id, "date": date, "step": s, "value": v, "detail": d} for s, v, d in rows]
    )


def explain_item_order(store_id, item_id, *, daily, date, horizon_days=7,
                       round_unit=BATCH_ROUND_UNIT, use_forecast=False):
    """품목 생산량 분해(통합 단일 체인): 카테고리 총량 × 품목 비중 → 배수 라운딩.

    category_total(prior_prod) × proportion = item_order(=seam our_order) →
    ceil_round_unit = final. apply_policy 미사용(이중계상 방지).
    """
    ct, iq = _forward_at_date(store_id, daily, date, horizon_days, use_forecast)
    row = iq[iq["item_id"] == str(item_id)]
    if row.empty:
        raise ValueError(f"item {item_id} not in forward forecast for {store_id} at {date}")
    prior_prod = float(ct["prior_prod"])
    item_order = float(row["our_order"].iloc[0])
    proportion = item_order / prior_prod if prior_prod > 0 else 0.0
    final = _round_up_to_unit(item_order, round_unit)
    rows = [
        ("category_total", prior_prod, "카테고리 생산총량 (prior_prod)"),
        ("proportion", proportion, "품목 비중 (base_sold×adj_trend×adj_stockout×adj_closing×adj_new / Σ)"),
        ("item_order", item_order, "품목 생산량 (= 총량 × 비중)"),
        ("final", final, f"배수 라운딩 (ceil to {round_unit})"),
    ]
    return pd.DataFrame(
        [{"store_id": store_id, "item_id": str(item_id), "date": date,
          "step": s, "value": v, "detail": d} for s, v, d in rows]
    )
