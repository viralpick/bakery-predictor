"""예약(대량) 주문 검출 — 실수요 아닌 사전예약 라인 판별 (single source of truth).

배경: 특정일에 한 품목이 비정상적으로 많이 팔리는 경우(한 영수증에 수십~수백 개)는
방문 수요가 아니라 **사전 예약 주문**이다. 예약은 구매자가 미리 알려주므로 시계열로
예측 불가·예측 불필요(발주에 이미 반영). 실수요 학습을 오염시키므로 데이터에서 제거한다.

POS에 명시적 예약 플래그가 없어(광교 '대량구매할인' 코드도 사실상 미사용) qty 분포와
집중도 기반 휴리스틱으로 판별한다. 임계값은 광교 5년 분포로 재산정한 값이며 재검토 대상.

규칙 (line = (영수증,품목) 단위, T1 OR T2):
  Tier-1 단일품목 집중 — 확실한 단일품목 스파이크
      (영수증,품목) qty ≥ SINGLE_FLOOR AND qty ≥ SINGLE_K × median_daily(품목)
      AND active_days(품목) ≥ ACTIVE_MIN
  Tier-2 다품목 event — 여러 품목을 몰아 주문한 event(케이터링 등)
      영수증 total ≥ EVENT_TOTAL AND 영수증 maxit ≥ EVENT_MAXIT
      → 그 영수증에서 qty ≥ EVENT_MAXIT 인 라인만 flag (line-level: 소량 애드온 보존 → footfall 유지)

median_daily / active_days 는 판매>0 인 일자 기준(robust — 평균은 bulk가 분자에 껴 순환 편향).
"""
from __future__ import annotations

import pandas as pd

# 광교 bread/pastry/sandwich 5년 분포 기반 재산정 (2026-07). 재검토 대상.
SINGLE_FLOOR = 10   # 단일품목 절대 floor (p99.9≈6, 확실 예약 zone 시작)
SINGLE_K = 3        # 단일품목: median_daily의 배수 (인기품목 정상수요 오탐 방지)
ACTIVE_MIN = 14     # median_daily 신뢰 위한 최소 active days
EVENT_TOTAL = 30    # 다품목 event: 영수증 총량 (이 지점 92% 집중형)
EVENT_MAXIT = 5     # 다품목 event: 최소 단일품목 집중 + line-level 제거 floor


def flag_bulk_lines(
    lines: pd.DataFrame,
    *,
    single_floor: int = SINGLE_FLOOR,
    single_k: int = SINGLE_K,
    active_min: int = ACTIVE_MIN,
    event_total: int = EVENT_TOTAL,
    event_maxit: int = EVENT_MAXIT,
) -> pd.Series:
    """per-line 프레임 → bulk(제거대상) 여부 boolean Series (입력 index 정렬).

    lines columns: receipt_id, item_id, date, qty. 같은 (영수증,품목)이 여러 라인으로
    쪼개져 있어도 합산해 판정하고, 판정 결과를 그 (영수증,품목)의 모든 라인에 브로드캐스트한다.
    """
    if lines.empty:
        return pd.Series([], dtype=bool, index=lines.index)

    df = lines[["receipt_id", "item_id", "date", "qty"]].copy()

    # (영수증,품목) 합산 qty
    ri = df.groupby(["receipt_id", "item_id"], observed=True)["qty"].sum().rename("item_qty")

    # 품목별 robust 일평균(median) + active days
    daily_item = df.groupby(["item_id", "date"], observed=True)["qty"].sum()
    per_item = daily_item.groupby(level="item_id").agg(median_daily="median", active_days="count")

    # 영수증 단위 total + maxit
    ri_df = ri.reset_index()
    receipt = ri_df.groupby("receipt_id", observed=True)["item_qty"].agg(
        total="sum", maxit="max"
    )

    ri_df = ri_df.merge(per_item, on="item_id", how="left").merge(
        receipt, on="receipt_id", how="left"
    )

    is_tier1 = (
        (ri_df["item_qty"] >= single_floor)
        & (ri_df["item_qty"] >= single_k * ri_df["median_daily"])
        & (ri_df["active_days"] >= active_min)
    )
    is_event_receipt = (ri_df["total"] >= event_total) & (ri_df["maxit"] >= event_maxit)
    is_tier2 = is_event_receipt & (ri_df["item_qty"] >= event_maxit)

    ri_df["is_bulk"] = is_tier1 | is_tier2

    # (영수증,품목) 판정을 원본 라인에 브로드캐스트
    bulk_map = ri_df.set_index(["receipt_id", "item_id"])["is_bulk"]
    keys = pd.MultiIndex.from_frame(df[["receipt_id", "item_id"]])
    return pd.Series(bulk_map.reindex(keys).to_numpy(), index=lines.index, dtype=bool)
