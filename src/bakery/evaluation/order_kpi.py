"""발주 KPI 단일 기준 — A/B basis 병기 + 아띠제 대비 절감률.

이 모듈이 **정의의 단일 출처**다. 같은 지표가 `order_cost.py` 와
`scripts/unified_policy_kpi.py` 에서 다르게 계산돼 "같은 축에서 비교하면 오도되는"
상태를 끝내기 위해 만들었다. 설계 = `docs/superpowers/specs/2026-07-30-kpi-plane-backbone.md` §2.

## 두 basis를 반드시 병기한다 (헌장 §5)

- **A = 아띠제 실측**: 폐기 = 재고정보 `QT_OUT` 그대로, 매진 = `is_stockout`(made>0 & waste<=0).
  아띠제 현행의 **실제** 성적이며, 절감률의 분모다.
- **B = 모델 시뮬**: 폐기 = max(발주 − 실수요, 0), 매진 = 발주 < 실수요.
  모델끼리 비교할 때 쓴다.

**A와 B의 간극이 censoring 크기다.** 하나만 보고하면 그 간극이 사라진다 — 그래서 병기가
선택이 아니라 계약이다. 같은 정책(아띠제 실생산)을 A와 B로 각각 재면 그 차이가 잣대 효과다.

## 확정된 정의 (축마다 하나)

| 축 | 확정 | 왜 |
|---|---|---|
| 원가율 | **품목별**(1511=0.40 / 1513=0.60) | 접두어 판별자가 정답 xlsx 41품목 오분류 0 |
| 품절 손실 | **전체매진만**(k=0) | 품목 품절은 흡수 — [[order_cost]] 참조 |
| SKU 품절율 | **날별 비율의 평균** | 헌장 문구가 "각 날 품절 SKU 비율의 날별 평균" |
| 매진시각 | **전체 median + 날별 median의 평균 병기** | 서로 다른 질문에 답한다 |
| 음수 폐기 | **양쪽 병기**(`clip` on/off) | 미규명이라 한쪽을 숨기면 절감률이 자의적이 된다 |

⚠️ **음수 폐기**: 재고정보 `QT_OUT` 이 음수인 행이 광교 최근 364일에 3.70%(494/13,359),
합 −1,010개로 폐기 총량의 **5.54%** 다. 재고 정정인지 반품 상계인지 **미규명**이므로
`waste_clip_negative` 를 파라미터로 노출하고 두 값을 함께 낸다.
(참고: 판매 반품(`SALES_FG=1`)은 별개 소스이며 전처리 백본에서 이미 net-out 된다.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .order_cost import (
    DEFAULT_EARLY_STOCKOUT_HOUR,
    category_stockout_cost,
    cost_rate_for,
)

BASIS_ACTUAL = "A_actual"
BASIS_SIM = "B_sim"
# 헌장 §4 관점①(A): 입력오류를 감안해 그날 총폐기가 이 값 이하면 전체매진으로 본다.
# unified_policy_kpi.py의 WASTE_TOL과 같은 값이어야 한다(정의 단일화).
CATEGORY_SELLOUT_WASTE_TOL = 5.0


def _waste_units(waste: pd.Series, *, clip_negative: bool) -> pd.Series:
    """폐기량 정규화. clip_negative=True면 음수를 0으로."""
    out = pd.to_numeric(waste, errors="coerce").fillna(0.0)
    return out.clip(lower=0.0) if clip_negative else out


def _daily_mean_of_rates(frame: pd.DataFrame, flag_col: str, date_col: str) -> float:
    """★날별 비율의 평균 — 행 단위 평균과 다르다(품목 수가 날마다 달라서).

    헌장 관점②가 "각 날 품절 SKU 비율의 날별 평균"이므로 이 계산이 정의다.
    """
    if frame.empty:
        return float("nan")
    per_day = frame.groupby(date_col)[flag_col].mean()
    return float(per_day.mean())


def _soldout_hour_stats(frame: pd.DataFrame, *, date_col: str) -> dict:
    """매진시각 2관점 병기. 매진 행이 없으면 둘 다 nan.

    - `soldout_hour_median`: 전체 매진 행의 median (분포의 중심)
    - `soldout_hour_median_mean`: 날별 median의 평균 (날을 동등 가중)
    """
    so = frame[frame["is_stockout"].astype(bool)]
    if so.empty or so["soldout_hour"].isna().all():
        return {"soldout_hour_median": float("nan"), "soldout_hour_median_mean": float("nan")}
    daily = so.groupby(date_col)["soldout_hour"].median()
    return {
        "soldout_hour_median": float(so["soldout_hour"].median()),
        "soldout_hour_median_mean": float(daily.mean()),
    }


def basis_actual(
    rows: pd.DataFrame,
    *,
    waste_col: str = "waste_qty",
    price_col: str = "unit_price",
    date_col: str = "date",
    waste_clip_negative: bool = True,
) -> dict:
    """A basis — 아띠제 실측. 폐기는 `QT_OUT` 그대로, 매진은 `is_stockout`.

    rows: 품목-일 프레임. 필요 컬럼 = item_id / date / waste_qty / is_stockout / unit_price.
    """
    waste = _waste_units(rows[waste_col], clip_negative=waste_clip_negative)
    rate = cost_rate_for(rows["item_id"]).to_numpy()
    price = pd.to_numeric(rows[price_col], errors="coerce").fillna(0.0).to_numpy()
    work = rows.assign(_waste=waste.to_numpy(), _flag=rows["is_stockout"].astype(bool))
    day_waste = work.groupby(date_col)["_waste"].sum()
    return {
        "basis": BASIS_ACTUAL,
        "waste_units": float(waste.sum()),
        "waste_cost_krw": float((waste.to_numpy() * price * rate).sum()),
        # 관점① 전체매진(A): 그날 총폐기가 tolerance 이하 = 카테고리 소진
        "category_stockout_day_rate": float((day_waste <= CATEGORY_SELLOUT_WASTE_TOL).mean()),
        # 관점② SKU 품절: 날별 비율의 평균
        "sku_soldout_rate": _daily_mean_of_rates(work, "_flag", date_col),
    }


def basis_sim(
    costed: pd.DataFrame,
    *,
    order_col: str,
    demand_col: str,
    date_col: str = "date",
    early_hour: int = DEFAULT_EARLY_STOCKOUT_HOUR,
) -> dict:
    """B basis — 모델 시뮬(발주 vs 실수요). 품목별 원가율 + 전체매진 손실(k=0).

    costed: `order_cost()` 출력(+ 필요하면 `stockout_timing()` 병합).
      필요 컬럼 = item_id / date / order_qty / 실수요 / waste_units / waste_cost_krw.
      매진시각 지표는 `soldout_hour`/`is_stockout` 이 있을 때만 채운다.
    """
    category = category_stockout_cost(
        costed, order_col=order_col, demand_col=demand_col, date_col=date_col,
    )
    order = pd.to_numeric(costed[order_col], errors="coerce").fillna(0.0)
    demand = pd.to_numeric(costed[demand_col], errors="coerce").fillna(0.0)
    work = costed.assign(_flag=(order < demand).to_numpy())
    waste_cost = float(costed["waste_cost_krw"].sum())
    cat_lost = float(category["category_lost_margin_krw"].sum())
    out = {
        "basis": BASIS_SIM,
        "waste_units": float(costed["waste_units"].sum()),
        "waste_cost_krw": waste_cost,
        "category_short_units": float(category["category_short_units"].sum()),
        "category_lost_margin_krw": cat_lost,
        "category_stockout_days": int(category["is_category_stockout"].sum()),
        "category_stockout_day_rate": float(category["is_category_stockout"].mean()),
        # ★k=0 총비용 = 품목 폐기 + 전체매진 마진손실
        "total_cost_krw": waste_cost + cat_lost,
        "sku_soldout_rate": _daily_mean_of_rates(work, "_flag", date_col),
    }
    if "soldout_hour" in costed.columns and "is_stockout" in costed.columns:
        out.update(_soldout_hour_stats(costed, date_col=date_col))
        so = costed[costed["is_stockout"].astype(bool)]
        out["early_stockout_rate"] = (
            float((so["soldout_hour"] < early_hour).mean()) if len(so) else float("nan")
        )
    return out


def waste_reduction_pct(model_waste_krw: float, reference_waste_krw: float) -> float:
    """폐기비용 절감률(%) — 음수가 절감이다. 기준이 0이면 nan."""
    if reference_waste_krw <= 0:
        return float("nan")
    return (model_waste_krw / reference_waste_krw - 1.0) * 100.0


def compare_to_actual(
    model: dict,
    actual: dict,
    *,
    actual_sim: dict | None = None,
) -> dict:
    """★아띠제 대비 절감률 — 어느 basis 대비인지 라벨과 함께 낸다.

    `actual_sim`(아띠제 실생산을 **B로 시뮬**한 결과)을 주면 동일-basis 비교도 함께 낸다.
    ⚠️ 모델(B) vs 아띠제(A)는 **잣대가 다르다** — A는 실측 폐기, B는 발주−실수요다.
    그 차이가 censoring이므로, 공정 비교는 `vs_actual_sim_pct`(B vs B)쪽이다.
    두 값을 함께 봐야 "절감이 진짜인가 잣대 효과인가"를 판별할 수 있다.
    """
    out = {
        "waste_krw_model": model["waste_cost_krw"],
        "waste_krw_actual": actual["waste_cost_krw"],
        "vs_actual_pct": waste_reduction_pct(model["waste_cost_krw"], actual["waste_cost_krw"]),
    }
    if actual_sim is not None:
        out["waste_krw_actual_sim"] = actual_sim["waste_cost_krw"]
        out["vs_actual_sim_pct"] = waste_reduction_pct(
            model["waste_cost_krw"], actual_sim["waste_cost_krw"]
        )
        # A/B 간극 = censoring 크기(같은 정책을 두 잣대로 잰 차이)
        out["censoring_gap_pct"] = waste_reduction_pct(
            actual_sim["waste_cost_krw"], actual["waste_cost_krw"]
        )
    return out


def waste_negative_diagnostics(
    rows: pd.DataFrame, *, waste_col: str = "waste_qty"
) -> dict:
    """음수 폐기 진단 — clip 여부가 절감률 분모를 얼마나 바꾸는지 드러낸다.

    미규명 이슈이므로 숨기지 않고 수치로 보고한다(절대 규칙 #6).
    """
    raw = pd.to_numeric(rows[waste_col], errors="coerce").fillna(0.0)
    clipped = raw.clip(lower=0.0)
    total_raw, total_clip = float(raw.sum()), float(clipped.sum())
    return {
        "negative_rows": int((raw < 0).sum()),
        "n_rows": int(len(raw)),
        "negative_row_rate": float((raw < 0).mean()) if len(raw) else float("nan"),
        "negative_units": float(raw[raw < 0].sum()),
        "waste_units_raw": total_raw,
        "waste_units_clipped": total_clip,
        "clip_effect_pct": (
            (total_clip / total_raw - 1.0) * 100.0 if total_raw > 0 else float("nan")
        ),
    }


def kpi_table(records: list[dict]) -> pd.DataFrame:
    """basis 라벨을 유지한 채 KPI dict 목록을 표로. 누락 키는 NaN(가림 금지)."""
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    lead = [c for c in ("policy", "basis") if c in frame.columns]
    return frame[lead + [c for c in frame.columns if c not in lead]]


def sku_soldout_rate_rowwise(frame: pd.DataFrame, flag_col: str) -> float:
    """행 단위 평균 — **정의가 아니다**. 날별 평균과의 차이를 보여줄 때만 쓴다.

    옛 `order_cost.summarize_order_kpi` 가 이 계산을 썼다. 날마다 품목 수가 달라
    날별 평균과 값이 다르며, 헌장 문구는 날별 평균이다. 회귀 비교용으로만 남긴다.
    """
    if frame.empty:
        return float("nan")
    return float(np.asarray(frame[flag_col], dtype=bool).mean())
