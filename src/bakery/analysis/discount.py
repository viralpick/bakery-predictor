"""Discount/promotion analysis for bonavi 광교.

Loads discount line-items from raw xlsx (판매정보 시트), classifies 30 codes
into business categories, and exposes helpers for closing-discount analysis
+ waste estimation (replaceable when 폐기 실측 data arrives).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

DEFAULT_XLSX = Path("data/internal/보나비 데이터_20260520.xlsx")


# ---------------------------------------------------------------------------
# Code classification (광교 30개)
# ---------------------------------------------------------------------------
# label categories:
#   closing   : 마감 회수 할인 (폐기 risk proxy)
#   payment   : 결제수단/멤버십 카드 할인 (PAYCO, T멤버십, 카드 등)
#   staff     : 직원/내부 (실손, 품질테스트, 사업부 등)
#   b2b       : 제휴 거래처 (대한제분, 우리와, 입주사 등)
#   marketing : 마케팅/이벤트 (T DAY, LSM 스탬프, 마케팅할인 등)
#   menu      : 일반 메뉴 할인

GWANGYO_CODE_LABELS: dict[str, str] = {
    "0077": "closing",      # 마감할인(30%)  — 49,975건 (33%)
    "0069": "closing",      # 마감할인(20%)  — 9,070건
    "0121": "payment",      # PAYCO할인(30%) — 44,293건
    "317":  "payment",      # PAYCO할인(20%) — 31,336건
    "334":  "payment",      # T멤버십할인(상시)
    "0118": "payment",      # 벤츠VIP할인(10%)
    "0053": "payment",      # 삼성RAUME카드(10%)
    "0078": "b2b",          # 입주사할인(10%) — 광교 입주사
    "0004": "b2b",          # (메세나폴리스)입주사할인(10%)
    "0082": "b2b",          # 그룹사할인(30%)
    "0080": "b2b",          # 사업부직원할인(20%)
    "330":  "b2b",          # 대한제분30%
    "331":  "b2b",          # 대한사료30%
    "332":  "b2b",          # 대한싸이로30%
    "333":  "b2b",          # 우리와 30%
    "0081": "staff",        # 보나비직원할인(50%)
    "0011": "staff",        # 파스텔직원할인(20%)
    "0083": "staff",        # 직원음료식사제공
    "0058": "staff",        # 직원실수
    "0084": "staff",        # 품질테스트
    "0049": "staff",        # 고객서비스
    "0059": "staff",        # 식음료테스트
    "339":  "marketing",    # T DAY할인 (40%)
    "0048": "marketing",    # LSM스탬프이벤트(50%)
    "0186": "marketing",    # (LSM)홀케이크 추가구매할인
    "0129": "marketing",    # 마케팅할인(100%)
    "0018": "menu",         # 메뉴할인(20%)
    "0013": "menu",         # 메뉴할인(10%)
    "0066": "menu",         # 대량구매할인(10%)
    "147":  "menu",         # (C)메뉴할인(30%)
}

LABEL_DESCRIPTIONS = {
    "closing":   "마감 회수 할인 — 폐기 risk proxy",
    "payment":   "결제수단/카드/멤버십 할인 — 마케팅 비용",
    "staff":     "직원/내부 (실손·품질·서비스)",
    "b2b":       "제휴 거래처 / 입주사",
    "marketing": "마케팅 이벤트 / 스탬프 / T DAY",
    "menu":      "일반 메뉴 할인",
}


def classify_code(code: str) -> str:
    """Return business label for a discount code; 'other' if unknown."""
    return GWANGYO_CODE_LABELS.get(str(code).strip(), "other")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

@dataclass
class DiscountSales:
    """Per-line-item sales with discount info."""
    rows: pd.DataFrame  # columns: receipt_id, date, hour, minute, item_id, qty, unit_price, paid, discount_amt, discount_code, label, is_set

    def discounted(self) -> pd.DataFrame:
        return self.rows[self.rows["discount_amt"] > 0]

    def closing_discount(self) -> pd.DataFrame:
        return self.rows[self.rows["label"] == "closing"]


def load_sales_with_discount(xlsx_path: Path | str = DEFAULT_XLSX) -> DiscountSales:
    """Load 판매정보 sheet with discount columns + classified label."""
    sales = pd.read_excel(xlsx_path, sheet_name="판매정보")
    set_col  = next(c for c in sales.columns if "셋트상품구분" in c)
    sale_col = next(c for c in sales.columns if "판매구분"   in c)
    time_col = next(c for c in sales.columns if "판매시간"   in c)

    # Drop returns (반품)
    sales = sales[sales[sale_col].astype(str).str[0] == "0"].copy()

    sales["판매시간_str"] = sales[time_col].astype(str).str.zfill(14)
    sales["hour"]   = sales["판매시간_str"].str[8:10].astype(int)
    sales["minute"] = sales["판매시간_str"].str[10:12].astype(int)
    sales["date"]   = pd.to_datetime(sales["판매일자"].astype(str), format="%Y%m%d")
    sales["discount_code"] = sales["할인코드"].astype(str).str.strip()
    sales["label"] = sales["discount_code"].map(classify_code).where(
        sales["할인금액"].fillna(0) > 0, "none"
    )
    sales["is_set"] = sales[set_col].astype(str).str.startswith("ST")

    rows = pd.DataFrame({
        "receipt_id":    sales["영수증번호"].astype(str),
        "date":          sales["date"],
        "hour":          sales["hour"],
        "minute":        sales["minute"],
        "item_id":       sales["품목코드"].astype(str),
        "qty":           sales["판매수량"].astype(int),
        "unit_price":    sales["단가"].astype(float),
        "paid":          sales["결제금액"].astype(float),
        "discount_amt":  sales["할인금액"].fillna(0).astype(float),
        "discount_code": sales["discount_code"],
        "label":         sales["label"],
        "is_set":        sales["is_set"],
    })
    return DiscountSales(rows=rows)


def load_closing_returns(xlsx_path: Path | str = DEFAULT_XLSX) -> pd.DataFrame:
    """마감할인 반품(판매구분=1 + closing label) → (item_id, date, ret_qty).

    `load_sales_with_discount`는 반품을 drop하므로 sold_closing이 gross-of-returns가
    된다(반품된 마감할인 매출이 남음). 이 함수가 마감할인 반품을 (item, date)별로 집계해
    sold_closing에서 차감(net-out)하게 한다 — sold_units(파퀫)이 이미 net이므로 타깃 두
    항의 기준을 통일. 대량취소 제외는 `_aggregate_returns`(단품 sold_units와 단일 출처).
    """
    from bakery.data.bonavi_loader import _aggregate_returns

    sales = pd.read_excel(xlsx_path, sheet_name="판매정보")
    set_col = next(c for c in sales.columns if "셋트상품구분" in c)
    sale_col = next(c for c in sales.columns if "판매구분" in c)
    ret = sales[
        (sales[sale_col].astype(str) == "1") & (sales[set_col].astype(str) == "SS")
    ].copy()
    if ret.empty:
        return pd.DataFrame({"item_id": [], "date": [], "ret_qty": []})
    code = ret["할인코드"].astype(str).str.strip()
    amt = pd.to_numeric(ret["할인금액"], errors="coerce").fillna(0)
    label = code.map(classify_code).where(amt > 0, "none")
    closing = ret[label == "closing"]
    if closing.empty:
        return pd.DataFrame({"item_id": [], "date": [], "ret_qty": []})
    lines = pd.DataFrame({
        "item_id": closing["품목코드"].astype(str),
        "date": pd.to_datetime(closing["판매일자"].astype(str), format="%Y%m%d"),
        "qty": pd.to_numeric(closing["판매수량"], errors="coerce").fillna(0),
    })
    return _aggregate_returns(lines)


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def discount_summary(ds: DiscountSales) -> pd.DataFrame:
    """Per-code summary: count, qty, avg discount, peak hour."""
    d = ds.discounted().copy()
    summ = d.groupby(["discount_code", "label"]).agg(
        rows         = ("qty", "size"),
        qty_total    = ("qty", "sum"),
        amt_total    = ("discount_amt", "sum"),
        avg_amt      = ("discount_amt", "mean"),
        peak_hour    = ("hour", lambda s: s.mode().iat[0] if len(s.mode()) else -1),
        share_at_pm8 = ("hour", lambda s: (s >= 20).mean()),
    ).reset_index()
    return summ.sort_values("rows", ascending=False)


def label_summary(ds: DiscountSales) -> pd.DataFrame:
    d = ds.discounted().copy()
    return d.groupby("label").agg(
        rows         = ("qty", "size"),
        qty_total    = ("qty", "sum"),
        amt_total    = ("discount_amt", "sum"),
        share_at_pm8 = ("hour", lambda s: (s >= 20).mean()),
    ).reset_index().sort_values("rows", ascending=False)


def closing_by_category_hour(ds: DiscountSales, item_to_category: pd.Series) -> pd.DataFrame:
    """Closing-discount qty/loss by category × hour."""
    c = ds.closing_discount().copy()
    c["category_id"] = c["item_id"].map(item_to_category)
    by = c.groupby(["category_id", "hour"]).agg(
        qty       = ("qty", "sum"),
        loss_won  = ("discount_amt", "sum"),
        rows      = ("qty", "size"),
    ).reset_index()
    return by


# ---------------------------------------------------------------------------
# Waste estimator (swappable)
# ---------------------------------------------------------------------------

class WasteSource(Protocol):
    """Implement when actual waste data arrives — drop-in replacement."""
    def waste_per_day_item(self) -> pd.DataFrame:
        """Return DataFrame[date, item_id, waste_qty, waste_cost_won]."""
        ...


@dataclass
class ClosingDiscountWasteProxy:
    """Until 폐기 실측 arrives, treat closing-discount qty as a lower-bound proxy:

      - Closing-discount qty = items that WOULD have been wasted but were
        marked-down and recovered some revenue.
      - Implied waste cost = (unit_price - paid_price) per discounted unit
        = the revenue we gave up to avoid throwing the bread away.

    Note: this estimates *recovered* waste cost, not *thrown-away* qty. Real
    waste = remaining unsold after closing-discount window. Need 입고량 data
    to compute that fully; see project_data_gaps.md.
    """
    ds: DiscountSales

    def waste_per_day_item(self) -> pd.DataFrame:
        c = self.ds.closing_discount().copy()
        out = c.groupby(["date", "item_id"]).agg(
            closing_discount_qty = ("qty", "sum"),
            revenue_loss_won     = ("discount_amt", "sum"),
        ).reset_index()
        return out
