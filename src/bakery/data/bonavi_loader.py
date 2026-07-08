"""Bonavi (보나비) excel → DAILY_COLUMNS adapter.

The vendor delivers a single multi-sheet xlsx:

  판매정보  영수증 line-item (458k rows over ~5 years)
  품목정보  품목 master (POS메뉴명 → keyword-mapped category_id)
  점포정보  매장 master (we only keep 광교 = 1000000047)
  품절정보  매장×품목×date 품절시각 (시분 HHMM)
  할인코드  (ignored for v1)

The xlsx is delivered with a dummy first row (CD_PARTNER / CD_ITEM / etc as
column placeholders) we drop, and every-sheet 판매구분 = 0 (정상) and
셋트상품구분 = SS (단품) only — we exclude returns and bundles for the
first PoC pass.

Output: `data/internal/bonavi_daily.parquet` conforming to
`schema.DAILY_COLUMNS` so `loader._load_real_dataset` can swap it in.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.potential_demand import StoreHours, attach_potential_demand
from .schema import DAILY_COLUMNS, validate_daily

XLSX_DEFAULT = Path("data/internal/보나비 데이터_20260520.xlsx")
OUT_DEFAULT = Path("data/internal/bonavi_daily.parquet")
DEFAULT_STORE_CODE = "1000000047"  # 아티제 아브뉴프랑광교점

# Keyword → category (order matters; cake/sandwich first, then pastry, then bread).
# 한국 제과제빵 표준 분류 기반:
#   - bread   : 식사빵 (식빵/바게트/치아바타/베이글/포카치아/잡곡/통밀/호밀)
#   - pastry  : 단과자빵 + 페이스트리 (단팥/모카/카스타드/크림/소보로/크루아상/데니쉬/스콘/머핀/롤)
#   - cake    : 케이크/타르트/몽블랑/바브카/파이
#   - sandwich: 완제품 (크로크무슈/토스트/파니니/버거/핫도그) — "식빵"은 가드로 별도
#   - sweets  : 작은 디저트 (마카롱/쿠키/마들렌 — 광교엔 없음, 미래 매장 대비)
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("cake", [
        "케이크", "케익", "케잌", "타르트", "티라미수", "갸또",
        "조각케", "몽블랑", "바브카", "쇼트케이크", "쉬폰", "파이",
    ]),
    ("sandwich", [
        "크로크무슈", "토스트", "파니니", "버거", "핫도그", "클럽샌드",
    ]),
    ("pastry", [
        # 페이스트리
        "크루아상", "크로와상", "데니쉬", "페이스트리", "페스트", "페스츄",
        "팡도르", "퀸아망", "사바랭", "팽 페르뒤", "팽페르뒤", "페르뒤",
        "파리브", "스트루델", "스투루델", "파네토네", "브리오슈", "크라운",
        # 작은 빵류
        "모닝롤", "버터롤", "치즈롤", "크림롤", "롤(",
        "머핀", "스콘", "꽈배기", "도넛", "도너츠",
        # 한국 단과자빵
        "팥", "앙금", "크림빵", "크림번", "초코번",
        "카스타드", "커스터드", "슈크림", "소보로",
        "모카번", "모카빵", "찰빵", "찹쌀",
        "고구마", "호박빵", "생크림빵",
        # 디저트 페이스트리
        "뺑오", "뺑 오", "쇼콜라", "초코",
        "크럼블", "턴오버", "발로나", "크런치",
        # 명시 패턴
        "슈레드치즈소시지", "슈레드 치즈 소시지",
        "포테이토 소시지 데니쉬",
    ]),
    ("bread", [
        "식빵", "바게트", "치아바타", "베이글", "포카치아",
        "호밀", "통밀", "잡곡", "곡물브레드", "곡물 브레드",
        "크림치즈브레드", "크림치즈 브레드",
        "베리 브레드", "크림브레드", "크림 브레드",
        "탕종", "맘모스",
        "마늘", "어니언", "부추", "토마토",
        "치즈브", "치즈 브",
        "브레드", "빵",  # fallback (pastry 키워드 미매칭 시)
    ]),
    ("sweets", [
        "마카롱", "쿠키", "마들렌", "휘낭시에", "다쿠아즈",
        "초콜릿", "초콜렛", "캔디", "젤리",
        "푸딩", "크림브륄레", "크림 브륄레",
    ]),
    ("beverage", [
        "커피", "라떼", "아메리카노", "에스프레소", "카푸치노", "콜드브루",
        "쥬스", "주스", "에이드", "스무디", "쉐이크",
        "드링크", "음료", "녹차", "우유", "두유", "요거트", "요구르트",
    ]),
]


def map_category(name: str) -> str:
    """POS menu name → category bucket (whitespace-normalized matching).

    가드: "식빵" 키워드가 들어가면 (C샌드위치식빵 같은 케이스) 무조건 bread.
    """
    norm = re.sub(r"\s+", "", str(name))
    if "식빵" in norm:
        return "bread"
    for cat, keywords in _CATEGORY_RULES:
        for kw in keywords:
            if re.sub(r"\s+", "", kw) in norm:
                return cat
    return "etc"


def load_items(xlsx: Path) -> pd.DataFrame:
    """품목정보 → (item_id, item_name, category_id, discard_daily) DataFrame."""
    raw = pd.read_excel(xlsx, "품목정보")
    raw = raw[raw["상품구분"] == "SS"].copy()
    raw["category_id"] = raw["POS메뉴명"].apply(map_category)
    return pd.DataFrame(
        {
            "item_id": raw["품목코드"].astype("string"),
            "item_name": raw["POS메뉴명"].astype("string"),
            "category_id": raw["category_id"].astype("string"),
            "discard_daily": raw["당일폐기여부"].astype("string").eq("Y"),
        }
    ).reset_index(drop=True)


def load_sales(xlsx: Path, store_code: str = DEFAULT_STORE_CODE) -> pd.DataFrame:
    """판매정보 → cleaned row-per-line-item frame.

    Drops the dummy header row (점포코드 == 'CD_PARTNER'), filters to the
    requested store, keeps only 정상 sales (판매구분 = 0) and 단품 lines
    (셋트상품구분 = SS). Includes hour (parsed from sale_time YYYYMMDDHHMMSS).
    """
    raw = pd.read_excel(xlsx, "판매정보")
    set_col = next(c for c in raw.columns if c.startswith("셋트상품구분"))
    sale_col = next(c for c in raw.columns if c.startswith("판매구분"))
    time_col = next(c for c in raw.columns if c.startswith("판매시간"))
    raw = raw[raw["점포코드"].astype(str) == store_code]
    raw = raw[raw[set_col] == "SS"]
    raw = raw[raw[sale_col].astype(str) == "0"]
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["판매일자"].astype(str), format="%Y%m%d").dt.normalize()
    raw[time_col] = raw[time_col].astype(str)
    raw["hour"] = raw[time_col].str.slice(8, 10)
    raw["hour"] = pd.to_numeric(raw["hour"], errors="coerce")
    return raw[["점포코드", "date", "품목코드", "판매수량", "hour"]].rename(
        columns={"점포코드": "store_id", "품목코드": "item_id", "판매수량": "qty"}
    )


def load_receipts_with_time(xlsx: Path, store_code: str = DEFAULT_STORE_CODE) -> pd.DataFrame:
    """판매정보 → receipts frame including hh:mm timestamp.

    Output columns:
        receipt_id (str)  — 영수증번호
        date (datetime64) — 판매일자 (normalized to 00:00)
        item_id (str)     — 품목코드
        hour (int 0-23)   — 판매시간 hour
        minute (int 0-59) — 판매시간 minute
        timestamp (datetime64) — full datetime (date + hour + minute)

    Same filters as load_sales: store + 정상매출(판매구분=0) + 단품(SS).
    Used by DiD substitution analysis where intra-day timing matters.
    """
    raw = pd.read_excel(xlsx, "판매정보")
    set_col = next(c for c in raw.columns if c.startswith("셋트상품구분"))
    sale_col = next(c for c in raw.columns if c.startswith("판매구분"))
    time_col = next(c for c in raw.columns if c.startswith("판매시간"))
    raw = raw[raw["점포코드"].astype(str) == store_code]
    raw = raw[raw[set_col] == "SS"]
    raw = raw[raw[sale_col].astype(str) == "0"]
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["판매일자"].astype(str), format="%Y%m%d").dt.normalize()
    tstr = raw[time_col].astype(str)
    raw["hour"] = pd.to_numeric(tstr.str.slice(8, 10), errors="coerce").astype("Int8")
    raw["minute"] = pd.to_numeric(tstr.str.slice(10, 12), errors="coerce").astype("Int8")
    raw = raw.dropna(subset=["hour", "minute"]).copy()
    raw["hour"] = raw["hour"].astype(int).clip(0, 23)
    raw["minute"] = raw["minute"].astype(int).clip(0, 59)
    raw["timestamp"] = (
        raw["date"]
        + pd.to_timedelta(raw["hour"], unit="h")
        + pd.to_timedelta(raw["minute"], unit="m")
    )
    # 영수증번호 is reset daily — make it globally unique by prefixing the date.
    raw["receipt_id"] = (
        raw["date"].dt.strftime("%Y%m%d") + "_" + raw["영수증번호"].astype(str)
    )
    return raw[["receipt_id", "date", "품목코드", "hour", "minute", "timestamp"]].rename(
        columns={"품목코드": "item_id"}
    ).assign(
        receipt_id=lambda d: d["receipt_id"].astype("string"),
        item_id=lambda d: d["item_id"].astype("string"),
    ).reset_index(drop=True)


def measure_hour_profile(sales: pd.DataFrame) -> dict[str, np.ndarray]:
    """Measure per-store hour-of-day sales distribution from receipt timestamps.

    Returns: {store_id: length-24 array summing to 1.0 over hours where sales exist}.
    Stores with no parseable hour data are skipped — caller falls back to the
    hard-coded 4-peak bakery curve in that case.
    """
    out: dict[str, np.ndarray] = {}
    valid = sales.dropna(subset=["hour"]).copy()
    valid["hour"] = valid["hour"].astype(int).clip(0, 23)
    valid["qty"] = pd.to_numeric(valid["qty"], errors="coerce").fillna(0)
    for sid, group in valid.groupby("store_id"):
        counts = group.groupby("hour")["qty"].sum()
        profile = np.zeros(24)
        for h, q in counts.items():
            profile[int(h)] = float(q)
        if profile.sum() > 0:
            profile = profile / profile.sum()
            out[str(sid)] = profile
    return out


def load_stockouts(xlsx: Path, store_code: str = DEFAULT_STORE_CODE) -> pd.DataFrame:
    """품절정보 → (store_id, date, item_id, stockout_time)."""
    raw = pd.read_excel(xlsx, "품절정보")
    raw = raw[raw["점포코드"].astype(str) == store_code].copy()
    raw["date"] = pd.to_datetime(raw["판매일자"].astype(str), format="%Y%m%d").dt.normalize()
    so_col = next(c for c in raw.columns if c.startswith("품절시간"))
    so = pd.to_numeric(raw[so_col], errors="coerce")
    # HHMM (e.g. 1730 = 17:30) — encode as full timestamp on the day
    hours = (so // 100).clip(0, 23).fillna(0).astype(int)
    minutes = (so % 100).clip(0, 59).fillna(0).astype(int)
    raw["stockout_time"] = raw["date"] + pd.to_timedelta(hours, "h") + pd.to_timedelta(minutes, "m")
    return pd.DataFrame(
        {
            "store_id": raw["점포코드"].astype("string"),
            "item_id": raw["품목코드"].astype("string"),
            "date": raw["date"],
            "stockout_time": raw["stockout_time"],
        }
    ).reset_index(drop=True)


def assign_stockout_fields(df: pd.DataFrame) -> pd.DataFrame:
    """물리 leftover 기반 진짜 최종소진. is_stockout=(made>0 & waste<=0),
    stockout_time=마지막 실판매(is_stockout일 때). 재고정보 결측(NaN)→False (NaN 비교가
    False라 자동). 첫 순간품절 이벤트를 쓰던 버그를 대체한다."""
    made = pd.to_numeric(df["production_qty"], errors="coerce")
    waste = pd.to_numeric(df["waste_qty"], errors="coerce")
    out = df.copy()
    out["is_stockout"] = ((made > 0) & (waste <= 0)).fillna(False).astype(bool)
    out["stockout_time"] = df["last_sale_ts"].where(out["is_stockout"])
    return out


def aggregate_daily(
    sales: pd.DataFrame,
    items: pd.DataFrame,
    inventory: pd.DataFrame,
    last_sale: pd.DataFrame,
    *,
    measured_profiles: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Combine sales + items + inventory + last_sale → DAILY_COLUMNS frame.

    sold_units: sum of qty per (store, item, date) — 정상 단품만.
    is_stockout / stockout_time: 물리 leftover 기반 진짜 최종소진.
    potential_demand: PoC 1차에서는 sold_units 그대로 (= no censoring correction)
                      stockout 보정은 features/potential_demand.py가 일관 처리하도록 추후 호출.
    capacity: PoC상 모름 → 매장×품목 sold_units max (관측 capacity proxy).
    open_hours: 매장 영업시간 unknown → 13 (대략) 고정. store_mapping에서 추후.
    """
    sales = sales.copy()
    sales["qty"] = pd.to_numeric(sales["qty"], errors="coerce").fillna(0).astype("int64")
    daily = (
        sales.groupby(["store_id", "item_id", "date"], as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "sold_units"})
    )
    daily = daily.merge(items[["item_id", "category_id"]], on="item_id", how="left")
    daily["category_id"] = daily["category_id"].fillna("etc")

    # is_stockout/stockout_time: 물리 leftover(폐기) 기반 진짜 최종소진 (첫 순간품절 이벤트 버그 대체)
    inv = inventory.copy()
    inv["item_id"] = inv["item_id"].astype(str)
    inv["date"] = pd.to_datetime(inv["date"]).dt.normalize()
    ls = last_sale.copy()
    ls["item_id"] = ls["item_id"].astype(str)
    ls["date"] = pd.to_datetime(ls["date"]).dt.normalize()
    daily["item_id"] = daily["item_id"].astype(str)
    daily = daily.merge(inv[["date", "item_id", "production_qty", "waste_qty"]],
                        on=["date", "item_id"], how="left")
    daily = daily.merge(ls[["date", "item_id", "last_sale_ts"]],
                        on=["date", "item_id"], how="left")
    daily = assign_stockout_fields(daily)
    daily = daily.drop(columns=["production_qty", "waste_qty", "last_sale_ts"])

    # Capacity proxy — running max of sold_units per (store, item)
    daily = daily.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    daily["capacity"] = (
        daily.groupby(["store_id", "item_id"])["sold_units"].cummax().astype("int32")
    )
    daily["open_hours"] = np.int16(13)

    # Censoring correction — 영업시간 7~22시 + 매장 실측 hour profile 적용.
    daily["potential_demand"] = daily["sold_units"].astype("float32")  # placeholder before attach
    store_hours_for_daily = [
        StoreHours(store_id=sid, open_hour=7, close_hour=22)
        for sid in daily["store_id"].unique()
    ]
    daily = attach_potential_demand(
        daily, store_hours_for_daily, measured_profiles=measured_profiles
    )

    daily = _coerce_daily_dtypes(daily)
    validate_daily(daily)
    return daily


def _coerce_daily_dtypes(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["store_id"] = daily["store_id"].astype("string")
    daily["item_id"] = daily["item_id"].astype("string")
    daily["category_id"] = daily["category_id"].astype("string")
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["sold_units"] = daily["sold_units"].astype("int32")
    daily["is_stockout"] = daily["is_stockout"].astype(bool)
    daily["stockout_time"] = pd.to_datetime(daily["stockout_time"])
    daily["open_hours"] = daily["open_hours"].astype("int16")
    daily["capacity"] = daily["capacity"].astype("int32")
    daily["potential_demand"] = daily["potential_demand"].astype("float32")
    daily = daily[list(DAILY_COLUMNS.keys())]
    return daily.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


def build(
    *,
    xlsx_path: Path | str = XLSX_DEFAULT,
    store_code: str = DEFAULT_STORE_CODE,
    out_path: Path | str = OUT_DEFAULT,
    rename_store_id: str | None = None,
    apply_substitution: bool = True,
    receipts_path: Path | str | None = None,
) -> Path:
    """End-to-end pipeline: xlsx → daily parquet conforming to DAILY_COLUMNS.

    `rename_store_id` lets us swap the numeric vendor code (e.g. '1000000047')
    for our internal alias (e.g. 'store_gw01') used in store_mapping.

    `apply_substitution=True` (default): builds an interim daily frame, runs
    `compute_substitution_matrix` against the receipts parquet, and re-calls
    `attach_potential_demand` with the per-item outflow ratio so v2/v3 models
    learn from substitution-corrected `potential_demand`. Falls back silently
    if the receipts parquet isn't there yet.
    """
    xlsx_path = Path(xlsx_path)
    out_path = Path(out_path)
    items = load_items(xlsx_path)
    sales = load_sales(xlsx_path, store_code=store_code)

    # Receipts with hh:mm — written once and reused by DiD substitution.
    receipts_out = Path(receipts_path or "data/internal/bonavi_receipts.parquet")
    receipts_df = load_receipts_with_time(xlsx_path, store_code=store_code)
    if rename_store_id is not None:
        receipts_df = receipts_df.copy()  # store_id stays implicit; receipts only carries receipt_id/item_id/timestamp
    receipts_out.parent.mkdir(parents=True, exist_ok=True)
    receipts_df.to_parquet(receipts_out, index=False)

    # Measure store's hour-of-day sales profile (vs hard-coded 4-peak default)
    measured_profiles = measure_hour_profile(sales)

    from ..ingest.inventory import load_inventory
    inv_store = rename_store_id or "store_gw01"
    inventory = load_inventory(str(xlsx_path), inv_store)
    # 마지막 실판매 시각 (item-day별 max) — receipts_df에서
    last_sale = (
        receipts_df.groupby(["date", "item_id"], as_index=False)["timestamp"].max()
        .rename(columns={"timestamp": "last_sale_ts"})
    )

    daily = aggregate_daily(sales, items, inventory, last_sale, measured_profiles=measured_profiles)
    if rename_store_id is not None:
        daily["store_id"] = rename_store_id
        # Re-key measured_profiles to the renamed store_id
        measured_profiles = {rename_store_id: next(iter(measured_profiles.values()))} if measured_profiles else {}

    # Second pass — apply substitution outflow to potential_demand
    receipts_path = Path(receipts_path or "data/internal/bonavi_receipts.parquet")
    if apply_substitution and receipts_path.exists():
        try:
            from ..analysis.substitution import compute_substitution_matrix
            receipts_df = pd.read_parquet(receipts_path)
            # Pass measured_profiles so substitution derives per-store cutoffs
            # from each store's actual sales distribution.
            matrix = compute_substitution_matrix(
                daily, receipts_df, hour_profiles=measured_profiles or None,
            )
            store_hours = [
                StoreHours(store_id=sid, open_hour=7, close_hour=22)
                for sid in daily["store_id"].unique()
            ]
            daily = attach_potential_demand(
                daily, store_hours,
                outflow_ratio=matrix.outflow_ratio.clip(lower=0.0, upper=1.0),
                measured_profiles=measured_profiles,
            )
            daily = _coerce_daily_dtypes(daily)
            validate_daily(daily)
        except Exception:  # noqa: BLE001 — receipts may be missing on first run
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_path, index=False)
    return out_path
