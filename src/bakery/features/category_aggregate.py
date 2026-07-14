"""Category-aggregate features for v4 Stage 1 (Category-Total Demand Model).

옵션 C+ 적용 (2026-05-23):
- cyclic encoding (dom/dow/month sin/cos)
- is_public_holiday + is_weekend + is_before_holiday
- days_to_events (xmas/valentine/white_day/children_day) ±14 cap
- weather + 강수 binning + hr1MaxRn 영업시간 강수
- 경쟁점 일자별 active 점포 수 (Haversine)
- unit + revenue 두 target 병행
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bakery.analysis.discount import load_sales_with_discount
from bakery.analysis.seasonal import filter_seasonal
from bakery.data.calendar import LUNAR_EVENT_DATES


TARGET_CATEGORIES = ("bread", "pastry", "sandwich")   # cake 제외 (사전 예약 + 시즌 특수)
DEFAULT_ALPHA = 0.8   # 마감할인 실수요 비율 α (2026-07 상향: 저녁 상시할인 구조상 높은 α 방향 증거)
LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)
EWMA_HALFLIVES = (7, 28)

EVENT_CLIP = 14
EVENTS: dict[str, tuple[int, int]] = {
    "days_to_xmas":         (12, 25),
    "days_to_valentine":    (2, 14),
    "days_to_white_day":    (3, 14),
    "days_to_children_day": (5, 5),
    # pepero 제외 (검증 결과 효과 약함 + 통계 유의 X, 데이터 한계)
}

# 음력 명절 (양력 변동) — 검증: 추석 +11.4% (p=0.029), 설날 +11.0% (p=0.005).
# 단일 출처는 data/calendar.py. v0~v3(calendar_features)와 날짜가 어긋나지 않도록 공유한다.
LUNAR_EVENTS: dict[str, dict[int, str]] = LUNAR_EVENT_DATES

# 광교 좌표 (store_mapping.yaml)
GWANGYO_LAT = 37.2853
GWANGYO_LON = 127.0593
COMPETITOR_RADIUS_KM = 1.0


@dataclass
class CategoryDaily:
    df: pd.DataFrame
    alpha: float


# ---------------------------------------------------------------------------
# Build base
# ---------------------------------------------------------------------------

def _attach_unit_price(daily: pd.DataFrame) -> pd.DataFrame:
    """item_id → 단가 (품목정보 시트). NaN은 평균 4000으로 fallback."""
    xl_path = Path("data/internal/보나비 데이터_20260520.xlsx")
    if not xl_path.exists():
        daily["unit_price"] = 4000.0
        return daily
    items = pd.read_excel(xl_path, sheet_name="품목정보")
    items["item_id"] = items["품목코드"].astype(str)
    items["판매단가"] = pd.to_numeric(items["판매단가"], errors="coerce")
    price_map = items.set_index("item_id")["판매단가"].to_dict()
    daily["unit_price"] = daily["item_id"].astype(str).map(price_map).fillna(4000.0)
    return daily


def build_category_daily(
    daily_raw: pd.DataFrame | None = None,
    discount_rows: pd.DataFrame | None = None,
    alpha: float = DEFAULT_ALPHA,
    categories: tuple[str, ...] = TARGET_CATEGORIES,
    closing_returns: pd.DataFrame | None = None,
) -> CategoryDaily:
    """카테고리 합 daily: unit + revenue 두 metric."""
    if daily_raw is None:
        daily_raw = pd.read_parquet("data/internal/bonavi_daily.parquet")
        daily_raw["item_id"] = daily_raw["item_id"].astype(str)

    daily = filter_seasonal(daily_raw)
    daily = daily[daily["category_id"].isin(categories)].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = _attach_unit_price(daily)
    daily["revenue"] = daily["sold_units"] * daily["unit_price"]

    # closing
    if discount_rows is None:
        ds = load_sales_with_discount()
        discount_rows = ds.closing_discount().copy()
        discount_rows["item_id"] = discount_rows["item_id"].astype(str)
        if closing_returns is None:
            from bakery.analysis.discount import load_closing_returns
            closing_returns = load_closing_returns()

    cd = filter_seasonal(discount_rows)
    cd["date"] = pd.to_datetime(cd["date"])
    cat_map = daily.drop_duplicates("item_id").set_index("item_id")["category_id"]
    cd = cd.assign(category_id=cd["item_id"].map(cat_map))
    cd = cd[cd["category_id"].isin(categories)]

    # closing revenue: closing unit × 정가 (할인 전 가격으로 평가)
    cd = _attach_unit_price(cd.rename(columns={"qty": "sold_units"})).rename(columns={"sold_units": "qty"})
    cd["closing_revenue"] = cd["qty"] * cd["unit_price"]
    closing_by_date = cd.groupby("date").agg(
        sold_closing=("qty", "sum"),
        sold_closing_revenue=("closing_revenue", "sum"),
    )

    # 마감할인 반품 net-out (sold_units 파퀫과 기준 통일) — unit + revenue 모두 차감(clip≥0)
    if closing_returns is not None and not closing_returns.empty:
        cr = closing_returns.copy()
        cr["item_id"] = cr["item_id"].astype(str)
        cr["date"] = pd.to_datetime(cr["date"])
        cr = filter_seasonal(cr)
        cr = cr.assign(category_id=cr["item_id"].map(cat_map))
        cr = cr[cr["category_id"].isin(categories)]
        cr = _attach_unit_price(cr.rename(columns={"ret_qty": "sold_units"})).rename(
            columns={"sold_units": "ret_qty"}
        )
        cr["ret_revenue"] = cr["ret_qty"] * cr["unit_price"]
        cr_by_date = cr.groupby("date").agg(
            ret_unit=("ret_qty", "sum"), ret_revenue=("ret_revenue", "sum")
        )
        closing_by_date = closing_by_date.join(cr_by_date, how="left")
        closing_by_date["sold_closing"] = (
            closing_by_date["sold_closing"] - closing_by_date["ret_unit"].fillna(0)
        ).clip(lower=0)
        closing_by_date["sold_closing_revenue"] = (
            closing_by_date["sold_closing_revenue"] - closing_by_date["ret_revenue"].fillna(0)
        ).clip(lower=0)
        closing_by_date = closing_by_date.drop(columns=["ret_unit", "ret_revenue"])

    daily["stockout_hour"] = pd.to_datetime(daily["stockout_time"]).dt.hour

    agg = daily.groupby("date").agg(
        sold_total_unit=("sold_units", "sum"),
        sold_total_revenue=("revenue", "sum"),
        n_items_active=("item_id", "nunique"),
        n_stockout_items=("is_stockout", "sum"),
        n_early_stockout=("stockout_hour", lambda s: (s.dropna() <= 16).sum()),
    ).reset_index()
    agg = agg.merge(closing_by_date.reset_index(), on="date", how="left")
    agg["sold_closing"] = agg["sold_closing"].fillna(0).astype(int)
    agg["sold_closing_revenue"] = agg["sold_closing_revenue"].fillna(0)

    agg["sold_normal_unit"]    = agg["sold_total_unit"]    - agg["sold_closing"]
    agg["sold_normal_revenue"] = agg["sold_total_revenue"] - agg["sold_closing_revenue"]
    agg["adjusted_demand_unit"]    = agg["sold_normal_unit"]    + agg["sold_closing"]         * alpha
    agg["adjusted_demand_revenue"] = agg["sold_normal_revenue"] + agg["sold_closing_revenue"] * alpha

    return CategoryDaily(df=agg, alpha=alpha)


def build_item_adjusted_demand(
    daily: pd.DataFrame,
    discount_rows: pd.DataFrame | None = None,
    closing_returns: pd.DataFrame | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """item-day별 adjusted_demand = sold_units − closing_qty × (1 − α) 를 추가.

    adjusted = normal + closing×α = (sold − closing) + closing×α = sold − closing×(1−α).
    closing 매칭 없는 item-day는 closing=0 → adjusted == sold_units.
    당일 관측 label(leakage-safe). 입력은 변형하지 않는다.

    closing_returns(load_closing_returns 출력, [item_id,date,ret_qty])가 주어지면
    closing_qty에서 마감할인 반품을 차감(clip≥0)한다 — sold_units(파퀫)이 이미 반품 net
    이므로 두 항 기준 통일. discount_rows 자동로드 시 closing_returns도 자동로드.
    """
    if discount_rows is None:
        discount_rows = load_sales_with_discount().closing_discount()
        if closing_returns is None:
            from bakery.analysis.discount import load_closing_returns
            closing_returns = load_closing_returns()
    out = daily.copy()
    out["item_id"] = out["item_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"])
    cd = discount_rows.copy()
    cd["item_id"] = cd["item_id"].astype(str)
    cd["date"] = pd.to_datetime(cd["date"])
    closing_qty = (
        cd.groupby(["item_id", "date"])["qty"].sum().rename("closing_qty").reset_index()
    )
    out = out.merge(closing_qty, on=["item_id", "date"], how="left")
    out["closing_qty"] = out["closing_qty"].fillna(0.0)

    # 마감할인 반품 net-out (sold_units와 기준 통일)
    if closing_returns is not None and not closing_returns.empty:
        cr = closing_returns.copy()
        cr["item_id"] = cr["item_id"].astype(str)
        cr["date"] = pd.to_datetime(cr["date"])
        out = out.merge(cr[["item_id", "date", "ret_qty"]], on=["item_id", "date"], how="left")
        out["closing_qty"] = (out["closing_qty"] - out["ret_qty"].fillna(0.0)).clip(lower=0)
        out = out.drop(columns=["ret_qty"])

    out["adjusted_demand"] = out["sold_units"] - out["closing_qty"] * (1.0 - alpha)
    return out.drop(columns=["closing_qty"])


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def add_cyclic_calendar(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    dt = d[date_col].dt
    d["dow"]   = dt.dayofweek
    d["month"] = dt.month
    d["dom"]   = dt.day
    days_in_month = dt.days_in_month
    d["dow_sin"]   = np.sin(2 * np.pi * d["dow"]   / 7)
    d["dow_cos"]   = np.cos(2 * np.pi * d["dow"]   / 7)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["dom_sin"]   = np.sin(2 * np.pi * (d["dom"] - 1) / days_in_month)
    d["dom_cos"]   = np.cos(2 * np.pi * (d["dom"] - 1) / days_in_month)
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    return d


def add_holiday_features(
    df: pd.DataFrame,
    calendar_path: str = "data/external/calendar_raw.parquet",
) -> pd.DataFrame:
    """
    is_public_holiday : 공휴일 (대체공휴일 포함)
    is_before_holiday : 좁은 정의 — 오늘 영업일(평일+비공휴일) AND 내일 휴일(공휴일 OR 주말)
                        예: 금/토/일/월 연휴 시 목요일만 True (금/토/일 모두 False)
                        직관: 연휴 직전 마지막 영업일 (손님이 미리 구매하는 효과)
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    dow_series = d["date"].dt.dayofweek  # 내부 변수만 (df에 추가 X — add_cyclic_calendar에서 처리)

    if not Path(calendar_path).exists():
        d["is_public_holiday"] = 0
        d["is_before_holiday"] = 0
        return d

    cal = pd.read_parquet(calendar_path)
    cal["date"] = pd.to_datetime(cal["date"])
    holiday_dates = set(cal.loc[cal["is_holiday"] == True, "date"])

    d["is_public_holiday"] = d["date"].isin(holiday_dates).astype(int)

    # 좁은 정의: 오늘 영업일(평일 비공휴일) + 내일 휴일(공휴일 OR 주말)
    is_off_today    = (d["is_public_holiday"] == 1) | (dow_series >= 5)
    next_date       = d["date"] + pd.Timedelta(days=1)
    next_dow        = (dow_series + 1) % 7
    is_next_holiday = next_date.isin(holiday_dates)
    is_next_weekend = (next_dow >= 5)
    is_off_next     = is_next_holiday | is_next_weekend

    d["is_before_holiday"] = (~is_off_today & is_off_next).astype(int)
    return d


def _signed_days_to_event(dates: pd.Series, month: int, day: int, clip: int = EVENT_CLIP) -> np.ndarray:
    out = np.full(len(dates), clip + 1, dtype="int64")
    for offset in [-1, 0, 1]:
        years = dates.dt.year + offset
        event_dates = pd.to_datetime({"year": years, "month": month, "day": day}, errors="coerce")
        delta = (event_dates.values - dates.values).astype("timedelta64[D]").astype("int64")
        delta = np.where(pd.isna(event_dates).values, clip + 1, delta)
        better = np.abs(delta) < np.abs(out)
        out = np.where(better, delta, out)
    return np.clip(out, -clip, clip)


def _days_to_lunar_event(dates_series: pd.Series, year_to_date: dict[int, str], clip: int = EVENT_CLIP) -> np.ndarray:
    """양력 변동 이벤트 (추석/설날) 까지 signed days."""
    out = np.full(len(dates_series), clip + 1, dtype="int64")
    for year, event_str in year_to_date.items():
        event = pd.Timestamp(event_str)
        delta = (event - dates_series).dt.days.values
        better = np.abs(delta) < np.abs(out)
        out = np.where(better, delta, out)
    return np.clip(out, -clip, clip)


def add_event_features(df: pd.DataFrame, events: dict = EVENTS, lunar_events: dict = LUNAR_EVENTS) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    dates_s = pd.Series(d["date"])
    # 양력 고정 (xmas/valentine/white_day/children_day)
    for feat_name, (m, day) in events.items():
        d[feat_name] = _signed_days_to_event(dates_s, m, day).astype("int16")
        d[feat_name.replace("days_to_", "is_within7_")] = (np.abs(d[feat_name]) <= 7).astype(int)
    # 음력 변동 (추석/설날)
    for feat_name, year_dates in lunar_events.items():
        d[feat_name] = _days_to_lunar_event(dates_s, year_dates).astype("int16")
        d[feat_name.replace("days_to_", "is_within7_")] = (np.abs(d[feat_name]) <= 7).astype(int)
    return d


def add_weather_features(
    df: pd.DataFrame,
    weather_path: str = "data/external/weather_observed.parquet",
    station_id: int = 119,
) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    if not Path(weather_path).exists():
        return d
    w = pd.read_parquet(weather_path)
    w = w[w["station_id"] == station_id].copy()
    w["date"] = pd.to_datetime(w["date"])

    # 숫자 변환
    num_cols = ["avgTa", "maxTa", "minTa", "sumRn", "avgRhm", "avgTca", "avgWs",
                "hr1MaxRn", "hr1MaxRnHrmt", "maxInsWs", "maxInsWsHrmt"]
    for c in num_cols:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")

    keep = ["date"] + [c for c in ["avgTa", "maxTa", "minTa", "sumRn",
                                    "avgRhm", "avgTca", "avgWs"] if c in w.columns]
    weather_basic = w[keep].copy()

    # 강수 binning + sumRn 자체 fillna (NaN dropna로 row 손실 방지)
    if "sumRn" in weather_basic.columns:
        weather_basic["sumRn"] = weather_basic["sumRn"].fillna(0)
    if "sumRn" in w.columns:
        rn = w["sumRn"].fillna(0)
        weather_basic["rain_level"] = pd.cut(
            rn, bins=[-1, 0, 5, 20, 1e9], labels=[0, 1, 2, 3]
        ).astype(int)

    # hr1MaxRn 시각 (영업시간 강수 추정): 7시~22시면 영업시간, 아니면 외
    if "hr1MaxRnHrmt" in w.columns:
        # HHMM 형식 → hour 추출
        hr_str = w["hr1MaxRnHrmt"].astype(str).str.zfill(4)
        max_rain_hour = pd.to_numeric(hr_str.str[:2], errors="coerce")
        in_biz = (max_rain_hour >= 7) & (max_rain_hour <= 22)
        weather_basic["heavy_rain_in_biz_hours"] = (in_biz & (w["hr1MaxRn"].fillna(0) >= 5)).astype(int)
    else:
        weather_basic["heavy_rain_in_biz_hours"] = 0

    # 체감온도 추정 (간단): avgTa + 풍속 effect
    if "avgTa" in w.columns and "avgWs" in w.columns:
        weather_basic["apparent_temp"] = (
            w["avgTa"].fillna(0) - 0.7 * w["avgWs"].fillna(0)
        )

    d = d.merge(weather_basic, on="date", how="left")
    return d


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r = np.radians(lat1); lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_competitor_features(
    df: pd.DataFrame,
    competitor_path: str = "data/external/competitor_raw.parquet",
    store_lat: float = GWANGYO_LAT,
    store_lon: float = GWANGYO_LON,
    radius_km: float = COMPETITOR_RADIUS_KM,
) -> pd.DataFrame:
    """각 일자별 광교 1km 반경 active bakery/cafe 점포 수."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    if not Path(competitor_path).exists():
        d["n_competitors_active"] = 0
        return d
    c = pd.read_parquet(competitor_path)
    # bakery + cafe만
    if "category" in c.columns:
        c = c[c["category"].isin(["bakery", "cafe", "coffee"])]
    c["license_date"] = pd.to_datetime(c["license_date"], errors="coerce")
    c["close_date"] = pd.to_datetime(c["close_date"], errors="coerce")
    # Haversine 반경 안만
    dist = _haversine_km(store_lat, store_lon, c["lat"].values, c["lon"].values)
    c = c[dist <= radius_km].copy()

    # 각 date별 active 점포 수
    dates = sorted(d["date"].unique())
    counts = []
    for date in dates:
        active = (
            (c["license_date"] <= date) &
            (c["close_date"].isna() | (c["close_date"] >= date))
        )
        counts.append({"date": date, "n_competitors_active": int(active.sum())})
    comp_daily = pd.DataFrame(counts)
    d = d.merge(comp_daily, on="date", how="left")
    d["n_competitors_active"] = d["n_competitors_active"].fillna(0).astype(int)
    return d


def add_lag_rolling_ewma(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True).copy()
    shifted = d[target_col].shift(1)
    for lag in LAG_DAYS:
        d[f"{target_col}_lag{lag}"] = d[target_col].shift(lag)
    for w in ROLLING_WINDOWS:
        d[f"{target_col}_rmean{w}"] = shifted.rolling(w, min_periods=max(2, w//3)).mean()
        d[f"{target_col}_rstd{w}"]  = shifted.rolling(w, min_periods=max(2, w//3)).std()
    for hl in EWMA_HALFLIVES:
        d[f"{target_col}_ewma{hl}"] = shifted.ewm(halflife=hl, min_periods=max(2, hl//3)).mean()
    return d


def build_features(cd: CategoryDaily, target_col: str = "adjusted_demand_unit") -> pd.DataFrame:
    """전체 features pipeline."""
    df = cd.df.copy()
    df = add_cyclic_calendar(df)
    df = add_holiday_features(df)
    df = add_event_features(df)
    df = add_weather_features(df)
    df = add_competitor_features(df)
    df = add_lag_rolling_ewma(df, target_col)
    return df
