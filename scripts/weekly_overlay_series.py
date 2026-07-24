"""주간 리포트용 — 빵 총량 일별 계열 오버레이 (방법론별), canonical 3cat 단일 소스.

모든 계열을 build_category_daily(daily_raw=None → canonical bonavi_daily, TARGET_CATEGORIES
= bread/pastry/sandwich, 옛 0520 closing = 발행 WAPE/발주 KPI와 동일 타깃)로 통일한다.
발주/예측/실측 모두 동일 (카테고리·소스·closing) → 같은 축 비교 정합.
scratchpad에 overlay_unified.parquet 산출.
"""
import os
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.cli import _category_order_predictions
from bakery.ingest.inventory import load_inventory
from store_predictive_power import (
    windowed_backtest, STORE_EVENT_PRIORS, DEFAULT_WINDOW_DAYS, MAIN_FOLDS, TARGET,
)

# 출력 디렉토리: WEEKLY_OUT_DIR 환경변수 우선, 없으면 reports/weekly (repo 내 기본).
SCR = os.environ.get("WEEKLY_OUT_DIR", "reports/weekly")
os.makedirs(SCR, exist_ok=True)
STORE = "store_gw01"; ALPHA = 0.8; VW = 8; NF = 8
INV_XLSX = "data/internal/보나비 데이터_20260526.xlsx"
CATS = ("bread", "pastry", "sandwich")

# 1) canonical 3cat 카테고리 총량 일별 (실수요·실판매) — 발행 수치와 동일 타깃
cd = build_category_daily(alpha=ALPHA)          # None→canonical, TARGET_CATEGORIES=3, 옛 closing
feat = build_features(cd, target_col=TARGET)    # build_features는 CategoryDaily를 받는다
base = cd.df[["date", TARGET, "sold_total_unit"]].rename(
    columns={TARGET: "adjusted_demand", "sold_total_unit": "sold_excl_bulk"})
base["date"] = pd.to_datetime(base["date"])

# 2) 예측 (헤드라인 expected — 발행 WAPE 재현)
cfg = STORE_EVENT_PRIORS["광교"]
main = windowed_backtest(feat, window_days=DEFAULT_WINDOW_DAYS, n_folds=MAIN_FOLDS,
                         events=cfg.get("events"), lunar_events=cfg.get("lunar_events"))
mp = main.predictions.copy(); mp["date"] = pd.to_datetime(mp["date"])
pred = mp.groupby("date")["expected"].sum().rename("pred_curstack").reset_index()

# 3) 발주 (quantile/nk15/nk30 — 동일 canonical 3cat, conformal은 test창 좁아 제외)
specs = {"order_quantile": dict(margin_method="quantile"),
         "order_nk15": dict(margin_method="nk", nk_mult=1.0, nk_add=15.0),
         "order_nk30": dict(margin_method="nk", nk_mult=1.0, nk_add=30.0)}
order_series = []
for col, kw in specs.items():
    p = _category_order_predictions(STORE, production_quantile=0.85, val_weeks=VW,
                                    n_folds=NF, alpha=ALPHA, **kw)
    p["date"] = pd.to_datetime(p["date"])
    order_series.append(p.groupby("date")["our_order"].sum().rename(col))

# 4) 아티제 실발주(=실생산 QT_MADE) — canonical inventory, 3cat 필터
inv = load_inventory(INV_XLSX, STORE)
inv["item_id"] = inv["item_id"].astype(str); inv["date"] = pd.to_datetime(inv["date"])
bd = pd.read_parquet("data/internal/bonavi_daily.parquet")[["item_id", "category_id"]].drop_duplicates()
bd["item_id"] = bd["item_id"].astype(str)
inv = inv.merge(bd, on="item_id", how="left")
inv = inv[inv["category_id"].isin(CATS)]
qt = inv.groupby("date")["production_qty"].sum().rename("qt_made").reset_index()

# merge (실측·qt는 전 구간, 예측·발주는 OOS — 최근 8주 그릴 때 다 존재)
out = base.merge(pred, on="date", how="left").merge(qt, on="date", how="left")
for s in order_series:
    out = out.merge(s, on="date", how="left")

# naive = 직전 4주 동일요일 평균(adjusted에서; 그래프용 근사)
s2 = out.set_index("date")["adjusted_demand"].sort_index()
full = s2.reindex(pd.date_range(s2.index.min() - pd.Timedelta(days=28), s2.index.max(), freq="D"))
out["pred_naive"] = [np.nanmean([full.get(d - pd.Timedelta(days=7 * k), np.nan) for k in (1, 2, 3, 4)])
                     for d in out["date"]]
out = out.sort_values("date").reset_index(drop=True)
out.to_parquet(f"{SCR}/overlay_unified.parquet", index=False)

wape = (mp["expected"] - mp["actual"]).abs().sum() / mp["actual"].abs().sum() * 100
oos = out.dropna(subset=["order_quantile"])
freq = (oos["order_quantile"] < oos["adjusted_demand"]).mean() * 100
print(f"wrote overlay_unified.parquet rows={len(out)} span {out.date.min().date()}~{out.date.max().date()}")
print(f"headline WAPE 재현: {wape:.2f}%  (발행 7.8% 기대)")
print(f"order_quantile < adjusted 빈도: {freq:.0f}%  (KPI 카테고리매진 42% 수렴 기대)")
print(out[["date","adjusted_demand","sold_excl_bulk","qt_made","pred_curstack","order_quantile"]].tail(3).to_string())
