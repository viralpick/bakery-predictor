"""타 3매장 category_total 참조 예측 (event_prior 없이). 타깃 아님·참조용.

브리프(.superpowers/sdd/task-4-brief.md) 대비 어댑테이션:
- events=None, lunar_events=None 대신 events={}, lunar_events={} 로 명시.
  windowed_backtest -> EventLevelPrior(events=events, ...)이며 EventLevelPrior.__init__은
  `events is None`일 때 DEFAULT_EVENTS={"xmas": (12, 25)}로 폴백한다
  (src/bakery/models/event_prior.py:24). 즉 events=None을 그대로 넘기면
  "event_prior 미적용"이 아니라 xmas 레벨-앵커가 암묵적으로 켜진 채로 돈다(런너의 기본
  no-event_prior 경로도 동일 특성). 타매장엔 event_prior를 절대 적용하지 않아야 하므로
  빈 dict를 명시 전달해 is_event_day()가 항상 False가 되게 한다(blend는 그 경우 no-op).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
from bakery.data import paths
from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.analysis.discount import load_sales_with_discount_v2, load_closing_returns_v2
from bakery.ingest.inventory import STORE_CODE_MAPPING
from bakery.harness.backtest_core import windowed_backtest, metrics_from_preds
from bakery.harness.registry import build_forecaster

REFERENCE_STORES = ["store_ss01", "store_mp01", "store_gh01"]  # 광교 제외
ms = pd.read_parquet(paths.dataset("multistore_daily"))
rows = []
for store_id in REFERENCE_STORES:
    code = STORE_CODE_MAPPING[store_id]
    daily_raw = ms[ms["store_id"] == store_id].copy()
    # ★store별 closing 명시 전달 (기본 광교 오염 방지)
    dr = load_sales_with_discount_v2(store_code=code).closing_discount()
    cr = load_closing_returns_v2(store_code=code)
    cd = build_category_daily(daily_raw=daily_raw, discount_rows=dr,
                              closing_returns=cr, alpha=0.8)
    feat = build_features(cd, target_col="adjusted_demand_unit")
    fc = build_forecaster("category_total")
    bt = windowed_backtest(
        feat, window_days=730, target_col="adjusted_demand_unit",
        n_folds=52, horizon_days=7, production_q=0.85, alpha=0.8,
        events={}, lunar_events={}, forecaster=fc,   # event_prior 미적용(빈 dict로 명시 disable)
    )
    m = metrics_from_preds(bt.predictions)
    rows.append({"store_id": store_id, **m})
    print(store_id, m)
out = pd.DataFrame(rows)
out.to_csv("docs/phase7/multistore_reference.csv", index=False)
print("wrote docs/phase7/multistore_reference.csv")
