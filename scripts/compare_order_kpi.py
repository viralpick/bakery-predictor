"""[일회성] 발주 정책(forecaster) 비용 KPI 비교 — category_total vs distributional_total.

harness-run(experiments/gwangyo_compare.yaml) 산출물인 두 arm의 predictions.csv
(카테고리 총량 `production`)을 읽어 품목 배분 → 품목별 비용/매진 KPI로 비교한다.
재구현 없음 — 전부 기존 프리미티브 호출:
  distribute_total(item_proportion) / build_item_adjusted_demand(category_aggregate) /
  build_arrival_profile(prospective) / order_cost·stockout_timing·summarize_order_kpi
  (evaluation.order_cost, 이번 태스크에서 신설) / EventLevelPrior.is_event_day.

결론이 재사용된다면(예: 정기 비교 리포트로 굳어지면) 이 스크립트는 프리미티브+registry로
승격해야 한다(.claude/CLAUDE.md 라우팅 표, escape hatch 조건).

실행:
    uv run python scripts/compare_order_kpi.py
    uv run python scripts/compare_order_kpi.py --absorption-k 1.0 0.7 0.4 --early-hour 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bakery.analysis.seasonal import filter_seasonal
from bakery.cli import REAL_INVENTORY_XLSX_PATH, _load_real_receipts, _load_unit_prices
from bakery.data import paths
from bakery.evaluation.order_cost import (
    DEFAULT_ABSORPTION_K,
    DEFAULT_EARLY_STOCKOUT_HOUR,
    category_stockout_cost,
    order_cost,
    stockout_timing,
    summarize_order_kpi,
)
from bakery.evaluation.prospective import build_arrival_profile
from bakery.features.category_aggregate import DEFAULT_ALPHA as ADJUSTED_DEMAND_ALPHA
from bakery.features.category_aggregate import TARGET_CATEGORIES, build_item_adjusted_demand
from bakery.features.potential_demand import StoreHours
from bakery.harness.event_priors import resolve_event_priors
from bakery.models.event_prior import EventLevelPrior
from bakery.models.item_proportion import distribute_total

STORE_ID = "store_gw01"
EVENT_PRIOR_KEY = "gwangyo"
DEFAULT_OPEN_HOUR = 8
DEFAULT_CLOSE_HOUR = 21
DEFAULT_UNIT_PRICE = 4000.0
DEFAULT_CATEGORY_CSV = "reports/gwangyo_compare/category_total/predictions.csv"
DEFAULT_DISTRIBUTIONAL_CSV = "reports/gwangyo_compare/distributional_total/predictions.csv"
DEFAULT_OUTPUT_CSV = "reports/order_kpi_compare.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category-total-csv", default=DEFAULT_CATEGORY_CSV)
    parser.add_argument("--distributional-csv", default=DEFAULT_DISTRIBUTIONAL_CSV)
    parser.add_argument("--absorption-k", type=float, nargs="+", default=[DEFAULT_ABSORPTION_K])
    parser.add_argument("--early-hour", type=int, default=DEFAULT_EARLY_STOCKOUT_HOUR)
    parser.add_argument("--open-hour", type=int, default=DEFAULT_OPEN_HOUR)
    parser.add_argument("--close-hour", type=int, default=DEFAULT_CLOSE_HOUR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def load_item_history() -> pd.DataFrame:
    """광교 bread/pastry/sandwich item-level daily + adjusted_demand(alpha=0.8)."""
    daily = pd.read_parquet(paths.dataset("bonavi_daily"))
    daily["item_id"] = daily["item_id"].astype(str)
    daily = filter_seasonal(daily)
    daily = daily[daily["category_id"].isin(TARGET_CATEGORIES)].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    return build_item_adjusted_demand(daily, alpha=ADJUSTED_DEMAND_ALPHA)


def load_arm_predictions(csv_path: str) -> pd.Series:
    """harness predictions.csv → date→production(카테고리 총량 발주) Series."""
    preds = pd.read_csv(csv_path, parse_dates=["date"])
    return preds.set_index("date")["production"]


def build_item_rows(history: pd.DataFrame, total_by_date: pd.Series, price_map: dict) -> pd.DataFrame:
    """카테고리 총량 발주를 distribute_total로 품목 배분 + 품목 실수요/단가 부착.

    해당 item-day가 history에 없으면(그날 무판매) adjusted_demand=0으로 채운다.
    """
    result = distribute_total(history, total_by_date)
    rows = result.quantities.rename(columns={"qty": "order_qty"})
    demand = history[["date", "item_id", "adjusted_demand"]]
    rows = rows.merge(demand, on=["date", "item_id"], how="left")
    rows["adjusted_demand"] = rows["adjusted_demand"].fillna(0.0)
    rows["unit_price"] = rows["item_id"].map(price_map).fillna(DEFAULT_UNIT_PRICE)
    return rows


def build_arrival_profiles(history: pd.DataFrame) -> dict[tuple, object]:
    """bonavi_receipts(bulk 제외·수량가중) → 품목별 24시간 도착곡선."""
    receipts = _load_real_receipts(set(history["item_id"]))
    return build_arrival_profile(receipts, group_cols=["item_id"])


def cost_and_timing(
    rows: pd.DataFrame, profiles: dict, store_hours: StoreHours, *, absorption_k: float, early_hour: int
) -> pd.DataFrame:
    """order_cost(비용) + stockout_timing(매진시각) 결합."""
    costed = order_cost(
        rows, order_col="order_qty", demand_col="adjusted_demand",
        price_col="unit_price", absorption_k=absorption_k,
    )
    return stockout_timing(
        costed, profiles, order_col="order_qty", demand_col="adjusted_demand",
        store_hours=store_hours, group_cols=["item_id"], early_hour=early_hour,
    )


def attach_event_flag(costed: pd.DataFrame, event_prior: EventLevelPrior) -> pd.DataFrame:
    """EventLevelPrior.is_event_day 기반 이벤트일 플래그 부착."""
    out = costed.copy()
    out["is_event_day"] = out["date"].apply(event_prior.is_event_day)
    return out


def segment_summaries(costed: pd.DataFrame, *, early_hour: int) -> list[tuple[str, int, dict]]:
    """전체/이벤트/비이벤트 구간별 KPI 요약.

    ★전체매진(카테고리) 비용은 구간별로 **그 구간의 날짜만** 다시 집계한다 —
    품목 단위 합계와 달리 날짜 단위 항이라 사후 분할이 안 된다.
    """
    out = []
    segments = (("all", costed), ("event", costed[costed["is_event_day"]]),
                ("non_event", costed[~costed["is_event_day"]]))
    for name, seg in segments:
        if seg.empty:
            continue
        cat = category_stockout_cost(seg, order_col="order_qty", demand_col="adjusted_demand")
        out.append((name, len(seg), summarize_order_kpi(seg, early_hour=early_hour, category=cat)))
    return out


def run_comparison(args: argparse.Namespace) -> pd.DataFrame:
    """두 arm × k 스윕 × 3분해(all/event/non_event) 전부 순회해 결과 행 생성."""
    history = load_item_history()
    price_map = _load_unit_prices(REAL_INVENTORY_XLSX_PATH)
    profiles = build_arrival_profiles(history)
    store_hours = StoreHours(store_id=STORE_ID, open_hour=args.open_hour, close_hour=args.close_hour)
    events, lunar_events = resolve_event_priors(EVENT_PRIOR_KEY)
    event_prior = EventLevelPrior(events=events, lunar_events=lunar_events)

    arms = {"category_total": args.category_total_csv, "distributional_total": args.distributional_csv}
    out_rows = []
    for arm_name, csv_path in arms.items():
        rows = build_item_rows(history, load_arm_predictions(csv_path), price_map)
        for k in args.absorption_k:
            costed = cost_and_timing(rows, profiles, store_hours, absorption_k=k, early_hour=args.early_hour)
            costed = attach_event_flag(costed, event_prior)
            for segment, n_rows, summary in segment_summaries(costed, early_hour=args.early_hour):
                out_rows.append({
                    "arm": arm_name, "absorption_k": k, "segment": segment, "n_rows": n_rows, **summary,
                })
    return pd.DataFrame(out_rows)


def main() -> None:
    args = parse_args()
    result = run_comparison(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
