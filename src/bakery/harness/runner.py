from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.features.forecast_panel import build_forecast_panel
from bakery.harness.backtest_core import metrics_from_preds, windowed_backtest
from bakery.harness.config import ExperimentSpec
from bakery.harness.event_priors import resolve_event_priors
from bakery.harness.panel_backtest import panel_backtest
from bakery.harness.registry import build_forecaster, is_runnable

STAGES: tuple[str, ...] = ("features", "backtest", "evaluate")


@dataclass
class RunResult:
    name: str
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    metrics: dict
    resolved: dict
    # 품목 배분 발주. order_level="item"일 때만 채워진다(KPI 입력).
    item_orders: pd.DataFrame | None = None


@dataclass
class ExperimentResult:
    name: str
    runs: dict[str, RunResult]
    comparison: pd.DataFrame
    # KPI 표(A/B basis 병기 + 아띠제 대비 절감률). spec.kpi=True일 때만 채운다.
    kpi: pd.DataFrame | None = None


KPI_OPEN_HOUR = 8
# ★광교 영업 종료 = 22시 (architect 확인 + 광교 실측 정합).
# 근거: `영업시간` 시트(광교=**1000000047**) median **21.93시**, 2026 상반기 median 21.83시.
# 영수증 마지막 판매 median 21.82시와 **날짜별로 76%가 15분 내 일치**(상관 0.495)하고
# 시트값이 오르면 마지막 판매도 오른다(20시대→20.87 / 21시대→21.70 / 22시+→22.12).
# 즉 이 시트는 광교에서 유효한 마감시각이며, 반올림하면 22시다.
# ⚠️**매장 코드 주의** — `1000000047`=광교 / `1000000009`=삼성타운. 처음 이 상수를 21로
# 잡았던 이유가 삼성타운 시트값(median 21.42)을 광교 영수증과 비교한 교차매장 오류였다.
# 매장은 `STORE_CODE_MAPPING`(store_gw01) 을 통해 접근할 것.
KPI_CLOSE_HOUR = 22
KPI_PRICE_FALLBACK = 4000.0


def _stage_key(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _load_or_compute(stage, key, cache_dir, compute, trace):
    if cache_dir is None:
        trace.append((stage, "nocache"))
        return compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{stage}_{key}.parquet"
    if path.exists():
        trace.append((stage, "hit"))
        return pd.read_parquet(path)
    trace.append((stage, "miss"))
    df = compute()
    df.to_parquet(path)
    return df


def _load_item_history(spec: ExperimentSpec) -> pd.DataFrame:
    """배분 비율 계산용 품목 일별 프레임.

    ★panel 엔진은 아직 미지원 — fold가 원점 기준이라 배분 대상일 매핑이 다르다.
    조용히 틀린 배분을 내는 대신 fails-loud한다.
    """
    if spec.engine == "panel":
        raise ValueError("order_level='item'은 engine='panel'과 아직 함께 쓸 수 없다(배분 대상일 매핑 미정의).")
    from bakery.cli import _load_real_daily

    daily = _load_real_daily(spec.data.store)
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def _kpi_inputs(spec: ExperimentSpec) -> dict:
    """KPI 계산 재료 — 품목 실수요 / 단가 / 아띠제 실측(A basis) / 도착 프로필.

    전부 기존 프리미티브 호출이다(재구현 0).
    """
    from bakery.cli import REAL_INVENTORY_XLSX_PATH, _load_real_receipts, _load_unit_prices
    from bakery.evaluation.prospective import build_arrival_profile
    from bakery.features.category_aggregate import (
        DEFAULT_ALPHA,
        TARGET_CATEGORIES,
        build_item_adjusted_demand,
    )
    from bakery.ingest.inventory import load_inventory

    daily = _load_item_history(spec)
    demand = build_item_adjusted_demand(daily, alpha=spec.alpha)
    prices = _load_unit_prices(REAL_INVENTORY_XLSX_PATH)
    inventory = load_inventory(str(REAL_INVENTORY_XLSX_PATH), spec.data.store)
    inventory = inventory.copy()
    inventory["date"] = pd.to_datetime(inventory["date"])
    inventory["item_id"] = inventory["item_id"].astype(str)
    cat_map = daily[["item_id", "category_id"]].drop_duplicates()
    cat_map["item_id"] = cat_map["item_id"].astype(str)
    inventory = inventory.merge(cat_map, on="item_id", how="inner")
    inventory = inventory[inventory["category_id"].isin(TARGET_CATEGORIES)]
    # _load_real_receipts는 store가 아니라 item_ids를 받는다(receipts에 store 컬럼 없음).
    receipts = _load_real_receipts(set(inventory["item_id"].astype(str)))
    profiles = build_arrival_profile(receipts, group_cols=["item_id"])
    return {
        "demand": demand, "prices": prices, "inventory": inventory,
        "profiles": profiles, "alpha_used": DEFAULT_ALPHA,
    }


def _kpi_rows(spec: ExperimentSpec, runs: dict, materials: dict) -> pd.DataFrame:
    """arm별 B basis + 공통 A basis + 절감률을 한 표로."""
    from bakery.evaluation.order_cost import order_cost, stockout_timing
    from bakery.evaluation.order_kpi import (
        basis_actual,
        basis_sim,
        compare_to_actual,
        kpi_table,
        waste_negative_diagnostics,
    )
    from bakery.features.potential_demand import StoreHours

    demand = materials["demand"][["date", "item_id", "adjusted_demand"]].copy()
    demand["item_id"] = demand["item_id"].astype(str)
    prices, inventory = materials["prices"], materials["inventory"]
    hours = StoreHours(spec.data.store, KPI_OPEN_HOUR, KPI_CLOSE_HOUR)

    records: list[dict] = []
    dates_seen: set = set()
    for name, run in runs.items():
        orders = run.item_orders
        if orders is None or orders.empty:
            continue
        rows = orders.copy()
        rows["item_id"] = rows["item_id"].astype(str)
        rows = rows.merge(demand, on=["date", "item_id"], how="left")
        rows["adjusted_demand"] = rows["adjusted_demand"].fillna(0.0)
        rows["unit_price"] = rows["item_id"].map(prices).fillna(KPI_PRICE_FALLBACK)
        costed = order_cost(rows, order_col="order_qty", demand_col="adjusted_demand",
                            price_col="unit_price")
        costed = costed.join(stockout_timing(
            rows, materials["profiles"], order_col="order_qty",
            demand_col="adjusted_demand", store_hours=hours, group_cols=["item_id"],
        )[["soldout_hour", "is_stockout"]])
        records.append({"policy": name, **basis_sim(
            costed, order_col="order_qty", demand_col="adjusted_demand")})
        dates_seen |= set(pd.to_datetime(orders["date"]).unique())

    if not records:
        return pd.DataFrame()
    # ★A basis는 모델이 평가한 **같은 날짜**로 제한한다 — 안 하면 분모가 달라져
    #   절감률이 기간 차이를 절감으로 오독한다.
    actual_rows = inventory[inventory["date"].isin(dates_seen)].copy()
    actual_rows["unit_price"] = actual_rows["item_id"].map(prices).fillna(KPI_PRICE_FALLBACK)
    actual_rows["is_stockout"] = (
        (actual_rows["production_qty"] > 0) & (actual_rows["waste_qty"] <= 0)
    )
    actual = basis_actual(actual_rows)

    # ★★actual_sim — 아띠제 실생산(QT_MADE)을 **B 잣대로** 다시 잰다.
    # 모델(B)을 아띠제(A)와 직접 비교하면 절감에 censoring(잣대 효과)이 섞인다.
    # 같은 정책을 A/B로 각각 재면 그 차이가 순수 잣대 효과이고, 모델 vs actual_sim이
    # 공정 비교다. 옛 헤드라인 "−37~45%"도 이 ΔvsB 축이다.
    sim_rows = actual_rows.merge(demand, on=["date", "item_id"], how="left")
    sim_rows["adjusted_demand"] = sim_rows["adjusted_demand"].fillna(0.0)
    sim_rows = sim_rows.rename(columns={"production_qty": "order_qty"})
    sim_costed = order_cost(sim_rows, order_col="order_qty", demand_col="adjusted_demand",
                            price_col="unit_price")
    actual_sim = basis_sim(sim_costed, order_col="order_qty", demand_col="adjusted_demand")

    records.append({"policy": "artisee_actual", **actual})
    records.append({"policy": "artisee_actual_sim", **actual_sim})
    for rec in records:
        if rec["policy"].startswith("artisee_actual"):
            continue
        rec.update(compare_to_actual(rec, actual, actual_sim=actual_sim))
    diag = waste_negative_diagnostics(actual_rows)
    table = kpi_table(records)
    for key, value in diag.items():
        table[f"actual_waste_{key}"] = value
    return table


def run_experiment(
    spec: ExperimentSpec, *, out_dir: Path, cache_dir: Path | None = None,
    _trace: list | None = None,
) -> ExperimentResult:
    trace = _trace if _trace is not None else []
    runnable = [f for f in spec.forecaster if is_runnable(f)]
    for f in spec.forecaster:
        if not is_runnable(f):
            warnings.warn(f"forecaster '{f}'는 실행 미지원(point/composite) — 스킵.", UserWarning)
    if not runnable:
        raise ValueError("실행 가능한 forecaster 없음(category_total/distributional_total 필요).")

    feat_key = _stage_key({"engine": spec.engine, "order_level": spec.order_level,
                           "source": spec.data.source, "store": spec.data.store,
                           "target": spec.target, "alpha": spec.alpha})

    def _feat():
        cd = build_category_daily(alpha=spec.alpha)
        if spec.engine == "panel":
            return build_forecast_panel(cd, target_col=spec.target)
        return build_features(cd, target_col=spec.target)

    feat = _load_or_compute("features", feat_key, cache_dir, _feat, trace)
    # 배분 비율 소스(품목 일별). order_level="item"일 때만 읽는다 — 헤드라인 경로는
    # 이 I/O를 타지 않는다.
    item_history = _load_item_history(spec) if spec.order_level == "item" else None
    events, lunar = resolve_event_priors(spec.event_priors) if "event_prior" in spec.layers else (None, None)

    out = out_dir / spec.name
    out.mkdir(parents=True, exist_ok=True)
    resolved = spec.model_dump()
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True), encoding="utf-8")

    runs: dict[str, RunResult] = {}
    rows = []
    for fname in runnable:
        fc = build_forecaster(fname)
        trace.append((f"backtest:{fname}", "run"))
        if spec.engine == "panel":
            # 패널은 fold를 원점으로 자른다(대상일 블록 아님) → 형제 함수 사용.
            bt = panel_backtest(
                feat, window_days=spec.window.window_days, target_col=spec.target,
                n_folds=spec.window.n_folds, production_q=spec.production_q,
                alpha=spec.alpha, events=events, lunar_events=lunar, forecaster=fc,
            )
        else:
            bt = windowed_backtest(
                feat, window_days=spec.window.window_days, target_col=spec.target,
                n_folds=spec.window.n_folds, horizon_days=spec.window.horizon_days,
                production_q=spec.production_q, alpha=spec.alpha,
                events=events, lunar_events=lunar, forecaster=fc,
                lead_days=spec.window.lead_days, anchor_dow=spec.window.anchor_dow,
                # gapless(dropna 이전) 프레임 — AR 재계산은 위치 shift라 연속성이 필수.
                ar_history=feat[["date", spec.target]] if spec.window.align_features else None,
                order_level=spec.order_level, item_history=item_history,
            )
        metrics = metrics_from_preds(bt.predictions)
        fout = out / fname
        fout.mkdir(parents=True, exist_ok=True)
        bt.predictions.to_csv(fout / "predictions.csv", index=False)
        bt.folds.to_csv(fout / "fold_results.csv", index=False)
        if bt.item_orders is not None:
            bt.item_orders.to_csv(fout / "item_orders.csv", index=False)
        (fout / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        runs[fname] = RunResult(name=fname, predictions=bt.predictions,
                                fold_metrics=bt.folds, metrics=metrics, resolved=resolved,
                                item_orders=bt.item_orders)
        rows.append({"forecaster": fname, **metrics})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "comparison.csv", index=False)

    kpi = None
    if spec.kpi:
        trace.append(("kpi", "run"))
        kpi = _kpi_rows(spec, runs, _kpi_inputs(spec))
        if not kpi.empty:
            kpi.to_csv(out / "kpi.csv", index=False)
    return ExperimentResult(name=spec.name, runs=runs, comparison=comparison, kpi=kpi)
