"""bakery CLI — generate-data / backtest / predict-next-week."""

from __future__ import annotations

from collections.abc import Collection
from datetime import date as Date
from pathlib import Path

import click
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .config import EXTERNAL_DATA_DIR
from .data.loader import DailyDataset, load_dataset
from .decision import PolicyParams, RiskParams, build_recommendation, lineage_to_frame
from .data.synthetic import generate_synthetic_bundle
from .data.weather import load_weather_forecast_from_local
from .evaluation.backtest import aggregate_by_model, per_category_wape, run_backtest
from .evaluation.business_metrics import CostParams, asymmetric_loss, simulate_profit
from .evaluation.classifier_metrics import base_rate, precision_at_k, recall_at_k, roc_auc
from .evaluation.diagnostics import decoupling_score
from .evaluation.metrics import quantile_exceedance_rate, wape, wpe
from .evaluation.prospective import (
    aggregate_fold_kpis,
    build_arrival_profile,
    compare_actual_vs_simulated_waste,
    compare_policies,
    compare_policies_by_fold,
    reconstruct_baseline_order,
    simulate_item_day_kpis,
)
from .evaluation.split import SplitWindow, apply_split, generate_time_splits
from .features.calendar_features import add_calendar_features
from .features.category_aggregate import (
    DEFAULT_ALPHA, EVENTS, LUNAR_EVENTS, TARGET_CATEGORIES, CategoryDaily,
    build_category_daily, build_features, build_item_adjusted_demand, fill_forecast_weather,
)
from .features.competitor_features import (
    add_competitor_features,
    compute_competitor_features,
)
from .features.consumption_features import (
    add_consumption_features,
    compute_store_consumption_features,
)
from .features.living_population_features import (
    add_living_pop_features,
    compute_store_living_features,
)
from .features.population_features import (
    add_population_features,
    compute_store_population_features,
)
from .features.potential_demand import StoreHours
from .features.scale import compute_item_scale
from .features.weather_features import add_weather_features
from .ingest import (
    calendar_api,
    competitor_api,
    consumption_api,
    forecast_api,
    living_population_api,
    living_population_csv,
    population_api,
    weather_api,
)
from .ingest.inventory import load_inventory, handle_negative_waste
from .ingest.store_mapping import load_store_mapping
from .models.artisee_baseline import ArtiseeBaseline
from .models.category_total import fit_category_total
from .models.distributional_total import fit_distributional_total
from .models.event_prior import EventLevelPrior
from .models.conformal_order import ConformalOrderCalibrator, DEFAULT_SERVICE_LEVEL
from .models.item_proportion import distribute_total
from .models.lightgbm_regressor import (
    FEATURE_GROUPS as LGBM_FEATURE_GROUPS,
    VALID_FEATURE_SETS,
    GlobalLGBM,
    LGBMParams,
)
from .models.moving_average import MovingAverage
from .models.seasonal_naive import SeasonalNaive
from .models.stockout_classifier import StockoutClassifier

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
REPORTS_DIR = Path("reports")


@app.command("generate-data")
def cmd_generate_data(
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
    out_dir: Path = REPORTS_DIR,
) -> None:
    """Materialize synthetic hourly/daily/weather/calendar parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = generate_synthetic_bundle(start=start, end=end, seed=seed)
    bundle.hourly.to_parquet(out_dir / "hourly.parquet", index=False)
    bundle.daily.to_parquet(out_dir / "daily.parquet", index=False)
    bundle.weather.to_parquet(out_dir / "weather.parquet", index=False)
    bundle.calendar.to_parquet(out_dir / "calendar.parquet", index=False)
    bundle.competitor.to_parquet(out_dir / "competitor.parquet", index=False)
    bundle.living_population.to_parquet(out_dir / "living_population.parquet", index=False)
    bundle.population.to_parquet(out_dir / "population.parquet", index=False)
    bundle.consumption.to_parquet(out_dir / "consumption.parquet", index=False)
    console.print(
        f"[green]wrote[/] hourly={len(bundle.hourly):,} daily={len(bundle.daily):,} "
        f"weather={len(bundle.weather):,} calendar={len(bundle.calendar):,} "
        f"competitor={len(bundle.competitor):,} living_pop={len(bundle.living_population):,} "
        f"population={len(bundle.population):,} consumption={len(bundle.consumption):,} → {out_dir}"
    )
    console.print(f"  stockout days: {int(bundle.daily['is_stockout'].sum()):,} / {len(bundle.daily):,}")


@app.command("backtest")
def cmd_backtest(
    source: str = "synthetic",
    data_dir: Path | None = None,
    n_splits: int = 4,
    horizon_days: int = 7,
    step_days: int = 7,
    variants: str = "v0,v1",
    include_production: bool = False,
    out_dir: Path = REPORTS_DIR,
    closing_alpha: float = DEFAULT_ALPHA,
    drop: str = "",
) -> None:
    """Compare baselines + LightGBM variants on the same rolling folds.

    --drop weather,competitor: LightGBM에서 뺄 feature 그룹(실험/ablation용).
    선택: calendar,weather,cannibalization,competitor,living_pop,population,consumption.
    """
    variant_list = _parse_variants(variants)
    drop_groups = _parse_drop_groups(drop)
    ds = _load_dataset(source, data_dir)
    daily = _enrich_if_needed(ds, variant_list)
    daily, demand_col = _resolve_demand_col(daily, source, closing_alpha)
    windows = generate_time_splits(
        daily["date"], n_splits=n_splits, val_horizon_days=horizon_days, step_days=step_days
    )
    forecasters = _build_forecasters(variant_list, include_production=include_production,
                                     v23_target=demand_col, drop_groups=drop_groups)
    console.print(
        f"[cyan]backtest[/] folds={len(windows)} horizon={horizon_days}d "
        f"variants={variant_list} models={[f.name for f in forecasters]}"
    )
    fold_df, pred_df = run_backtest(daily, forecasters, windows)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "fold_results.csv", index=False)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    _print_summary(fold_df, pred_df)
    console.print(f"[green]wrote[/] {out_dir}/fold_results.csv, predictions.csv")


def _predict_next_week_category(
    store_id: str, *, production_quantile: float, total_model: str, event_prior: bool,
    alpha: float, use_forecast: bool, out_dir: Path, horizon_days: int = 7,
) -> None:
    """predict-next-week category 모드: 미래 카테고리 발주 → next_week_predictions.csv.

    _category_future_order_predictions(c-1)를 소비해 item 경로와 동일 스키마로 출력한다.
    recommended_production=our_order(q{production_quantile}+prior), demand_col=demand_point(median).
    stockout_prob은 item v2 경로와 동일하게 NaN(σ(x)→item-risk 매핑은 c-2b)."""
    preds = _category_future_order_predictions(
        store_id, horizon_days=horizon_days, production_quantile=production_quantile,
        total_model=total_model, event_prior=event_prior, alpha=alpha, use_forecast=use_forecast,
    )
    demand_col = "yhat_adjusted_demand_unit"
    model_name = f"category_total:{total_model}"
    out = preds.rename(columns={"demand_point": demand_col}).assign(
        stockout_prob=float("nan"),
        recommended_production=preds["our_order"].round(0).to_numpy(),
        model=model_name,
    )
    out[demand_col] = out[demand_col].round(2)
    cols = [
        "store_id", "item_id", "category_id", "date",
        demand_col, "stockout_prob", "recommended_production", "model",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(out_dir / "next_week_predictions.csv", index=False)
    horizon = sorted(out["date"].unique())
    console.print(
        f"[green]wrote[/] {len(out):,} predictions "
        f"({pd.Timestamp(horizon[0]).date()} ~ {pd.Timestamp(horizon[-1]).date()}, model={model_name}) → "
        f"{out_dir}/next_week_predictions.csv"
    )
    _print_next_week_preview(out, demand_col=demand_col)


@app.command("predict-next-week")
def cmd_predict_next_week(
    source: str = "synthetic",
    data_dir: Path | None = None,
    model: str = "lightgbm_v2",
    production_quantile: float = 0.85,
    base_safety_margin: float = 0.15,
    risk_bonus: float = 0.25,
    use_forecast: bool = False,
    out_dir: Path = REPORTS_DIR,
    closing_alpha: float = DEFAULT_ALPHA,
    order_level: str = typer.Option(
        "item", help="item(기존 v0~v3 LGBM) | category(v4 카테고리 총합→배분, real 전용)",
        click_type=click.Choice(["item", "category"]),
    ),
    store_id: str = typer.Option(
        "store_gw01", help="order_level=category의 대상 매장(단일). item 모드는 무시(store-agnostic)."
    ),
    total_model: str = typer.Option(
        "lightgbm", help="order_level=category 총량 모델: lightgbm | distributional(NGBoost). "
                         "prospective-eval과 동일.",
        click_type=click.Choice(["lightgbm", "distributional"]),
    ),
    event_prior: bool = typer.Option(
        True, "--event-prior/--no-event-prior",
        help="order_level=category에서 sharp 이벤트 레벨-앵커 블렌드(기본 on).",
    ),
) -> None:
    """Train on all history; emit demand prediction + recommended production.

    v2: trains two LightGBMs — median (regression) for demand and a
    `production_quantile` (default 0.85) for recommended production. Newsvendor
    intuition: higher quantile = safer against stockouts, more potential waste.

    v1/v0 (legacy): fall back to the demand-times-margin heuristic with
    `recommended_production = yhat * (1 + base_safety_margin + risk_bonus * stockout_prob)`.

    order_level=category: v4 카테고리 총합→배분 경로(real 전용). 미래 horizon 카테고리
    총량을 total_model로 예측(event_prior 블렌드) 후 item 비율로 배분한다.
    """
    if order_level == "category":
        if source != "real":
            raise typer.BadParameter(
                "--order-level category는 --source real 전용이다 — 카테고리 총합 경로는 "
                "실 bonavi_daily/재고 스키마를 요구한다(synthetic 미지원)."
            )
        _predict_next_week_category(
            store_id, production_quantile=production_quantile, total_model=total_model,
            event_prior=event_prior, alpha=closing_alpha, use_forecast=use_forecast,
            out_dir=out_dir,
        )
        return
    feature_set = _model_to_feature_set(model)
    ds = _load_dataset(source, data_dir)
    daily = _enrich_if_needed(ds, [feature_set]) if feature_set else ds.daily
    if feature_set in {"v2", "v3"}:
        daily, target_col = _resolve_demand_col(daily, source, closing_alpha)
    else:
        target_col = None
    last = daily["date"].max()
    horizon = pd.date_range(last + pd.Timedelta(days=1), periods=7, freq="D")
    forecaster = _pick_model(model)
    if feature_set in {"v2", "v3"}:
        forecaster = GlobalLGBM(feature_set=feature_set, y_col=target_col)
    forecaster.fit(daily)
    pairs = daily[["store_id", "item_id", "category_id"]].drop_duplicates()
    target = pairs.merge(pd.DataFrame({"date": horizon}), how="cross")
    if feature_set in {"v1", "v2", "v3"}:
        forecast_weather = _load_forecast_weather(horizon) if use_forecast else None
        target = _enrich_target(
            target, ds, forecast_weather=forecast_weather, include_external=(feature_set == "v3"),
        )
    yhat = forecaster.predict(target)
    demand_col = f"yhat_{target_col}" if feature_set in {"v2", "v3"} else "yhat_sold_units"
    target = target.assign(**{demand_col: yhat.round(2).to_numpy(), "model": forecaster.name})

    if feature_set in {"v2", "v3"}:
        prod_params = LGBMParams(objective="quantile", alpha=production_quantile)
        prod_model = GlobalLGBM(feature_set=feature_set, params=prod_params,
                                y_col=target_col).fit(daily)
        prod_yhat = prod_model.predict(target)
        target["stockout_prob"] = float("nan")
        target["recommended_production"] = prod_yhat.round(0).to_numpy()
        console.print(f"  production model: {prod_model.name} (quantile α={production_quantile})")
    elif feature_set == "v1":
        clf = StockoutClassifier(feature_set="v1").fit(daily)
        risk = clf.predict_proba(target)
        target["stockout_prob"] = risk.round(4).to_numpy()
        margin = base_safety_margin + risk_bonus * target["stockout_prob"]
        target["recommended_production"] = (target[demand_col] * (1.0 + margin)).round(0)
    else:
        target["stockout_prob"] = float("nan")
        target["recommended_production"] = (target[demand_col] * (1.0 + base_safety_margin)).round(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "store_id", "item_id", "category_id", "date",
        demand_col, "stockout_prob", "recommended_production", "model",
    ]
    target[cols].to_csv(out_dir / "next_week_predictions.csv", index=False)
    if isinstance(forecaster, GlobalLGBM):
        imp = forecaster.feature_importance()
        imp.to_csv(out_dir / f"feature_importance_{forecaster.name}.csv", index=False)
        console.print(f"  feature importance → {out_dir}/feature_importance_{forecaster.name}.csv")
    console.print(
        f"[green]wrote[/] {len(target):,} predictions ({horizon[0].date()} ~ {horizon[-1].date()}) → "
        f"{out_dir}/next_week_predictions.csv"
    )
    _print_next_week_preview(target, demand_col=demand_col)


def _demand_points_next_week(
    ds: DailyDataset, model: str, use_forecast: bool,
    source: str = "synthetic", closing_alpha: float = DEFAULT_ALPHA,
) -> tuple[pd.DataFrame, str]:
    """Per-item demand point estimates for the next 7 days (v6 point estimate)."""
    feature_set = _model_to_feature_set(model)
    daily = _enrich_if_needed(ds, [feature_set]) if feature_set else ds.daily
    if feature_set in {"v2", "v3"}:
        daily, target_col = _resolve_demand_col(daily, source, closing_alpha)
    else:
        target_col = None
    last = daily["date"].max()
    horizon = pd.date_range(last + pd.Timedelta(days=1), periods=7, freq="D")
    forecaster = _pick_model(model)
    if feature_set in {"v2", "v3"}:
        forecaster = GlobalLGBM(feature_set=feature_set, y_col=target_col)
    forecaster.fit(daily)
    pairs = daily[["store_id", "item_id", "category_id"]].drop_duplicates()
    target = pairs.merge(pd.DataFrame({"date": horizon}), how="cross")
    if feature_set in {"v1", "v2", "v3"}:
        fw = _load_forecast_weather(horizon) if use_forecast else None
        target = _enrich_target(target, ds, forecast_weather=fw, include_external=(feature_set == "v3"))
    target["demand_point"] = forecaster.predict(target).clip(lower=0).round(2).to_numpy()
    return target, forecaster.name


@app.command("v6-predict")
def cmd_v6_predict(
    source: str = "synthetic",
    data_dir: Path | None = None,
    model: str = "lightgbm_v2",
    safety_margin: float = 0.15,
    demand_cv: float = 0.30,
    n_samples: int = 5000,
    use_forecast: bool = False,
    out_dir: Path = REPORTS_DIR,
    closing_alpha: float = DEFAULT_ALPHA,
) -> None:
    """v6 산출물: 점추정 + 발주량 + 매진/폐기 위험 수치 + 결정 lineage.

    Forecast(LightGBM 점추정) → 결정론 정책(안전마진·반올림) → Monte-Carlo 위험.
    예측은 학습 코어, 이 레이어는 예측 *이후*의 결정/위험/lineage 껍질이다
    (docs/kinetic_layer_fit_analysis.md §8·§10). 예측 이후 단계라 leakage 없음.
    """
    ds = _load_dataset(source, data_dir)
    items, model_name = _demand_points_next_week(ds, model, use_forecast, source, closing_alpha)
    rec = build_recommendation(
        items[["store_id", "category_id", "item_id", "date", "demand_point"]],
        PolicyParams(safety_margin=safety_margin),
        RiskParams(demand_cv=demand_cv, n_samples=n_samples),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.table.to_csv(out_dir / "v6_recommendations.csv", index=False)
    lineage_to_frame(rec.lineages).to_csv(out_dir / "v6_decision_lineage.csv", index=False)
    console.print(
        f"[green]wrote[/] {len(rec.table):,} v6 recommendations (model={model_name}) → "
        f"{out_dir}/v6_recommendations.csv  (+ v6_decision_lineage.csv)"
    )
    _print_v6_preview(rec.table)


def _print_v6_preview(table: pd.DataFrame, *, top: int = 8) -> None:
    view = table.sort_values("p_stockout", ascending=False).head(top)
    t = Table(title=f"v6 — 매진위험 상위 {min(top, len(view))}품목")
    for col in ("item_id", "demand_point", "order_qty", "p_stockout", "p_waste", "expected_cost"):
        t.add_column(col, justify="right")
    for r in view.itertuples(index=False):
        t.add_row(
            str(r.item_id), f"{r.demand_point:.1f}", f"{r.order_qty:.0f}",
            f"{r.p_stockout:.0%}", f"{r.p_waste:.0%}", f"{r.expected_cost:.1f}",
        )
    console.print(t)


def _parse_variants(variants: str) -> list[str]:
    parts = [v.strip() for v in variants.split(",") if v.strip()]
    bad = [v for v in parts if v not in VALID_FEATURE_SETS]
    if bad:
        raise typer.BadParameter(f"unknown variants {bad}. choose from {list(VALID_FEATURE_SETS)}")
    return parts


def _parse_drop_groups(drop: str) -> frozenset[str]:
    """--drop weather,competitor → frozenset. LightGBM feature 그룹 실험적 제거."""
    parts = frozenset(g.strip() for g in drop.split(",") if g.strip())
    bad = parts - LGBM_FEATURE_GROUPS.keys()
    if bad:
        raise typer.BadParameter(
            f"unknown feature groups {sorted(bad)}. choose from {sorted(LGBM_FEATURE_GROUPS)}"
        )
    return parts


def _resolve_demand_col(
    daily: pd.DataFrame,
    source: str,
    closing_alpha: float,
    discount_rows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """소스별 수요 컬럼 결정.

    real  → build_item_adjusted_demand로 adjusted_demand 부착 후 컬럼명 반환.
    synth → 입력 프레임 그대로, 'potential_demand'.

    potential_demand는 real에서 stockout_time 로더 버그로 오염돼 소비 금지
    (docs/superpowers/specs/2026-07-10-potential-demand-audit-design.md).
    """
    if source == "real":
        enriched = build_item_adjusted_demand(
            daily, discount_rows=discount_rows, alpha=closing_alpha
        )
        return enriched, "adjusted_demand"
    return daily, "potential_demand"


def _load_dataset(source: str, data_dir: Path | None) -> DailyDataset:
    if source == "parquet" and data_dir is None:
        data_dir = REPORTS_DIR
    return load_dataset(source=source, data_dir=data_dir)


def _enrich_if_needed(ds: DailyDataset, variants: list[str]) -> pd.DataFrame:
    """Return a daily frame enriched with calendar/weather/competitor as required by the variants."""
    needs_cal_weather = any(v in {"v1", "v2", "v3"} for v in variants)
    needs_external = any(v == "v3" for v in variants)
    if not needs_cal_weather:
        return ds.daily
    enriched = add_calendar_features(ds.daily, ds.calendar)
    enriched = add_weather_features(enriched, ds.weather)
    if needs_external:
        mapping = load_store_mapping()
        competitor_feats = compute_competitor_features(
            ds.competitor, mapping, pd.DatetimeIndex(sorted(enriched["date"].unique())),
        )
        enriched = add_competitor_features(enriched, competitor_feats)
        living_static = compute_store_living_features(ds.living_population, mapping)
        enriched = add_living_pop_features(enriched, living_static)
        pop_static = compute_store_population_features(ds.population, mapping)
        enriched = add_population_features(enriched, pop_static)
        cons_static = compute_store_consumption_features(ds.consumption, mapping)
        enriched = add_consumption_features(enriched, cons_static)
    return enriched


def _enrich_target(
    target: pd.DataFrame, ds: DailyDataset, *,
    forecast_weather: pd.DataFrame | None = None,
    include_external: bool = False,
) -> pd.DataFrame:
    """Merge calendar (future-safe), weather, and (optionally) competitor
    (forecast-safe via past-only license/close events) onto horizon dates."""
    target = add_calendar_features(target, ds.calendar)
    weather_frame = forecast_weather if forecast_weather is not None else ds.weather
    target = add_weather_features(target, weather_frame)
    if include_external:
        mapping = load_store_mapping()
        competitor_feats = compute_competitor_features(
            ds.competitor, mapping, pd.DatetimeIndex(sorted(target["date"].unique())),
        )
        target = add_competitor_features(target, competitor_feats)
        living_static = compute_store_living_features(ds.living_population, mapping)
        target = add_living_pop_features(target, living_static)
        pop_static = compute_store_population_features(ds.population, mapping)
        target = add_population_features(target, pop_static)
        cons_static = compute_store_consumption_features(ds.consumption, mapping)
        target = add_consumption_features(target, cons_static)
    return target


def _load_forecast_weather(horizon: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Long-form horizon weather frame keyed by (store_id, date), one row per
    (store, day) — each store's nx/ny/mid_reg from the store mapping is
    matched against the latest forecast parquet, falling back to recent
    observed averages when the forecast is missing.
    """
    short_p = EXTERNAL_DATA_DIR / "forecast_short_term_daily.parquet"
    mid_p = EXTERNAL_DATA_DIR / "forecast_mid_term_daily.parquet"
    observed_p = EXTERNAL_DATA_DIR / "weather_observed.parquet"
    if not short_p.exists() and not mid_p.exists():
        console.print(
            "[yellow]forecast[/] parquet 없음 — `bakery ingest-forecast` 먼저 실행. "
            "이번엔 fallback (최근 28일 평균)으로 horizon 채움."
        )
    mapping = load_store_mapping()
    return load_weather_forecast_from_local(
        short_daily_path=short_p,
        mid_daily_path=mid_p,
        observed_parquet_path=observed_p,
        mapping=mapping,
        horizon_start=horizon[0],
        horizon_end=horizon[-1],
    )


def _build_forecasters(variants: list[str], *, include_production: bool = False,
                       production_quantile: float = 0.85,
                       v23_target: str | None = None,
                       drop_groups: Collection[str] = ()):
    """Build baseline + LightGBM-per-variant list, optionally adding quantile
    production models for v2/v3 (lightgbm_v2_q85 etc.).

    v23_target: v2/v3의 학습 target을 명시(예: real→'adjusted_demand'). None이면
    모델 기본값(_default_target=potential_demand). v0/v1은 항상 sold_units.
    drop_groups: 실험용 feature 그룹 제거(LightGBM만; baseline엔 영향 없음).
    """
    forecasters = [SeasonalNaive(n_weeks=4), MovingAverage(window=28)]
    for v in variants:
        y = v23_target if (v23_target and v in {"v2", "v3"}) else None
        forecasters.append(GlobalLGBM(feature_set=v, y_col=y, drop_groups=drop_groups))
        if include_production and v in {"v2", "v3"}:
            prod_params = LGBMParams(objective="quantile", alpha=production_quantile)
            forecasters.append(
                GlobalLGBM(feature_set=v, params=prod_params, y_col=y, drop_groups=drop_groups)
            )
    return forecasters


def _model_to_feature_set(model: str) -> str | None:
    """For predict-next-week: map model name → feature_set; baselines return None."""
    if model == "lightgbm":
        return "v0"
    if model == "lightgbm_v1":
        return "v1"
    if model == "lightgbm_v2":
        return "v2"
    if model == "lightgbm_v3":
        return "v3"
    return None


def _pick_model(name: str):
    table = {
        "seasonal_naive": SeasonalNaive(n_weeks=4),
        "moving_average": MovingAverage(window=28),
        "lightgbm": GlobalLGBM(feature_set="v0"),
        "lightgbm_v1": GlobalLGBM(feature_set="v1"),
        "lightgbm_v2": GlobalLGBM(feature_set="v2"),
        "lightgbm_v3": GlobalLGBM(feature_set="v3"),
    }
    if name not in table:
        raise typer.BadParameter(f"unknown model {name!r}. choose from {list(table)}")
    return table[name]


def _print_summary(fold_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    summary = aggregate_by_model(fold_df)
    table = Table(title="Backtest summary (avg across folds)")
    cols = (
        ("model", "left"),
        ("wape_all", "right"),
        ("wape_no_stockout", "right"),
        ("mae", "right"),
        ("rmse", "right"),
        ("pct_under", "right"),
        ("pct_over", "right"),
        ("folds", "right"),
    )
    for col, justify in cols:
        table.add_column(col, justify=justify)
    for _, row in summary.iterrows():
        table.add_row(
            row["model"],
            f"{row['wape_all']:.4f}",
            f"{row['wape_no_stockout']:.4f}",
            f"{row['mae']:.2f}",
            f"{row['rmse']:.2f}",
            f"{row['pct_underpredict']:.2%}",
            f"{row['pct_overpredict']:.2%}",
            str(int(row["folds"])),
        )
    console.print(table)
    cat = per_category_wape(pred_df)
    cat_table = Table(title="Per-category WAPE")
    cat_table.add_column("model")
    cat_table.add_column("category", justify="left")
    cat_table.add_column("wape", justify="right")
    for _, row in cat.sort_values(["model", "wape"]).iterrows():
        cat_table.add_row(row["model"], str(row["category_id"]), f"{row['wape']:.4f}")
    console.print(cat_table)


def _print_next_week_preview(target: pd.DataFrame, *, demand_col: str = "yhat_sold_units") -> None:
    preview = (
        target.groupby(["store_id", "date"], as_index=False)[demand_col]
        .sum()
        .pivot(index="date", columns="store_id", values=demand_col)
        .round(0)
    )
    table = Table(title=f"Next-week total ({demand_col}) by store")
    table.add_column("date")
    for c in preview.columns:
        table.add_column(c, justify="right")
    for d, row in preview.iterrows():
        table.add_row(str(d.date()), *(str(int(v)) for v in row))
    console.print(table)


@app.command("stockout-risk")
def cmd_stockout_risk(
    source: str = "synthetic",
    data_dir: Path | None = None,
    n_splits: int = 4,
    horizon_days: int = 7,
    step_days: int = 7,
    feature_set: str = "v1",
    top_k: int = 50,
    out_dir: Path = REPORTS_DIR,
) -> None:
    """Train StockoutClassifier per fold; report AUC + precision/recall@k."""
    ds = _load_dataset(source, data_dir)
    daily = _enrich_if_needed(ds, [feature_set])
    windows = generate_time_splits(
        daily["date"], n_splits=n_splits, val_horizon_days=horizon_days, step_days=step_days
    )
    fold_rows: list[dict] = []
    pred_chunks: list[pd.DataFrame] = []
    for w in windows:
        train, val = apply_split(daily, w)
        if train.empty or val.empty:
            continue
        clf = StockoutClassifier(feature_set=feature_set).fit(train)
        proba = clf.predict_proba(val)
        y = val["is_stockout"].astype(int).to_numpy()
        s = proba.to_numpy()
        fold_rows.append(
            {
                "fold": w.fold_index,
                "auc": roc_auc(y, s),
                "base_rate": base_rate(y),
                f"precision@{top_k}": precision_at_k(y, s, top_k),
                f"recall@{top_k}": recall_at_k(y, s, top_k),
                "val_start": w.val_start,
                "val_end": w.val_end,
            }
        )
        pred_chunks.append(
            val[["store_id", "item_id", "category_id", "date", "is_stockout"]].assign(
                fold=w.fold_index, stockout_prob=s
            )
        )
    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "stockout_fold_results.csv", index=False)
    pred_df.to_csv(out_dir / "stockout_predictions.csv", index=False)
    _print_stockout_summary(fold_df)
    console.print(f"[green]wrote[/] {out_dir}/stockout_fold_results.csv, stockout_predictions.csv")


def _print_stockout_summary(fold_df: pd.DataFrame) -> None:
    table = Table(title="Stockout classifier — per fold")
    for col in fold_df.columns:
        table.add_column(col, justify="right" if col != "val_start" and col != "val_end" else "left")
    for _, row in fold_df.iterrows():
        cells = []
        for col in fold_df.columns:
            v = row[col]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        table.add_row(*cells)
    console.print(table)


@app.command("alpha-sweep")
def cmd_alpha_sweep(
    source: str = "real",
    n_splits: int = 4,
    horizon_days: int = 7,
    step_days: int = 7,
    variant: str = "v2",
    alphas: str = "0.50,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    margin_rate: float = 0.50,
    cost_rate: float = 0.30,
    lost_sale_multiplier: float = 1.7,
    item_master: Path = Path("data/internal/보나비 데이터_20260520.xlsx"),
    out_dir: Path = REPORTS_DIR,
    closing_alpha: float = DEFAULT_ALPHA,
) -> None:
    """Production-model quantile α sweep — net_profit-최대 α 찾기.

    α 0.50 (median) ~ 0.95 범위 backtest + 사업 KPI 비교. 매장·카테고리별 최적 α 결정.
    """
    import warnings
    import numpy as np

    warnings.filterwarnings("ignore")
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_params = CostParams(
        margin_rate=margin_rate, cost_rate=cost_rate, lost_sale_multiplier=lost_sale_multiplier
    )
    alpha_list = [float(a) for a in alphas.split(",") if a.strip()]

    items_xl = pd.read_excel(item_master, "품목정보")
    items_xl = items_xl[items_xl["상품구분"] == "SS"]
    unit_prices = dict(
        zip(items_xl["품목코드"].astype(str), pd.to_numeric(items_xl["판매단가"], errors="coerce").fillna(3000))
    )

    ds = _load_dataset(source, None)
    daily = _enrich_if_needed(ds, [variant])
    daily, demand_col = _resolve_demand_col(daily, source, closing_alpha)
    windows = generate_time_splits(
        daily["date"], n_splits=n_splits, val_horizon_days=horizon_days, step_days=step_days
    )

    forecasters = []
    for a in alpha_list:
        if a == 0.5:
            forecasters.append(GlobalLGBM(feature_set=variant, y_col=demand_col))  # median (regression)
        else:
            params = LGBMParams(objective="quantile", alpha=a)
            forecasters.append(GlobalLGBM(feature_set=variant, params=params, y_col=demand_col))

    console.print(
        f"[cyan]alpha-sweep[/] variant={variant} αs={alpha_list} "
        f"folds={len(windows)} horizon={horizon_days}d"
    )
    fold_df, pred_df = run_backtest(daily, forecasters, windows)

    # Inject demand column for true-demand-aware profit simulation
    if demand_col in daily.columns:
        d_lookup = daily.set_index(["store_id", "item_id", "date"])[demand_col]
        pred_df = pred_df.copy()
        pred_df[demand_col] = pred_df.set_index(
            ["store_id", "item_id", "date"]
        ).index.map(d_lookup)

    rows = []
    for model, sub in pred_df.groupby("model"):
        asym = asymmetric_loss(sub["yhat"], sub["sold_units"], params=cost_params)
        profit = simulate_profit(sub, unit_prices=unit_prices, params=cost_params,
                                 potential_col=demand_col)
        rows.append({
            "model": model,
            "asymmetric_loss": asym,
            "pct_under": (sub["yhat"] < sub["sold_units"]).mean(),
            "pct_over": (sub["yhat"] > sub["sold_units"]).mean(),
            "revenue_krw": float(profit["revenue_krw"].sum()),
            "waste_cost_krw": float(profit["waste_cost_krw"].sum()),
            "lost_margin_krw": float(profit["lost_margin_krw"].sum()),
            "net_profit_krw": float(profit["net_profit_krw"].sum()),
        })
    sweep_df = pd.DataFrame(rows).sort_values("net_profit_krw", ascending=False)
    table = Table(title=f"α sweep — {variant} model 사업 KPI 누계 (28일 backtest)")
    for c in ("model", "asymmetric_loss", "pct_under", "pct_over",
              "revenue_krw", "waste_cost_krw", "lost_margin_krw", "net_profit_krw"):
        table.add_column(c, justify="left" if c == "model" else "right")
    for _, r in sweep_df.iterrows():
        table.add_row(
            r["model"], f"{r['asymmetric_loss']:.4f}",
            f"{r['pct_under']:.2%}", f"{r['pct_over']:.2%}",
            f"{r['revenue_krw']/1e6:.2f}M", f"{r['waste_cost_krw']/1e6:.2f}M",
            f"{r['lost_margin_krw']/1e6:.2f}M", f"{r['net_profit_krw']/1e6:.2f}M",
        )
    console.print(table)

    best = sweep_df.iloc[0]
    console.print(
        f"\n[green]최적 모델[/] {best['model']} → net_profit {best['net_profit_krw']/1e6:.2f}M원"
    )

    sweep_df.to_csv(out_dir / "alpha_sweep.csv", index=False)
    console.print(f"[green]wrote[/] {out_dir}/alpha_sweep.csv")


@app.command("business-report")
def cmd_business_report(
    source: str = "real",
    data_dir: Path | None = None,
    n_splits: int = 4,
    horizon_days: int = 7,
    step_days: int = 7,
    variants: str = "v0,v1,v2,v3",
    production_quantile: float = 0.85,
    margin_rate: float = 0.50,
    cost_rate: float = 0.30,
    lost_sale_multiplier: float = 1.7,
    item_master: Path = Path("data/internal/보나비 데이터_20260520.xlsx"),
    out_dir: Path = REPORTS_DIR,
    closing_alpha: float = DEFAULT_ALPHA,
) -> None:
    """v0 / v2 / v3 도입 시 광교 매장 예상 사업 임팩트 종합 리포트.

    1) Self-fulfilling stockout 패턴 (광교 top 품목)
    2) 영구 손실 시뮬레이션 (24개월 누계 KRW)
    3) Production model backtest (v2/v3 quantile α 포함)
    4) 사업 KPI (asymmetric loss + profit simulation)
    """
    import warnings

    import numpy as np

    from .analysis.self_fulfillment import (
        estimated_lost_demand,
        top_self_fulfilling_items,
    )
    from .analysis.substitution import (
        adjust_lost_units,
        compute_substitution_matrix,
    )
    from .evaluation.business_metrics import (
        CostParams,
        aggregate_profit,
        asymmetric_loss,
    )

    warnings.filterwarnings("ignore")
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_params = CostParams(
        margin_rate=margin_rate,
        cost_rate=cost_rate,
        lost_sale_multiplier=lost_sale_multiplier,
    )

    # Item master + unit prices
    items_xl = pd.read_excel(item_master, "품목정보")
    items_xl = items_xl[items_xl["상품구분"] == "SS"]
    unit_prices = dict(
        zip(items_xl["품목코드"].astype(str), pd.to_numeric(items_xl["판매단가"], errors="coerce").fillna(3000))
    )
    item_names = dict(zip(items_xl["품목코드"].astype(str), items_xl["POS메뉴명"]))

    # Data load
    variant_list = _parse_variants(variants)
    ds = _load_dataset(source, data_dir)
    daily = _enrich_if_needed(ds, variant_list)
    daily, demand_col = _resolve_demand_col(daily, source, closing_alpha)

    console.print("\n[bold cyan]━━━ 사업 임팩트 리포트 ━━━[/]\n")
    console.print(
        f"매장: store_gw01 (아티제 아브뉴프랑광교점) | "
        f"기간: {daily['date'].min().date()} ~ {daily['date'].max().date()} | "
        f"품목: {daily['item_id'].nunique()}개"
    )
    console.print(
        f"가정: 마진율 {margin_rate:.0%}, 원가율 {cost_rate:.0%}, 품절 비용 multiplier {lost_sale_multiplier}×\n"
    )

    # ─── A. Self-fulfilling pattern ───
    console.print("[bold]1. Self-fulfilling stockout 패턴[/]")
    top_self = top_self_fulfilling_items(daily, n=10)
    top_self["item_name"] = top_self["item_id"].map(item_names)
    self_table = Table(title="Top 10 self-fulfilling 품목 (매주 일관 품절 + 낮은 CV)")
    for col in ("item_name", "sold_total", "avg_stockout_rate", "avg_sold_cv", "avg_stockout_hour", "covered_dows"):
        self_table.add_column(col, justify="right" if col != "item_name" else "left")
    for _, r in top_self.iterrows():
        self_table.add_row(
            str(r["item_name"]),
            f"{int(r['sold_total']):,}",
            f"{r['avg_stockout_rate']:.1%}",
            f"{r['avg_sold_cv']:.2f}",
            f"{r['avg_stockout_hour']:.1f}시",
            str(int(r["covered_dows"])),
        )
    console.print(self_table)

    # ─── B. Permanent lost-revenue simulation ───
    console.print("\n[bold]2. 영구 손실 시뮬레이션 (v0 운영 가정)[/]")
    lost = estimated_lost_demand(daily)
    lost["item_id"] = lost["item_id"].astype(str)
    lost["unit_price"] = lost["item_id"].map(unit_prices).fillna(3000)
    lost["actual_revenue"] = lost["sold_units"] * lost["unit_price"]
    lost["lost_revenue"] = lost["lost_units"] * lost["unit_price"]
    lost["lost_margin"] = lost["lost_revenue"] * margin_rate * lost_sale_multiplier

    # Substitution-adjusted: subtract intra-category outflow estimated from receipts
    receipts_path = Path("data/internal/bonavi_receipts.parquet")
    sub_matrix = None
    if receipts_path.exists():
        try:
            receipts_df = pd.read_parquet(receipts_path)
            receipts_df = receipts_df[receipts_df["date"] >= daily["date"].min()]
            sub_matrix = compute_substitution_matrix(daily, receipts_df)
            lost_adj = adjust_lost_units(lost, sub_matrix.outflow_ratio)
            lost["lost_units_adjusted"] = lost_adj["lost_units_adjusted"]
            lost["lost_revenue_adjusted"] = lost["lost_units_adjusted"] * lost["unit_price"]
            lost["lost_margin_adjusted"] = (
                lost["lost_revenue_adjusted"] * margin_rate * lost_sale_multiplier
            )
        except Exception as exc:
            console.print(f"[yellow]substitution matrix 계산 실패 — receipts 없거나 오류: {exc}[/]")
            sub_matrix = None
    months = (daily["date"].max() - daily["date"].min()).days / 30
    summary_b = Table(title=f"매장 24개월 환산 추정 (실제 데이터 {months:.0f}개월)")
    summary_b.add_column("항목"); summary_b.add_column("KRW", justify="right")
    actual_rev = lost["actual_revenue"].sum() / months * 24
    lost_rev = lost["lost_revenue"].sum() / months * 24
    lost_marg = lost["lost_margin"].sum() / months * 24
    summary_b.add_row("실제 매출 (24개월 환산)", f"{actual_rev/1e8:.2f}억원")
    summary_b.add_row("실제 마진", f"{actual_rev*margin_rate/1e8:.2f}억원")
    summary_b.add_row("[red]잃은 매출 (independent 보정)[/]", f"[red]{lost_rev/1e8:.2f}억원[/]")
    summary_b.add_row("[red]잃은 마진 (cross-sell + 평판)[/]", f"[red]{lost_marg/1e8:.2f}억원[/]")
    summary_b.add_row("잠재 회수율 (보정 전)", f"{lost_rev / actual_rev:.1%}")
    if sub_matrix is not None:
        lost_rev_adj = lost["lost_revenue_adjusted"].sum() / months * 24
        lost_marg_adj = lost["lost_margin_adjusted"].sum() / months * 24
        avg_outflow = float(sub_matrix.outflow_ratio.mean())
        summary_b.add_row(
            "[yellow]Substitution 평균 outflow[/]", f"[yellow]{avg_outflow:.1%}[/]"
        )
        summary_b.add_row(
            "[green]잃은 매출 (substitution-adjusted)[/]",
            f"[green]{lost_rev_adj/1e8:.2f}억원[/]",
        )
        summary_b.add_row(
            "[green]잃은 마진 (substitution-adjusted)[/]",
            f"[green]{lost_marg_adj/1e8:.2f}억원[/]",
        )
    console.print(summary_b)

    # Top items by lost revenue
    top_lost = (
        lost.groupby("item_id", as_index=False)
        .agg(lost_units=("lost_units", "sum"), lost_revenue=("lost_revenue", "sum"), unit_price=("unit_price", "first"))
        .sort_values("lost_revenue", ascending=False)
        .head(10)
    )
    top_lost["item_name"] = top_lost["item_id"].map(item_names)
    lost_table = Table(title="Top 10 잃은 매출 품목 (전 기간 누계)")
    for col in ("item_name", "lost_units", "unit_price", "lost_revenue"):
        lost_table.add_column(col, justify="right" if col != "item_name" else "left")
    for _, r in top_lost.iterrows():
        lost_table.add_row(
            str(r["item_name"]),
            f"{int(r['lost_units']):,}개",
            f"{int(r['unit_price']):,}원",
            f"{r['lost_revenue']/1e6:.1f}M원",
        )
    console.print(lost_table)

    # ─── C. Production-model backtest ───
    console.print("\n[bold]3. Production model backtest (demand + quantile α)[/]")
    windows = generate_time_splits(
        daily["date"], n_splits=n_splits, val_horizon_days=horizon_days, step_days=step_days
    )
    forecasters = _build_forecasters(
        variant_list, include_production=True, production_quantile=production_quantile,
        v23_target=demand_col,
    )
    fold_df, pred_df = run_backtest(daily, forecasters, windows)
    summary = aggregate_by_model(fold_df).sort_values("wape_all")
    bt_table = Table(title="모델별 backtest 요약 (production model 포함)")
    cols = (("model", "left"), ("wape_all", "right"), ("pct_underpredict", "right"),
            ("pct_overpredict", "right"), ("mae", "right"))
    for c, j in cols: bt_table.add_column(c, justify=j)
    for _, r in summary.iterrows():
        bt_table.add_row(
            r["model"], f"{r['wape_all']:.4f}", f"{r['pct_underpredict']:.2%}",
            f"{r['pct_overpredict']:.2%}", f"{r['mae']:.2f}",
        )
    console.print(bt_table)

    # ─── D. Business KPI per model ───
    console.print("\n[bold]4. 모델별 사업 KPI (asymmetric loss + profit simulation)[/]")
    # Inject demand column into pred_df from the enriched daily so simulate_profit
    # can see the censoring-corrected target.
    if demand_col in daily.columns:
        d_lookup = daily.set_index(["store_id", "item_id", "date"])[demand_col]
        pred_df = pred_df.copy()
        pred_df[demand_col] = pred_df.set_index(
            ["store_id", "item_id", "date"]
        ).index.map(d_lookup)
    biz_rows = []
    for model, sub in pred_df.groupby("model"):
        asym = asymmetric_loss(sub["yhat"], sub["sold_units"], params=cost_params)
        profit = simulate_profit(sub, unit_prices=unit_prices, params=cost_params,
                                 potential_col=demand_col)
        biz_rows.append({
            "model": model, "asymmetric_loss": asym,
            "revenue_krw": float(profit["revenue_krw"].sum()),
            "waste_cost_krw": float(profit["waste_cost_krw"].sum()),
            "lost_margin_krw": float(profit["lost_margin_krw"].sum()),
            "net_profit_krw": float(profit["net_profit_krw"].sum()),
        })
    biz_df = pd.DataFrame(biz_rows).sort_values("net_profit_krw", ascending=False)
    biz_table = Table(title="Backtest fold 누계 — 모델별 사업 KPI")
    biz_cols = (("model", "left"), ("asymmetric_loss", "right"),
                ("revenue_krw", "right"), ("waste_cost_krw", "right"),
                ("lost_margin_krw", "right"), ("net_profit_krw", "right"))
    for c, j in biz_cols: biz_table.add_column(c, justify=j)
    for _, r in biz_df.iterrows():
        biz_table.add_row(
            r["model"],
            f"{r['asymmetric_loss']:.4f}",
            f"{r['revenue_krw']/1e6:.2f}M",
            f"{r['waste_cost_krw']/1e6:.2f}M",
            f"{r['lost_margin_krw']/1e6:.2f}M",
            f"{r['net_profit_krw']/1e6:.2f}M",
        )
    console.print(biz_table)

    # Persist
    biz_df.to_csv(out_dir / "business_report_kpi.csv", index=False)
    fold_df.to_csv(out_dir / "business_report_folds.csv", index=False)
    top_self.to_csv(out_dir / "business_report_self_fulfilling.csv", index=False)
    top_lost.to_csv(out_dir / "business_report_top_lost.csv", index=False)
    console.print(
        f"\n[green]wrote[/] {out_dir}/business_report_*.csv (kpi/folds/self_fulfilling/top_lost)"
    )


@app.command("mnl-substitution")
def cmd_mnl_substitution(
    source: str = "real",
    receipts: Path = Path("data/internal/bonavi_receipts.parquet"),
    item_master: Path = Path("data/internal/보나비 데이터_20260520.xlsx"),
    out_dir: Path = Path("reports"),
) -> None:
    """Multinomial Logit choice model — receipt-level substitution matrix.

    Compares MNL (theory-grounded, receipt-microscopic) against the daily-RD
    substitution module on the same data. Outputs:
      - mnl_utilities.csv         (per-item α, by category)
      - mnl_substitution.csv      (per-pair s_share, s_raw)
      - mnl_vs_rd_top_pairs.csv   (side-by-side ranking)
    """
    from .analysis.mnl_substitution import fit_mnl_per_category
    from .analysis.substitution import compute_substitution_matrix

    out_dir.mkdir(parents=True, exist_ok=True)

    if not receipts.exists():
        console.print(f"[red]{receipts} 없음 — receipts parquet 필요[/]")
        raise typer.Exit(code=1)

    receipts_df = pd.read_parquet(receipts)
    receipts_df["date"] = pd.to_datetime(receipts_df["date"])

    ds = _load_dataset(source, None)
    daily = ds.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    item_names: dict[str, str] = {}
    if item_master.exists():
        items_xl = pd.read_excel(item_master, "품목정보")
        items_xl = items_xl[items_xl["상품구분"] == "SS"]
        item_names = dict(zip(items_xl["품목코드"].astype(str), items_xl["POS메뉴명"]))

    console.print("[cyan]MNL fit ...[/]")
    mnl = fit_mnl_per_category(receipts_df, daily)
    console.print(f"  utilities: {len(mnl.utilities)} items in {mnl.utilities['category_id'].nunique()} categories")

    console.print("[cyan]RD substitution (baseline) ...[/]")
    rd = compute_substitution_matrix(daily, receipts_df, include_inter_category=False)

    # Top pairs comparison
    mnl_top = mnl.substitution.nlargest(20, "s_share").copy()
    mnl_top["from_name"] = mnl_top["from_item"].map(item_names)
    mnl_top["to_name"] = mnl_top["to_item"].map(item_names)
    rd_top = rd.coefficients[rd.coefficients["same_category"]].nlargest(20, "sub_rate").copy()
    rd_top["from_name"] = rd_top["from_item"].map(item_names)
    rd_top["to_name"] = rd_top["to_item"].map(item_names)

    mnl_table = Table(title="MNL — top 15 substitution pairs (by s_share)")
    for col in ("category_id", "from_name", "to_name", "s_share", "s_raw"):
        mnl_table.add_column(col, justify="right" if col not in ("from_name", "to_name", "category_id") else "left")
    for _, r in mnl_top.head(15).iterrows():
        mnl_table.add_row(
            r["category_id"], str(r["from_name"]), str(r["to_name"]),
            f"{r['s_share']:.3f}", f"{r['s_raw']:.3f}",
        )
    console.print(mnl_table)

    rd_table = Table(title="RD — top 15 substitution pairs (by sub_rate, within-category)")
    for col in ("category_id", "from_name", "to_name", "sub_rate", "beta_rd", "co_occ"):
        rd_table.add_column(col, justify="right" if col not in ("from_name", "to_name", "category_id") else "left")
    for _, r in rd_top.head(15).iterrows():
        rd_table.add_row(
            r["category_id"], str(r["from_name"]), str(r["to_name"]),
            f"{r['sub_rate']:.3f}", f"{r['beta_rd']:.3f}", f"{r['co_occ']:.3f}",
        )
    console.print(rd_table)

    # Rank agreement: Spearman of overlapping (from, to) pairs
    merged = mnl.substitution.merge(
        rd.coefficients[["from_item", "to_item", "sub_rate", "beta_rd", "co_occ"]],
        on=["from_item", "to_item"], how="inner",
    )
    if len(merged) > 10:
        rho = merged[["s_share", "sub_rate"]].corr(method="spearman").iat[0, 1]
        console.print(f"[bold]MNL ↔ RD rank correlation (Spearman):[/] {rho:.3f}  (over {len(merged)} pairs)")

    mnl.utilities.to_csv(out_dir / "mnl_utilities.csv", index=False)
    mnl.substitution.to_csv(out_dir / "mnl_substitution.csv", index=False)
    merged.to_csv(out_dir / "mnl_vs_rd_pairs.csv", index=False)
    pd.DataFrame({"item_id": mnl.outflow_ratio.index, "outflow_ratio": mnl.outflow_ratio.values}).to_csv(
        out_dir / "mnl_outflow.csv", index=False,
    )
    console.print(f"[green]wrote[/] {out_dir}/mnl_*.csv (utilities/substitution/vs_rd/outflow)")


@app.command("nested-logit")
def cmd_nested_logit(
    source: str = "real",
    receipts: Path = Path("data/internal/bonavi_receipts.parquet"),
    item_master: Path = Path("data/internal/보나비 데이터_20260520.xlsx"),
    out_dir: Path = Path("reports"),
) -> None:
    """Nested logit (cross-category) — relaxes IIA via per-nest λ_g.

    Outputs:
      - nested_utilities.csv (α_i with category)
      - nested_lambdas.csv   (λ_g per nest)
      - nested_substitution.csv (within + cross-nest pairs with same_nest flag)
    """
    from .analysis.nested_logit import fit_nested_logit
    from .analysis.mnl_substitution import fit_mnl_per_category

    out_dir.mkdir(parents=True, exist_ok=True)
    if not receipts.exists():
        console.print(f"[red]{receipts} 없음 — receipts parquet 필요[/]")
        raise typer.Exit(code=1)

    receipts_df = pd.read_parquet(receipts)
    receipts_df["date"] = pd.to_datetime(receipts_df["date"])
    ds = _load_dataset(source, None)
    daily = ds.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    item_names: dict[str, str] = {}
    if item_master.exists():
        items_xl = pd.read_excel(item_master, "품목정보")
        items_xl = items_xl[items_xl["상품구분"] == "SS"]
        item_names = dict(zip(items_xl["품목코드"].astype(str), items_xl["POS메뉴명"]))

    console.print("[cyan]nested logit fit ...[/]")
    nl = fit_nested_logit(receipts_df, daily)
    console.print(f"  utilities: {len(nl.utilities)} items, λ per nest:")

    lam_table = Table(title="λ_g per nest (closer to 0 = stronger within-nest substitution)")
    lam_table.add_column("nest")
    lam_table.add_column("λ_g", justify="right")
    for cat, lam in nl.lambdas.items():
        lam_table.add_row(cat, f"{lam:.3f}")
    console.print(lam_table)

    within = nl.substitution[nl.substitution["same_nest"]]
    cross = nl.substitution[~nl.substitution["same_nest"]]
    ratio_table = Table(title="Within-nest vs cross-nest substitution")
    ratio_table.add_column("kind"); ratio_table.add_column("pairs", justify="right")
    ratio_table.add_column("mean s_share", justify="right"); ratio_table.add_column("median", justify="right")
    ratio_table.add_row("within-nest", str(len(within)), f"{within['s_share'].mean():.4f}", f"{within['s_share'].median():.4f}")
    ratio_table.add_row("cross-nest", str(len(cross)), f"{cross['s_share'].mean():.4f}", f"{cross['s_share'].median():.4f}")
    if cross['s_share'].mean() > 0:
        ratio_table.add_row("ratio (within/cross)", "—", f"{within['s_share'].mean()/cross['s_share'].mean():.2f}×", "—")
    console.print(ratio_table)

    # Top pairs side by side
    top_within = within.nlargest(10, "s_share").copy()
    top_within["from_name"] = top_within["from_item"].map(item_names)
    top_within["to_name"] = top_within["to_item"].map(item_names)
    win_tbl = Table(title="Top within-nest substitution pairs (s_share)")
    for col in ("from_cat", "from_name", "to_name", "s_share"):
        win_tbl.add_column(col, justify="right" if col == "s_share" else "left")
    for _, r in top_within.iterrows():
        win_tbl.add_row(r["from_cat"], str(r["from_name"]), str(r["to_name"]), f"{r['s_share']:.3f}")
    console.print(win_tbl)

    # Compare vs MNL
    console.print("\n[cyan]MNL (per-category, IIA) for comparison ...[/]")
    mnl = fit_mnl_per_category(receipts_df, daily)
    overlap = nl.substitution.merge(
        mnl.substitution[["from_item", "to_item", "s_share"]].rename(columns={"s_share": "mnl_s_share"}),
        on=["from_item", "to_item"], how="inner",
    )
    if len(overlap) > 10:
        rho = overlap[["s_share", "mnl_s_share"]].corr(method="spearman").iat[0, 1]
        console.print(f"[bold]Nested ↔ MNL within-nest pair rank correlation (Spearman):[/] {rho:.3f}  (over {len(overlap)} pairs)")

    nl.utilities.to_csv(out_dir / "nested_utilities.csv", index=False)
    nl.lambdas.to_csv(out_dir / "nested_lambdas.csv")
    nl.substitution.to_csv(out_dir / "nested_substitution.csv", index=False)
    overlap.to_csv(out_dir / "nested_vs_mnl_pairs.csv", index=False)
    console.print(f"[green]wrote[/] {out_dir}/nested_*.csv (utilities/lambdas/substitution/vs_mnl)")


@app.command("ingest-calendar")
def cmd_ingest_calendar(
    start_year: int = 2024,
    end_year: int = 2026,
) -> None:
    """천문연 특일정보 API에서 공휴일 + 24절기 backfill → data/external/calendar_raw.parquet."""
    console.print(f"[cyan]calendar[/] 공휴일 + 24절기 {start_year}~{end_year} backfill 시작")
    out = calendar_api.backfill(start_year, end_year)
    df = pd.read_parquet(out)
    console.print(f"[green]wrote[/] {out} ({len(df):,} rows)")
    console.print(df.head(6).to_string(index=False))


@app.command("ingest-weather")
def cmd_ingest_weather(
    start: str = "2024-01-01",
    end: str | None = None,
) -> None:
    """기상청 ASOS 일자료 API backfill → data/external/weather_observed.parquet."""
    end_date = Date.fromisoformat(end) if end else Date.today()
    start_date = Date.fromisoformat(start)
    console.print(f"[cyan]weather[/] ASOS 일자료 {start_date} ~ {end_date} backfill 시작")
    out = weather_api.backfill(start_date, end_date)
    df = pd.read_parquet(out)
    console.print(f"[green]wrote[/] {out} ({len(df):,} rows, {df['station_id'].nunique()} stations)")


@app.command("ingest-living-population")
def cmd_ingest_living_population(
    start: str | None = None,
    end: str | None = None,
) -> None:
    """서울 열린데이터광장 SPOP_LOCAL_RESD_DONG backfill → data/external/living_population.parquet.

    Open Data Plaza retains only the last ~2 months via OpenAPI. Default window
    is last 30 days from today. Older history requires monthly CSV zip download.
    """
    today = Date.today()
    end_date = Date.fromisoformat(end) if end else today
    start_date = Date.fromisoformat(start) if start else (today - pd.Timedelta(days=30).to_pytimedelta())
    console.print(f"[cyan]living-pop[/] SPOP_LOCAL_RESD_DONG {start_date} ~ {end_date} backfill 시작")
    out = living_population_api.backfill(start_date, end_date)
    df = pd.read_parquet(out)
    console.print(
        f"[green]wrote[/] {out} ({len(df):,} rows, "
        f"{df['admin_dong_code'].nunique()} dongs, {df['date'].nunique()} days)"
    )


@app.command("ingest-population")
def cmd_ingest_population() -> None:
    """행안부 admmSexdAgePpltn (행정동 성/연령별 인구) → data/external/population.parquet.

    Snapshot은 월 단위. statsYm은 API 응답에 포함되어 자동 추출. 활용신청한
    DATA_GO_KR_API_KEY 그대로 사용 (admmSexdAgePpltn 활용신청 필요).
    """
    console.print("[cyan]population[/] admmSexdAgePpltn 전체 백필 시작")
    out = population_api.backfill()
    df = pd.read_parquet(out)
    console.print(
        f"[green]wrote[/] {out} ({len(df):,} rows, "
        f"{df['admin_dong_code'].nunique()} dongs)"
    )


@app.command("ingest-consumption")
def cmd_ingest_consumption() -> None:
    """서울 VwsmAdstrdNcmCnsmpW (상권분석 소비-행정동) 전체 분기 백필 → data/external/consumption.parquet."""
    console.print("[cyan]consumption[/] VwsmAdstrdNcmCnsmpW 전체 백필 시작")
    out = consumption_api.backfill()
    df = pd.read_parquet(out)
    console.print(
        f"[green]wrote[/] {out} ({len(df):,} rows, "
        f"{df['admin_dong_code'].nunique()} dongs, {df['quarter'].nunique()} quarters)"
    )


@app.command("format-bonavi")
def cmd_format_bonavi(
    xlsx_path: Path = Path("data/internal/보나비 데이터_20260520.xlsx"),
    store_code: str = "1000000047",
    rename_store_id: str = "store_gw01",
    out_path: Path = Path("data/internal/bonavi_daily.parquet"),
) -> None:
    """보나비 xlsx → DAILY_COLUMNS parquet 변환. 광교점 1매장 단품·정상매출만."""
    from .data.bonavi_loader import build

    console.print(f"[cyan]bonavi[/] {xlsx_path} → {out_path} (store={store_code} → {rename_store_id})")
    out = build(xlsx_path=xlsx_path, store_code=store_code, rename_store_id=rename_store_id, out_path=out_path)
    df = pd.read_parquet(out)
    console.print(
        f"[green]wrote[/] {out} ({len(df):,} rows, "
        f"{df['item_id'].nunique()} items, "
        f"{df['category_id'].nunique()} categories, "
        f"{df['date'].nunique()} days, "
        f"stockout {df['is_stockout'].sum():,})"
    )


@app.command("ingest-living-pop-csv")
def cmd_ingest_living_pop_csv() -> None:
    """data/external/living_pop_zips/*.zip(LOCAL_PEOPLE_DONG history) → living_population.parquet.

    OpenAPI는 최근 2개월만 retention이라 학습 윈도우 cover를 위해 월별 zip 다운로드.
    매장 dong만 필터링해 적재 + 기존 parquet과 merge dedup.
    """
    console.print("[cyan]living-pop CSV[/] data/external/living_pop_zips/ 안 zip 일괄 처리")
    out = living_population_csv.ingest_zip_dir()
    df = pd.read_parquet(out)
    months = df['date'].dt.to_period('M').nunique()
    console.print(
        f"[green]wrote[/] {out} ({len(df):,} rows, "
        f"{df['admin_dong_code'].nunique()} dongs, "
        f"{df['date'].nunique()} days, {months} months)"
    )


@app.command("ingest-competitor")
def cmd_ingest_competitor() -> None:
    """소상공인진흥공단 storeListInRadius (반경 1km 빵/도넛 + 카페) → data/external/competitor_raw.parquet.

    PoC 한계: SBIZ 데이터는 현재 영업 중인 점포만 — license_date/close_date 정보 없어
    신규/폐업 90일 trend features는 0으로 채워짐.
    """
    console.print("[cyan]competitor[/] storeListInRadius 매장별 1km 백필 시작")
    out = competitor_api.backfill()
    df = pd.read_parquet(out)
    by_cat = df.groupby("category").size().to_dict()
    console.print(f"[green]wrote[/] {out} ({len(df):,} rows, {by_cat})")


@app.command("ingest-forecast")
def cmd_ingest_forecast() -> None:
    """기상청 단기/중기예보 최신 발표 → data/external/forecast_*.parquet.

    단기예보: 매일 8회 발표(02·05·08·11·14·17·20·23). horizon D+1~D+3 커버.
    중기예보: 매일 2회 발표(06·18). horizon D+4~D+10 커버.
    """
    console.print("[cyan]forecast[/] 최신 단기 + 중기예보 ingestion")
    paths = forecast_api.backfill_forecast()
    for kind, p in paths.items():
        df = pd.read_parquet(p)
        console.print(f"[green]wrote[/] {kind}: {p} ({len(df):,} rows)")
    if not paths:
        console.print("[yellow]warning[/] no forecast rows returned — check API status / region codes")


@app.command("grounding-eval")
def cmd_grounding_eval(
    provider: str = "auto",
    model: str = "gpt-5-mini",
    source: str = "synthetic",
) -> None:
    """v7 그라운딩 eval: grounded arm vs RAG-only arm delta 측정.

    두 arm 모두 동일한 모델(provider/model)을 사용해 공정하게 비교한다.
    --source synthetic 이면 시연용 (실데이터 없이 실행 가능).
    """
    from .ontology.grounding.run import run_eval

    report = run_eval(provider=provider, model=model, source=source)
    console.print(f"[bold]grounded_accuracy[/] {report.grounded_accuracy:.3f}")
    console.print(f"[bold]rag_accuracy[/]      {report.rag_accuracy:.3f}")
    console.print(f"[bold]delta[/]             {report.delta:+.3f}")
    console.print(
        "[yellow]synthetic 시연 (measured on synthetic, not real data)[/]"
        if source == "synthetic"
        else f"[cyan]source={source}[/]"
    )




def _select_gate_policy(gate: str):
    from .ontology.loop import auto_approve, approve_as_proposed
    if gate == "auto":
        return auto_approve
    if gate == "human":
        return approve_as_proposed
    raise ValueError(f"unknown gate: {gate} (auto|human)")


def _parse_period(period: str, now: str) -> tuple[str, str, str]:
    """Split "start,end" and resolve the commit timestamp (default = start 09:00)."""
    start, end = (s.strip() for s in period.split(","))
    stamp = now or f"{start}T09:00:00"
    return start, end, stamp


def _write_and_label(wb: WritebackStore, out: str, source: str) -> None:
    """Optionally persist the writeback store, then print the source/demo label."""
    if out:
        wb.to_parquet(out)
        console.print(f"[green]wrote[/] {out} ({len(wb.records)} records)")
    console.print(
        "[yellow]synthetic 메커니즘 시연 (mechanism demo, not accuracy)[/]"
        if source == "synthetic" else f"[cyan]source={source}[/]")


def _lever_warning(before_demand: float) -> str | None:
    """Warn when the baseline re-forecast collapsed to 0 (unresolved lag features):
    the driver lever then looks inert (before≈after≈0) though nothing actually moved."""
    if before_demand <= 0:
        return ("[yellow]⚠ before_demand=0 — 재예측 붕괴(unresolved lag). "
                "레버가 무효처럼 보일 수 있음; 기간/품목 데이터 커버리지 확인.[/]")
    return None


@app.command("closed-loop")
def cmd_closed_loop(
    store: str,
    period: str,                        # "YYYY-MM-DD,YYYY-MM-DD"
    gate: str = "human",               # auto(frontier) | human(rubber-stamp)
    source: str = "synthetic",
    provider: str = "auto",
    model: str = "gpt-5-mini",
    now: str = "",                      # ISO; 비면 period start의 09:00
    out: str = "",                      # parquet 경로(옵션)
) -> None:
    """v7 하류 closed-loop: grounded 추천 → 사람 게이트 → writeback commit.

    추천은 read 도구만 쓰는 grounded 에이전트가 한다(쓰기는 게이트 통과 후 결정론 코드).
    --source synthetic 이면 시연용. closed-loop은 메커니즘 시연이며 정확도 주장이 아니다.
    """
    from .data.loader import load_dataset
    from .ontology.grounding.llm import make_llm_client
    from .ontology.loop import run_closed_loop
    from .ontology.writeback import WritebackStore

    start, end, stamp = _parse_period(period, now)
    gate_policy = _select_gate_policy(gate)
    client = make_llm_client(provider, model)
    dataset = load_dataset(source)
    wb = WritebackStore(require_approval=True)
    recs = run_closed_loop(client, dataset, store, (start, end), wb, gate_policy, now=stamp)

    console.print(f"[bold]closed-loop[/] store={store} period={start}~{end} gate={gate}")
    for r in recs:
        console.print(f"  {r.item_id}: proposed={r.proposed_qty} → "
                      f"{r.status} qty={r.approved_qty} by={r.approver}")
    if not recs:
        console.print("[yellow]no valid proposals[/]")
    _write_and_label(wb, out, source)


def _parse_drivers(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad driver spec: {part!r} (expected key=value)")
        key, val = part.split("=", 1)
        out[key.strip()] = float(val.strip())
    if not out:
        raise ValueError("no drivers parsed; expected e.g. 'is_rain=1,is_snow=0'")
    return out


@app.command("scenario-commit")
def cmd_scenario_commit(
    store: str,
    item: str,
    period: str,                        # "YYYY-MM-DD,YYYY-MM-DD"
    drivers: str,                       # "is_rain=1,is_snow=0"
    gate: str = "human",               # auto(frontier) | human(rubber-stamp)
    source: str = "synthetic",
    now: str = "",                      # ISO; 비면 period start의 09:00
    out: str = "",                      # parquet 경로(옵션)
) -> None:
    """v7 Scenario→commit: 가상 드라이버 시나리오 하 조정 발주량을 사람 게이트 통과해 확정.

    상류 what_if_driver(재예측) + 하류 writeback(게이트)를 잇는 결정론 closed-loop.
    --source synthetic 이면 시연용. 정확도 주장이 아니라 메커니즘 시연.
    """
    from .data.loader import load_dataset
    from .ontology.loop import run_scenario_commit
    from .ontology.writeback import WritebackStore

    start, end, stamp = _parse_period(period, now)
    gate_policy = _select_gate_policy(gate)
    driver_overrides = _parse_drivers(drivers)
    dataset = load_dataset(source)
    wb = WritebackStore(require_approval=True)
    res = run_scenario_commit(dataset, store, item, (start, end), driver_overrides,
                              wb, gate_policy, now=stamp, train_cutoff=start)

    w = res.whatif
    console.print(f"[bold]scenario-commit[/] store={store} item={item} drivers={driver_overrides}")
    console.print(f"  demand {w.before_demand:.1f} → {w.after_demand:.1f} (Δ{w.demand_delta:+.1f})"
                  + ("  [yellow]out-of-support[/]" if w.out_of_support else ""))
    warn = _lever_warning(w.before_demand)
    if warn:
        console.print(warn)
    console.print(f"  order {res.base_order:.0f} → {res.committed.proposed_qty:.0f}  "
                  f"{res.committed.status} qty={res.committed.approved_qty} by={res.committed.approver}")
    _write_and_label(wb, out, source)



@app.command("scenario-commit-batch")
def cmd_scenario_commit_batch(
    store: str,
    period: str,                        # "YYYY-MM-DD,YYYY-MM-DD"
    drivers: str,                       # "is_rain=1,is_snow=0"
    items: str = "",                    # "a,b,c"; 비면 매장 전 품목
    gate: str = "human",               # auto(frontier) | human(rubber-stamp)
    source: str = "synthetic",
    now: str = "",                      # ISO; 비면 period start의 09:00
    out: str = "",                      # parquet 경로(옵션)
) -> None:
    """v7 다품목 Scenario→commit: 여러 품목에 같은 드라이버 시나리오를 배치 커밋.

    모델 fit 1회 공유. --items 생략 시 해당 매장 전 품목. 결정론(LLM 미개입).
    --source synthetic 이면 메커니즘 시연.
    """
    from .data.loader import load_dataset
    from .ontology.loop import run_scenario_commit_batch
    from .ontology.writeback import WritebackStore

    start, end, stamp = _parse_period(period, now)
    gate_policy = _select_gate_policy(gate)
    driver_overrides = _parse_drivers(drivers)
    dataset = load_dataset(source)
    item_ids = ([s.strip() for s in items.split(",")] if items
                else sorted(dataset.daily.loc[dataset.daily["store_id"] == store,
                                              "item_id"].unique()))
    wb = WritebackStore(require_approval=True)
    results = run_scenario_commit_batch(
        dataset, store, item_ids, (start, end), driver_overrides, wb, gate_policy,
        now=stamp, train_cutoff=start)

    console.print(f"[bold]scenario-commit-batch[/] store={store} "
                  f"items={len(item_ids)} drivers={driver_overrides}")
    for res in results:
        w = res.whatif
        console.print(f"  {w.item_id}: demand {w.before_demand:.1f}→{w.after_demand:.1f} "
                      f"order {res.base_order:.0f}→{res.committed.proposed_qty:.0f} "
                      f"{res.committed.status} by={res.committed.approver}")
    if not results:
        console.print("[yellow]no committed items (all skipped?)[/]")
    _write_and_label(wb, out, source)


@app.command("demand-absorption")
def cmd_demand_absorption(
    source: str = "real",
    data_dir: Path | None = None,
    close_hour: int = 22,
    out_dir: Path = REPORTS_DIR / "demand_absorption",
) -> None:
    """W0 게이트: 카테고리 총량 수요이전 흡수 검정 (leave-one-out 총량보존 β + TOST).

    β≈0(TOST 통과)=흡수→Stage 2 진입 허가, β<0=walk-away. raw sold 타깃.
    """
    from .analysis.demand_absorption import build_absorption_panel, run_absorption

    ds = _load_dataset(source, data_dir)
    panel = build_absorption_panel(ds.daily, close_hour=close_hour)
    results = run_absorption(ds.daily, close_hour=close_hour)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "panel.parquet", index=False)
    rows = pd.DataFrame([r.__dict__ for r in results])
    rows.to_csv(out_dir / "results.csv", index=False)

    console.print(f"[bold]demand-absorption[/] source={source} close_hour={close_hour}")
    for r in results:
        color = {"absorb": "green", "walkaway": "red"}.get(r.verdict, "yellow")
        console.print(f"  {r.store_id}/{r.category_id}: β={r.beta:+.3f} "
                      f"CI90[{r.ci_low:+.3f},{r.ci_high:+.3f}] δ={r.delta:.3f} "
                      f"[{color}]{r.verdict}[/] (n={r.n})")
    console.print(f"[green]wrote[/] {out_dir}/panel.parquet, results.csv")


CLOSING_DEMAND_CATEGORIES = ("bread", "pastry")
CLOSING_DEMAND_WASTE_PARQUET = Path("data/internal/v2/waste_alpha_4stores.parquet")
CLOSING_DEMAND_STORE = "광교"
CLOSING_ALPHA_CSV = "closing_alpha_estimates.csv"
CLOSING_PANEL_CSV = "closing_panel.csv"


def _load_closing_demand_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load real 광교 line-items + waste + item→category map for closing-demand α."""
    from .analysis.discount import DEFAULT_XLSX, load_sales_with_discount
    from .data import bonavi_loader as bl

    rows = load_sales_with_discount(DEFAULT_XLSX).rows
    items = bl.load_items(DEFAULT_XLSX)
    item_to_category = pd.Series(items.set_index("item_id")["category_id"])
    w = pd.read_parquet(CLOSING_DEMAND_WASTE_PARQUET)
    w = w[w["store"] == CLOSING_DEMAND_STORE]
    waste = pd.DataFrame({"date": w["date"], "item_id": w["item_id"], "waste_qty": w["out"]})
    return rows, waste, item_to_category


def _print_closing_demand_result(category: str, result: dict, diagnostics: dict) -> None:
    a = result["alpha"]
    console.print(
        f"[bold]{category}[/] α∈[{a.alpha_low:.3f}, {a.alpha_high:.3f}] "
        f"(A1={a.a1:.3f} A2={a.a2:.3f} A3_slope={a.a3_slope:.3f}) {a.note}"
    )
    console.print(f"  A2 note: {result['depth'].note}")
    console.print(
        f"  footfall: receipts_ratio={diagnostics.get('footfall_receipts_ratio', float('nan')):.3f} "
        f"qty_ratio={diagnostics.get('footfall_qty_ratio', float('nan')):.3f} "
        f"traffic_stable={diagnostics.get('traffic_stable')} a1_bias={diagnostics.get('a1_bias')}"
    )
    console.print(f"  diagnostics: {diagnostics}")


def _closing_demand_for_category(
    rows: pd.DataFrame, waste: pd.DataFrame, item_to_category: pd.Series, category: str,
) -> tuple[dict, pd.DataFrame]:
    """Run A1/A2/A3 + diagnostics for one category, print the result, return (csv_row, panel)."""
    from .analysis.closing_demand import (
        depth_time_overlap,
        evening_traffic_check,
        run_closing_demand,
    )

    result = run_closing_demand(rows, waste, item_to_category, category=category)
    a = result["alpha"]
    cat_rows = rows[rows["item_id"].map(item_to_category) == category]
    overlap = depth_time_overlap(cat_rows)
    evening_check = evening_traffic_check(rows, item_to_category, category)
    alpha_row = {
        "category": category, "a1": a.a1, "a2": a.a2,
        "alpha_low": a.alpha_low, "alpha_high": a.alpha_high,
        "a3_slope": a.a3_slope, "note": a.note,
        "a1_floor_valid": evening_check["a1_floor_valid"],
        "footfall_receipts_ratio": evening_check["footfall_receipts_ratio"],
        "footfall_qty_ratio": evening_check["footfall_qty_ratio"],
        "traffic_stable": evening_check["traffic_stable"],
        "a1_bias": evening_check["a1_bias"],
        "depth_median_hour_20": overlap["median_hour_20"],
        "depth_median_hour_30": overlap["median_hour_30"],
        "depth_time_separated": overlap["time_separated"],
    }
    _print_closing_demand_result(category, result, {**overlap, **evening_check})
    return alpha_row, result["panel"]


@app.command("closing-demand")
def cmd_closing_demand(out_dir: Path = REPORTS_DIR) -> None:
    """마감할인 실수요 α 추정 (A1 kink-in-time + A2 depth elasticity + A3 surplus counterfactual).

    광교 실측 데이터로 bread/pastry 각각 α 구간을 추정하고 CSV로 저장한다.
    """
    rows, waste, item_to_category = _load_closing_demand_inputs()
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_rows = []
    panels = []
    for category in CLOSING_DEMAND_CATEGORIES:
        alpha_row, panel = _closing_demand_for_category(rows, waste, item_to_category, category)
        alpha_rows.append(alpha_row)
        panels.append(panel)

    pd.DataFrame(alpha_rows).to_csv(out_dir / CLOSING_ALPHA_CSV, index=False)
    pd.concat(panels, ignore_index=True).to_csv(out_dir / CLOSING_PANEL_CSV, index=False)
    console.print(f"[green]wrote[/] {out_dir}/{CLOSING_ALPHA_CSV}, {CLOSING_PANEL_CSV}")


PHASEB_C_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
PHASEB_IMPLIED_C_CSV = "phaseB_implied_c.csv"
PHASEB_SAVINGS_CSV = "phaseB_order_savings.csv"


def _load_phaseb_inputs() -> tuple[pd.DataFrame, pd.Series]:
    """Load real 광교 waste rows + item→category map for Phase B order optimization."""
    from .analysis.discount import DEFAULT_XLSX
    from .data import bonavi_loader as bl

    rows = pd.read_parquet(CLOSING_DEMAND_WASTE_PARQUET)
    rows = rows[rows["store"] == CLOSING_DEMAND_STORE]
    items = bl.load_items(DEFAULT_XLSX)
    item_to_category = pd.Series(items.set_index("item_id")["category_id"])
    return rows, item_to_category


def _phaseb_exclusion_stats(rows: pd.DataFrame, item_to_category: pd.Series, category: str) -> dict:
    """Raw vs identity-kept category-day counts, for honest exclusion-rate reporting.

    Reports both the whole-day metric (a day only counts as excluded if it
    lost ALL its item-rows -- rare, since most days mix clean and dirty
    items) and the row-level metrics that actually reveal the coverage gap:
    what fraction of item-rows are dropped, and what fraction of days lose
    at least one item-row (and therefore have understated demand/made
    aggregates even though the day itself still appears in kept_days).
    """
    from .analysis.order_optimization import _identity_excluded_mask

    df = rows.copy()
    df["category_id"] = df["item_id"].astype(str).map(item_to_category)
    df = df[df["category_id"] == category]
    raw_days = int(df["date"].nunique())
    raw_rows = len(df)

    excluded = _identity_excluded_mask(df)
    kept_days = int(df.loc[~excluded, "date"].nunique())
    excl_rate = 1.0 - (kept_days / raw_days) if raw_days else float("nan")

    row_exclusion_rate = (excluded.sum() / raw_rows) if raw_rows else float("nan")
    days_with_excluded_row = int(df.loc[excluded, "date"].nunique())
    days_with_partial_exclusion_rate = (days_with_excluded_row / raw_days) if raw_days else float("nan")

    return {
        "raw_days": raw_days,
        "kept_days": kept_days,
        "exclusion_rate": excl_rate,
        "row_exclusion_rate": float(row_exclusion_rate),
        "days_with_partial_exclusion_rate": float(days_with_partial_exclusion_rate),
    }


def _phaseb_for_category(
    rows: pd.DataFrame, item_to_category: pd.Series, category: str,
) -> tuple[dict, pd.DataFrame]:
    """Run Phase B orchestrator for one category; return (implied_c row, savings table)."""
    from .analysis.order_optimization import run_phaseb

    excl = _phaseb_exclusion_stats(rows, item_to_category, category)
    result = run_phaseb(rows, item_to_category, category, c_grid=PHASEB_C_GRID)
    implied_c = result["implied_c_current"]
    savings = result["savings_table"].copy()
    savings.insert(0, "category", category)
    implied_row = {
        "category": category,
        "mean_implied_c_current": implied_c,
        "service_level": 1.0 - implied_c if pd.notna(implied_c) else float("nan"),
        **excl,
    }
    return implied_row, savings


def _print_phaseb_result(category: str, implied_row: dict, savings: pd.DataFrame) -> None:
    console.print(
        f"[bold]{category}[/] implied_c_current={implied_row['mean_implied_c_current']:.3f} "
        f"service_level={implied_row['service_level']:.3f} "
        f"(raw_days={implied_row['raw_days']} kept_days={implied_row['kept_days']} "
        f"exclusion_rate={implied_row['exclusion_rate']:.1%} "
        f"row_exclusion_rate={implied_row['row_exclusion_rate']:.1%} "
        f"days_with_partial_exclusion_rate={implied_row['days_with_partial_exclusion_rate']:.1%})"
    )
    for _, r in savings.iterrows():
        console.print(
            f"  c={r['c']:.2f} mean_implied_c={r['mean_implied_c']:.3f} "
            f"savings_vs_made={r['savings_vs_made']:+.1f} savings_l1={r['savings_l1']:+.1f} "
            f"n_days={int(r['n_days'])}"
        )


@app.command("phaseb-order")
def cmd_phaseb_order(out_dir: Path = REPORTS_DIR) -> None:
    """Phase B: 카테고리 발주 최적화 — 현행 implied c 갭 + Q* 절감 시뮬레이션.

    광교 실측 데이터로 bread/pastry 각각 implied c와 c-그리드별 절감액을 계산한다.
    절감액(savings_vs_made)이 작거나 음수일 수 있다 — placebo(Q=made) 대비 값이므로
    있는 그대로 보고한다.
    """
    rows, item_to_category = _load_phaseb_inputs()
    out_dir.mkdir(parents=True, exist_ok=True)

    implied_rows, savings_tables = [], []
    for category in CLOSING_DEMAND_CATEGORIES:
        implied_row, savings = _phaseb_for_category(rows, item_to_category, category)
        _print_phaseb_result(category, implied_row, savings)
        implied_rows.append(implied_row)
        savings_tables.append(savings)

    pd.DataFrame(implied_rows).to_csv(out_dir / PHASEB_IMPLIED_C_CSV, index=False)
    pd.concat(savings_tables, ignore_index=True).to_csv(out_dir / PHASEB_SAVINGS_CSV, index=False)
    console.print(f"[green]wrote[/] {out_dir}/{PHASEB_IMPLIED_C_CSV}, {PHASEB_SAVINGS_CSV}")


REGIME_PLACEBO_DATES = ["2022-07-17", "2023-01-17", "2023-07-17", "2024-01-17", "2024-07-17"]
REGIME_CSV = "regime_shift_estimates.csv"


def _regime_row(category: str, result: dict) -> dict:
    """Flatten a run_discount_regime result into one CSV row + print it."""
    s, i = result["closing_share"], result["closing_intensity"]
    placebo = [p.beta for p in result["placebo"] if not p.ill_posed]
    max_placebo = max((abs(b) for b in placebo), default=float("nan"))
    console.print(
        f"[bold]{category}[/] verdict={result['verdict']} (n={result['n']})\n"
        f"  closing_share post_cut β={s.beta:+.4f} 95%CI[{s.ci_low:+.4f},{s.ci_high:+.4f}]\n"
        f"  closing/made  post_cut β={i.beta:+.4f} 95%CI[{i.ci_low:+.4f},{i.ci_high:+.4f}]\n"
        f"  placebo max|β|={max_placebo:.4f}"
    )
    return {
        "category": category, "cut_date": result["cut_date"], "n": result["n"],
        "verdict": result["verdict"],
        "share_beta": s.beta, "share_ci_low": s.ci_low, "share_ci_high": s.ci_high,
        "intensity_beta": i.beta, "intensity_ci_low": i.ci_low, "intensity_ci_high": i.ci_high,
        "placebo_max_abs_beta": max_placebo,
    }


@app.command("regime-alpha")
def cmd_regime_alpha(out_dir: Path = REPORTS_DIR) -> None:
    """다중시각 재검증 ①: 2025-01-17 마감할인 depth cut(30%→20%) 자연실험.

    대조군이 없어(전 매장 동시전환) 총수요는 식별 불가. 내부통제되는 구성비
    closing_share = closing/(normal+closing)가 depth cut에 반응하는지를 item-FE
    회귀 + placebo break-date 분포로 검정한다. null(depth_invariant)이면 마감판매가
    supply-driven이며 가격민감 떨이수요가 아님 → 높은 α 방향(단 α 점식별은 아님).
    """
    from .analysis.discount_regime import run_discount_regime

    rows, item_to_category = _load_phaseb_inputs()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for category in CLOSING_DEMAND_CATEGORIES:
        result = run_discount_regime(rows, item_to_category, category,
                                     placebo_cut_dates=REGIME_PLACEBO_DATES)
        csv_rows.append(_regime_row(category, result))
    pd.DataFrame(csv_rows).to_csv(out_dir / REGIME_CSV, index=False)
    console.print(f"[green]wrote[/] {out_dir}/{REGIME_CSV}")


BASKET_CSV = "basket_composition.csv"
BASKET_CUT_DATE = "2025-01-17"


def _load_basket_inputs() -> pd.DataFrame:
    """Load 광교 receipt line-items with a proper basket key + closing label + category.

    A basket is one receipt = (판매일자, POS번호, 영수증번호); the shared discount
    loader collapses receipts to 영수증번호 alone (which recycles daily), so we build
    the composite key here from the raw sheet.
    """
    from .analysis.discount import DEFAULT_XLSX, classify_code
    from .data import bonavi_loader as bl

    sales = pd.read_excel(DEFAULT_XLSX, sheet_name="판매정보")
    sale_col = next(c for c in sales.columns if "판매구분" in c)
    sales = sales[sales[sale_col].astype(str).str[0] == "0"].copy()   # drop 반품

    code = sales["할인코드"].astype(str).str.strip()
    label = code.map(classify_code).where(sales["할인금액"].fillna(0) > 0, "none")
    items = bl.load_items(DEFAULT_XLSX)
    item_to_category = pd.Series(items.set_index("item_id")["category_id"])
    date = pd.to_datetime(sales["판매일자"].astype(str), format="%Y%m%d")
    return pd.DataFrame({
        "basket_id": (date.dt.strftime("%Y%m%d") + "-"
                      + sales["POS번호"].astype(str) + "-"
                      + sales["영수증번호"].astype(str)),
        "date": date,
        "label": label.to_numpy(),
        "category_id": sales["품목코드"].astype(str).map(item_to_category).to_numpy(),
        "qty": sales["판매수량"].astype(float).to_numpy(),
        "paid": sales["결제금액"].astype(float).to_numpy(),
    })


def _basket_row(scope: str, category, summary: dict) -> dict:
    console.print(
        f"[bold]{scope}[/] closing_category={category}: "
        f"n_closing={summary['n_closing_baskets']} "
        f"mixed_rate={summary['mixed_rate']:.3f} "
        f"fp_value_share={summary['fullprice_value_share']:.3f} "
        f"size(closing/other)={summary['mean_size_closing']:.2f}/{summary['mean_size_noclosing']:.2f}"
    )
    return {"scope": scope, "closing_category": category or "any", **summary}


@app.command("basket-alpha")
def cmd_basket_alpha(out_dir: Path = REPORTS_DIR) -> None:
    """다중시각 재검증 ③: 마감할인 basket 구성 (실쇼핑 vs 떨이단독) — 저녁 confound 확인용.

    영수증(basket) 단위로, 마감할인 line을 담은 basket이 정가품도 함께 담는지
    (mixed_rate / 정가 금액비중)를 본다. ⚠️ 마감할인이 저녁(20-21h) time-lock이고
    그 시각 정가품 재고가 ~5%로 붕괴하므로 낮은 mixed_rate는 시각 confound다 —
    독립 α 판별자가 아니라 Phase A 저녁 잠식의 재현으로 읽어야 한다.
    전체 + depth regime(30% pre / 20% post 2025-01-17)별로 리포트.
    """
    from .analysis.basket_composition import basket_composition_summary

    lines = _load_basket_inputs()
    out_dir.mkdir(parents=True, exist_ok=True)
    cut = pd.Timestamp(BASKET_CUT_DATE)
    scopes = {
        "all": lines,
        "pre_cut_30pct": lines[lines["date"] < cut],
        "post_cut_20pct": lines[lines["date"] >= cut],
    }
    rows = []
    for scope, sub in scopes.items():
        for category in (None, "bread", "pastry"):
            rows.append(_basket_row(scope, category,
                                    basket_composition_summary(sub, closing_category=category)))
    pd.DataFrame(rows).to_csv(out_dir / BASKET_CSV, index=False)
    console.print(f"[green]wrote[/] {out_dir}/{BASKET_CSV}")


# ---------------------------------------------------------------------------
# prospective-eval: 전향적 KPI harness (우리 발주 추천 vs 현행 발주)
# ---------------------------------------------------------------------------

def _synthetic_prospective_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """소형 결정론 합성 데이터 — item a/b, 3일. 우리 발주=수요근접, 현행=과발주."""
    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    rows = pd.DataFrame({
        "item_id": ["a"] * 3 + ["b"] * 3,
        "date": dates * 2,
        "potential_demand": [100.0, 100.0, 100.0, 60.0, 60.0, 60.0],
        "our_order": [105.0, 105.0, 105.0, 63.0, 63.0, 63.0],
        "base_order": [140.0, 140.0, 140.0, 90.0, 90.0, 90.0],
    })
    receipts = pd.DataFrame({
        "item_id": ["a"] * 6 + ["b"] * 6,
        "date": [d for d in dates for _ in range(2)] * 2,
        "hour": [9, 14] * 6,
        "qty": [50.0, 50.0] * 3 + [30.0, 30.0] * 3,
    })
    unit_prices = {"a": 1000.0, "b": 1500.0}
    return rows, receipts, unit_prices


# 실데이터 진입점 — 재고정보 시트가 있는 파일만 생산량/폐기량/품목단가를 갖는다.
REAL_INVENTORY_XLSX_PATH = "data/internal/보나비 데이터_20260526.xlsx"
REAL_DAILY_PARQUET_PATH = "data/internal/bonavi_daily.parquet"
REAL_RECEIPTS_PARQUET_PATH = "data/internal/bonavi_receipts.parquet"

REAL_ROWS_COLUMNS = [
    "item_id", "date", "category_id", "potential_demand", "adjusted_demand",
    "sold_units", "is_stockout", "base_order", "waste_qty",
]


def select_base_order(merged: pd.DataFrame, *, source: str = "production") -> pd.Series:
    """현행 발주 proxy 선택. 지금은 생산량만. 전향 실발주 수령 시 여기만 확장(swap 지점)."""
    if source == "production":
        return merged["production_qty"].astype(float)
    raise ValueError(f"unsupported base_order source: {source!r} (only 'production' until 실발주 수령)")


def _assemble_real_rows(
    daily: pd.DataFrame, inventory: pd.DataFrame, bulk_qty: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """bonavi_daily(store/category 필터됨) + load_inventory(A) 결과를 (date,item_id) 조인.

    daily["date"]는 datetime64(receipts와 동일 표현 유지 — build_arrival_profile의
    str-cast 키 비교 계약 때문에 여기서 별도 문자열로 바꾸지 않는다). inventory["date"]는
    load_inventory 계약상 YYYYMMDD 문자열이므로 조인 전에만 datetime으로 정규화한다.
    재고정보에 매칭이 없는 item-day는 base_order가 없어 평가셋에서 제외한다(inner join).

    bulk_qty(item_id/date/bulk_qty, `_real_bulk_qty` 출력)가 주어지면 생산량(QT_MADE)에서
    해당 item-day 예약(bulk) 판매량을 차감한다(측정 헌장 §1: 생산=판매+폐기 정합 —
    sold_units는 bulk 제외본인데 QT_MADE는 미제외라 base_order가 부풀려짐).
    receipts 소스가 정확한 build-time 소스와 광교에서 완전 일치함을 확인
    (scripts/check_production_bulk_consistency.py, regression guard). 차감 후 <1% item-day는
    생산<판매+폐기가 되나(예약이 생산과 다른 날 잡힌 소수 케이스) 편향이 보수적이라 clip만."""
    inv = inventory.copy()
    inv["date"] = pd.to_datetime(inv["date"], format="%Y%m%d")
    inv["item_id"] = inv["item_id"].astype(str)
    d = daily.copy()
    d["item_id"] = d["item_id"].astype(str)
    merged = d.merge(
        inv[["date", "item_id", "production_qty", "waste_qty"]],
        on=["date", "item_id"], how="inner",
    )
    if bulk_qty is not None and not bulk_qty.empty:
        b = bulk_qty.copy()
        b["item_id"] = b["item_id"].astype(str)
        b["date"] = pd.to_datetime(b["date"])
        merged = merged.merge(b[["item_id", "date", "bulk_qty"]], on=["item_id", "date"], how="left")
        removed = merged["bulk_qty"].fillna(0.0)
        net = (merged["production_qty"].astype(float) - removed)
        n_clipped = int((net < 0).sum())
        merged["production_qty"] = net.clip(lower=0)
        merged = merged.drop(columns=["bulk_qty"])
        console.print(f"[cyan]생산 bulk 제외[/] {int((removed > 0).sum())} item-day에서 "
                      f"{removed.sum():.0f}개 차감 (음수→0 clip {n_clipped}개, 보수적)")
    merged["base_order"] = select_base_order(merged, source="production")
    return merged[REAL_ROWS_COLUMNS].reset_index(drop=True)


def _real_bulk_qty() -> pd.DataFrame:
    """bonavi_receipts → item-day별 예약(bulk) 판매량. 생산량 bulk 제외용(헌장 §1).

    receipts의 is_bulk(flag_bulk_lines) 라인 qty를 (item_id,date)로 합산. 이 값이
    build-time load_sales의 정확한 bulk 제거량과 광교에서 완전 일치함을 검증
    (scripts/check_production_bulk_consistency.py). is_bulk 컬럼 없는 legacy parquet은 빈 프레임."""
    rec = pd.read_parquet(REAL_RECEIPTS_PARQUET_PATH)
    if "is_bulk" not in rec.columns:
        return pd.DataFrame({"item_id": [], "date": [], "bulk_qty": []})
    rec = rec[rec["is_bulk"].astype(bool)].copy()
    rec["item_id"] = rec["item_id"].astype(str)
    rec["date"] = pd.to_datetime(rec["date"]).dt.normalize()
    rec["qty"] = pd.to_numeric(rec["qty"], errors="coerce").fillna(0.0)
    return rec.groupby(["item_id", "date"], as_index=False)["qty"].sum().rename(
        columns={"qty": "bulk_qty"}
    )


def _load_real_daily(store_id: str) -> pd.DataFrame:
    """bonavi_daily.parquet을 store_id + TARGET_CATEGORIES로 필터."""
    daily = pd.read_parquet(REAL_DAILY_PARQUET_PATH)
    n_stores = daily["store_id"].nunique()
    if n_stores != 1:
        raise ValueError(f"real path assumes single-store data; found {n_stores} stores. Multi-store needs store-qualified receipts/merge wiring.")
    daily["item_id"] = daily["item_id"].astype(str)
    daily = daily[daily["store_id"] == store_id]
    daily = daily[daily["category_id"].isin(TARGET_CATEGORIES)]
    return daily.reset_index(drop=True)


def _receipts_profile_frame(receipts: pd.DataFrame, item_ids: set[str]) -> pd.DataFrame:
    """receipts 프레임 → arrival profile 입력(item_id/date/hour/qty).

    - **수량-가중**: qty=판매수량(라인당 실수량). 시간대별 재고 소진을 방문 건수가 아니라
      실제 판매량으로 재구성 → 매진시각 시뮬 정교화.
    - **bulk 국소 제외**: is_bulk(예약) 라인 제외. profile이 분배하는 daily 수요
      (adjusted_demand)가 bulk 제외본이므로 shape도 walk-in 수량으로 population을 맞춘다.
    - legacy parquet(qty·is_bulk 컬럼 부재)는 footfall 1-count로 graceful fallback.
    """
    receipts = receipts.copy()
    receipts["item_id"] = receipts["item_id"].astype(str)
    receipts = receipts[receipts["item_id"].isin(item_ids)]
    if "is_bulk" in receipts.columns:
        receipts = receipts[~receipts["is_bulk"].astype(bool)]
    if "qty" in receipts.columns:
        receipts = receipts.assign(
            qty=pd.to_numeric(receipts["qty"], errors="coerce").fillna(0.0).astype(float)
        )
    else:
        receipts = receipts.assign(qty=1.0)
    return receipts[["item_id", "date", "hour", "qty"]].reset_index(drop=True)


def _load_real_receipts(item_ids: set[str]) -> pd.DataFrame:
    """bonavi_receipts.parquet → item_id/date/hour/qty (수량-가중·bulk 제외).

    주의: receipts 원본에는 store 컬럼이 없다. bonavi 데이터셋 자체가 광교(store_gw01)
    단일 매장 실측이라(bonavi_daily의 store_id도 store_gw01뿐) 현재는 store 필터가
    불필요하지만, 다매장 실데이터로 확장되면 이 함수는 store 컬럼 부재로 깨진다.
    """
    receipts = pd.read_parquet(REAL_RECEIPTS_PARQUET_PATH)
    return _receipts_profile_frame(receipts, item_ids)


def _load_unit_prices(xlsx_path: str) -> dict[str, float]:
    """품목정보 시트 → item_id→판매단가 dict. NaN은 category_aggregate와 동일하게 4000 fallback."""
    items = pd.read_excel(xlsx_path, sheet_name="품목정보")
    items["item_id"] = items["품목코드"].astype(str)
    items["판매단가"] = pd.to_numeric(items["판매단가"], errors="coerce")
    return items.set_index("item_id")["판매단가"].fillna(4000.0).to_dict()


def _category_base_predict(
    train: pd.DataFrame, test: pd.DataFrame, *,
    target_col: str, total_model: str, production_quantile: float,
) -> tuple:
    """train으로 카테고리 총량 모델 fit → test의 (base_median, base_prod, base_sigma) 반환(clip≥0).

    base_sigma는 distributional일 때 날짜별 σ(log-space) ndarray, lightgbm이면 None.
    total_model 분기: lightgbm(production_q fit 고정) | distributional(production_q predict 시).
    fold backtest·future 예측 공용(중복 제거)."""
    import numpy as np

    if total_model == "distributional":
        model = fit_distributional_total(train, target_col=target_col)
        base_prod = np.clip(model.predict_production(test, production_q=production_quantile), 0.0, None)
        base_sigma = np.asarray(model.predict_sigma(test), dtype=float)
    elif total_model == "lightgbm":
        model = fit_category_total(train, target_col=target_col, production_q=production_quantile)
        base_prod = np.clip(model.predict_production(test), 0.0, None)
        base_sigma = None
    else:
        raise ValueError(f"unknown total_model: {total_model!r} (expected 'lightgbm' or 'distributional')")
    base_median = np.clip(model.predict_expected(test), 0.0, None)
    return base_median, base_prod, base_sigma


def _blend_event_prior(
    train: pd.DataFrame, dates, base_median, base_prod, *, target_col: str,
) -> tuple:
    """EventLevelPrior를 train(pre-test history)으로 fit 후 이벤트일만 레벨-앵커 블렌드.

    leakage-safe: prior는 예측창 이전 데이터로만 fit, level_for는 ed<date 엄격 필터."""
    import numpy as np

    prior = EventLevelPrior(events=EVENTS, lunar_events=LUNAR_EVENTS).fit(train, target_col=target_col)
    base_median, base_prod = prior.blend(dates, base_median, base_prod)
    return np.clip(base_median, 0.0, None), np.clip(base_prod, 0.0, None)


def _category_total_fold_predictions(
    features: pd.DataFrame, *, production_quantile: float, horizon_days: int,
    n_folds: int, target_col: str = "adjusted_demand_unit", min_train_days: int = 365,
    total_model: str = "lightgbm", event_prior: bool = True,
) -> pd.DataFrame:
    """expanding-window fold별 q{production_quantile} 카테고리 총합 발주.

    category_total.expanding_window_backtest의 leakage-safe 패턴(train=이전/test=이후
    iloc 분할, sorted date 1행/1일)을 따르되 production 예측을 test date별로 반환한다.

    total_model: "lightgbm"(CategoryTotalModel, production_q를 fit 시 고정) |
    "distributional"(DistributionalTotalModel/NGBoost, production_q를 predict 시 지정).
    둘 다 predict_expected(median)/predict_production 계약을 공유해 drop-in swap.

    event_prior=True: sharp 캘린더 이벤트(xmas/설/추석 등)에 EventLevelPrior 레벨-앵커
    블렌드를 post-model로 적용(모델 무관). prior는 fold마다 pre-test history로만 fit
    (leakage-safe) 후 base_median/base_prod를 이벤트일에서만 보정한다. margin(quantile/
    nk/conformal) 적용 전이라 3방식 모두에 반영된다.
    """
    import numpy as np

    df = features.sort_values("date").dropna().reset_index(drop=True)
    total = len(df)
    if total < min_train_days + n_folds * horizon_days:
        raise ValueError(f"not enough category-days: {total} < {min_train_days + n_folds * horizon_days}")
    chunks = []
    for k in range(n_folds):
        test_end = total - k * horizon_days
        test_start = test_end - horizon_days
        test_df = df.iloc[test_start:test_end]
        train_df = df.iloc[:test_start]
        base_median, base_prod, _ = _category_base_predict(
            train_df, test_df, target_col=target_col,
            total_model=total_model, production_quantile=production_quantile,
        )
        if event_prior:
            base_median, base_prod = _blend_event_prior(
                train_df, test_df["date"], base_median, base_prod, target_col=target_col,
            )
        chunks.append(pd.DataFrame({
            "date": test_df["date"].to_numpy(), "fold": k,
            "base_median": base_median, "base_prod": base_prod,
            "actual": test_df[target_col].to_numpy(),
        }))
    return pd.concat(chunks, ignore_index=True)


def _apply_category_margin(
    totals: pd.DataFrame, method: str, *,
    nk_mult: float = 1.0, nk_add: float = 0.0,
    service_level: float = 0.85, cal_fold_frac: float = 0.5,
) -> pd.DataFrame:
    """총량 fold 예측[date,fold,base_median,base_prod,actual] → [date,fold,total_order].

    발주 마진을 카테고리 **총량**에 적용하는 3방식(선택):
    - quantile: total_order = base_prod (LGBM production-quantile 직접). 현행·하위호환.
    - nk: total_order = base_prod × nk_mult + nk_add (수동 안전마진, margin_optimize식).
    - conformal: base_median + q_s × scale (scale=base_median, 레벨-정규화 승법 마진).
      fold half-split(앞=cal/뒤=test)로 q_s를 데이터에서 fit → 목표 서비스레벨 s 달성.
      cal folds는 결과에서 빠진다(item conformal과 동일).
    """
    import numpy as np

    if method == "quantile":
        return totals.assign(total_order=totals["base_prod"])[["date", "fold", "total_order"]]
    if method == "nk":
        # 버퍼는 **median(q0.5) 기준**에 얹는다 — margin_optimize/HTML 철학(median×K+N)과
        # 일치, conformal(median+q_s×scale)과도 동일 base라 공정 비교. (q0.85 위 버퍼는 이중마진)
        order = np.clip(totals["base_median"].to_numpy() * nk_mult + nk_add, 0.0, None)
        return totals.assign(total_order=order)[["date", "fold", "total_order"]]
    if method == "conformal":
        return _category_conformal_total(
            totals, service_level=service_level, cal_fold_frac=cal_fold_frac,
        )
    raise ValueError(f"unknown category-margin method: {method!r}")


def _category_conformal_total(
    totals: pd.DataFrame, *, service_level: float, cal_fold_frac: float,
) -> pd.DataFrame:
    """카테고리 총량 conformal 보정(순수 함수). base=median, scale=median(레벨 정규화).

    fold 정수는 시간순 아님 → fold별 최소 날짜로 연대 정렬, 앞쪽=cal/뒤쪽=test.
    E=(actual−base_median)/scale 의 s-분위 q_s를 cal에서 fit, test에 base+q_s×scale 적용.
    """
    import numpy as np

    fold_order = totals.groupby("fold")["date"].min().sort_values().index.tolist()
    n_cal = max(1, int(len(fold_order) * cal_fold_frac))
    cal_folds = set(fold_order[:n_cal])
    test_folds = set(fold_order[n_cal:]) or {fold_order[-1]}   # 최소 1개 test 보장
    cal = totals[totals["fold"].isin(cal_folds)]
    test = totals[totals["fold"].isin(test_folds)].copy()

    def _scale(base: np.ndarray) -> np.ndarray:
        return np.where(base > 0, base, 1.0)

    cal_scale = _scale(cal["base_median"].to_numpy())
    scores = (cal["actual"].to_numpy() - cal["base_median"].to_numpy()) / cal_scale
    calib = ConformalOrderCalibrator().fit(scores, service_level)
    test["total_order"] = calib.apply(
        test["base_median"].to_numpy(), _scale(test["base_median"].to_numpy())
    )
    return test[["date", "fold", "total_order"]].reset_index(drop=True)


def _category_order_predictions(
    store_id: str, *, production_quantile: float = 0.85, val_weeks: int = 8, n_folds: int = 1,
    alpha: float = DEFAULT_ALPHA, margin_method: str = "quantile",
    nk_mult: float = 1.0, nk_add: float = 0.0,
    service_level: float = 0.85, cal_fold_frac: float = 0.5,
    drop_features: Collection[str] = (), total_model: str = "lightgbm",
    event_prior: bool = True,
) -> pd.DataFrame:
    """v4 카테고리 스택: build_category_daily → fold별 총합 base(Task1) → 마진 적용
    (quantile/nk/conformal) → distribute_total 배분 → item별 our_order.
    item 경로(_our_order_predictions)와 동일 [item_id,date,fold,our_order].

    drop_features: build_features에서 뺄 feature 그룹(실험용, 기본=아무것도 안 뺌).
    total_model: 총량 base 학습 모델(lightgbm | distributional). distributional의
    σ(x) 이득은 margin_method=quantile에서만 흐른다(nk/conformal은 median 기반).
    event_prior: sharp 캘린더 이벤트 레벨-앵커 블렌드(post-model, 모델 무관)."""
    # build_category_daily()는 store-agnostic(parquet 전체 읽음) — 단일매장 데이터셋 +
    # _load_real_daily의 단일매장 가드 덕에 안전. 다매장 확장 시 store_id 필터링 재검토 필요.
    features = build_features(
        build_category_daily(alpha=alpha),
        target_col="adjusted_demand_unit", drop_groups=drop_features,
    )
    base = _category_total_fold_predictions(
        features, production_quantile=production_quantile,
        horizon_days=val_weeks * 7, n_folds=n_folds, total_model=total_model,
        event_prior=event_prior,
    )
    totals = _apply_category_margin(
        base, margin_method, nk_mult=nk_mult, nk_add=nk_add,
        service_level=service_level, cal_fold_frac=cal_fold_frac,
    )
    daily = _load_real_daily(store_id)          # 배분 비율 history (compute_proportions가 <date만 사용)
    chunks = []
    for fold, g in totals.groupby("fold"):
        res = distribute_total(daily, g.set_index("date")["total_order"])
        q = res.quantities.rename(columns={"qty": "our_order"})
        q["fold"] = int(fold)
        chunks.append(q[["item_id", "date", "fold", "our_order"]])
    preds = pd.concat(chunks, ignore_index=True)
    preds["item_id"] = preds["item_id"].astype(str)
    margin_desc = {
        "quantile": f"q={production_quantile}",
        "nk": f"nk(×{nk_mult}+{nk_add}, base q={production_quantile})",
        "conformal": f"conformal(s={service_level}, cal_frac={cal_fold_frac})",
    }.get(margin_method, margin_method)
    console.print(
        f"[cyan]category our_order[/] {n_folds} fold(s) × {val_weeks}주, model={total_model}, "
        f"margin={margin_desc}, event_prior={'on' if event_prior else 'off'}, "
        f"{preds['date'].nunique()} dates × {preds['item_id'].nunique()} items"
    )
    return preds


# item-schema 예보 → category weather 스키마 부분 매핑(기온/강수/습도만; 구름/풍속 미대응).
_FORECAST_TO_CATEGORY_WEATHER = {
    "avg_temp": "avgTa", "max_temp": "maxTa", "min_temp": "minTa",
    "precipitation_mm": "sumRn", "humidity": "avgRhm",
}


def _forecast_to_category_weather(forecast_weather: pd.DataFrame, store_id: str) -> pd.DataFrame | None:
    """_load_forecast_weather의 (store_id,date) 프레임 → category weather 스키마(date-keyed).

    기온/강수/습도만 매핑하고 rain_level은 sumRn에서 재계산. 구름(avgTca)·풍속(avgWs)·
    apparent_temp는 예보에 없어 미제공(fill_forecast_weather가 NaN 유지). store 미매칭이면 None."""
    fw = forecast_weather[forecast_weather["store_id"] == store_id]
    if fw.empty:
        return None
    out = pd.DataFrame({"date": pd.to_datetime(fw["date"].to_numpy())})
    for src_col, dst_col in _FORECAST_TO_CATEGORY_WEATHER.items():
        if src_col in fw.columns:
            out[dst_col] = fw[src_col].to_numpy()
    if "sumRn" in out.columns:
        out["rain_level"] = pd.cut(
            out["sumRn"].fillna(0), bins=[-1, 0, 5, 20, 1e9], labels=[0, 1, 2, 3]
        ).astype(int)
    return out


def _extend_category_features(
    hist: pd.DataFrame, *, horizon_days: int, alpha: float, target_col: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """history 카테고리 daily에 미래 horizon_days 행(target=NaN)을 append 후 build_features.

    미래 행의 target이 NaN이라 lag=shift가 미래-reaching 구간에서 NaN이 된다(leakage 차단).
    (feats, horizon) 반환. build_features와 분리해 leakage 회귀 테스트가 붙는 지점."""
    hist = hist.sort_values("date").reset_index(drop=True)
    last = hist["date"].max()
    horizon = pd.date_range(last + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    ext = pd.concat([hist, pd.DataFrame({"date": horizon})], ignore_index=True)
    feats = build_features(CategoryDaily(df=ext, alpha=alpha), target_col=target_col)
    return feats.sort_values("date").reset_index(drop=True), horizon


def _category_future_order_predictions(
    store_id: str, *, horizon_days: int = 7, production_quantile: float = 0.85,
    total_model: str = "lightgbm", event_prior: bool = True,
    alpha: float = DEFAULT_ALPHA, use_forecast: bool = True,
) -> pd.DataFrame:
    """미래 horizon_days일 카테고리 총량 예측 → item 배분.

    [store_id, item_id, category_id, date, demand_point, our_order] 반환.
    demand_point=median 배분(점추정), our_order=production 분위수 배분(발주). 동일 비율로
    배분해 두 컬럼이 정합한다(demand_point는 c-2b decision·미리보기가 소비).

    leakage-safe: 미래 카테고리 행을 target=NaN으로 append하면 build_features의 lag가 seam
    너머 계산돼 미래-reaching lag는 NaN이 된다(미래 실측 원천 차단, item _join_history와 동일
    원리). fit은 관측 history(dropna)만 사용하고 미래 행은 예측 대상이다. 날씨는 부분 예보
    주입(기온/강수/습도; 구름/풍속 NaN). production은 total_model(lightgbm|distributional)·
    event_prior 재사용(PR#49 배선)."""
    target_col = "adjusted_demand_unit"
    hist = build_category_daily(alpha=alpha).df
    feats, horizon = _extend_category_features(
        hist, horizon_days=horizon_days, alpha=alpha, target_col=target_col,
    )
    if use_forecast:
        fw = _load_forecast_weather(horizon)
        cat_fw = _forecast_to_category_weather(fw, store_id) if fw is not None else None
        if cat_fw is not None:
            feats = fill_forecast_weather(feats, cat_fw)
        else:
            console.print(f"[yellow]forecast[/] {store_id} 미매칭 — 미래 날씨 NaN 유지")
    feats = feats.sort_values("date").reset_index(drop=True)
    is_future = feats["date"].isin(horizon)
    train = feats[~is_future].dropna(subset=[target_col])
    test = feats[is_future]
    base_median, base_prod, base_sigma = _category_base_predict(
        train, test, target_col=target_col,
        total_model=total_model, production_quantile=production_quantile,
    )
    if event_prior:
        base_median, base_prod = _blend_event_prior(
            train, test["date"], base_median, base_prod, target_col=target_col,
        )
    dates = test["date"].to_numpy()
    daily = _load_real_daily(store_id)          # 배분 비율 history (compute_proportions가 <date만)
    order = distribute_total(daily, pd.Series(base_prod, index=dates)).quantities.rename(
        columns={"qty": "our_order"})
    point = distribute_total(daily, pd.Series(base_median, index=dates)).quantities.rename(
        columns={"qty": "demand_point"})
    preds = order.merge(point, on=["item_id", "date"], how="left")
    preds["item_id"] = preds["item_id"].astype(str)
    cat_src = daily.drop_duplicates("item_id").assign(item_id=lambda d: d["item_id"].astype(str))
    cat_map = cat_src.set_index("item_id")["category_id"]
    preds["store_id"] = store_id
    preds["category_id"] = preds["item_id"].map(cat_map)
    if base_sigma is not None:
        sigma_by_date = dict(zip(pd.to_datetime(dates), base_sigma))
        preds["demand_sigma_log"] = pd.to_datetime(preds["date"]).map(sigma_by_date).astype(float)
    else:
        preds["demand_sigma_log"] = float("nan")
    console.print(
        f"[cyan]category future order[/] {horizon[0].date()}~{horizon[-1].date()}, "
        f"model={total_model}, event_prior={'on' if event_prior else 'off'}, "
        f"forecast={'on' if use_forecast else 'off'}, "
        f"{preds['date'].nunique()} dates × {preds['item_id'].nunique()} items"
    )
    return preds[["store_id", "item_id", "category_id", "date",
                  "demand_point", "our_order", "demand_sigma_log"]]


def _quantile_backtest_predictions(
    daily: pd.DataFrame, *, val_weeks: int, production_quantile: float, n_folds: int = 1,
    target_col: str = "adjusted_demand",
) -> tuple[pd.DataFrame, list[SplitWindow]]:
    """최근 n_folds개 non-overlapping val 창(각 val_weeks)에서 q{α} v2 예측.

    Leakage 없음 — val 이전 전체 기간이 train(generate_time_splits의 expanding
    모드 + apply_split의 leakage assertion에 의존, 직접 구현하지 않는다).
    daily는 호출자가 store/기간 필터·calendar/weather enrich를 마친 상태여야 하고,
    target_col 컬럼을 반드시 보유해야 한다.
    fold별 KPI 집계를 위해 fold 컬럼을 보존한다.
    """
    windows = generate_time_splits(
        daily["date"], n_splits=n_folds,
        val_horizon_days=val_weeks * 7, step_days=val_weeks * 7,
    )
    forecaster = GlobalLGBM(
        feature_set="v2", y_col=target_col,
        params=LGBMParams(objective="quantile", alpha=production_quantile),
    )
    _, pred_df = run_backtest(daily, [forecaster], windows, y_col=target_col)
    preds = pred_df[["item_id", "date", "fold", "yhat"]].rename(columns={"yhat": "our_order"})
    return preds, windows


def _our_order_predictions(
    store_id: str, *, production_quantile: float = 0.85, val_weeks: int = 8, n_folds: int = 1,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """bonavi_daily(real) → v2 enrich → 최근 n_folds개(각 val_weeks) backtest val의
    q{production_quantile} 예측. 5년 전체를 expanding-window backtest하기엔 비용이 크므로
    최근 구간으로 제한(fold 수는 console.print로 명시 — 무언 축소 금지)."""
    ds = _load_dataset("real", None)
    daily = _enrich_if_needed(ds, ["v2"])
    daily = build_item_adjusted_demand(daily, alpha=alpha)
    preds, windows = _quantile_backtest_predictions(
        daily, val_weeks=val_weeks, production_quantile=production_quantile, n_folds=n_folds,
        target_col="adjusted_demand",
    )
    console.print(
        f"[cyan]our_order[/] {len(windows)} fold(s), each {val_weeks}주 "
        f"(store={store_id}, quantile α={production_quantile})"
    )
    return preds


def _apply_conformal_to_folds(
    pred_df: pd.DataFrame, scale: dict[str, float], *,
    service_level: float, cal_fold_frac: float,
) -> pd.DataFrame:
    """fold별 base 예측을 앞쪽(cal)/뒤쪽(test)로 half-split → conformal 보정.

    순수 함수(실 LGBM 무관): pred_df[item_id,date,fold,adjusted_demand,yhat] +
    item scale dict → test folds의 [item_id,date,fold,our_order].
    """
    # fold 정수는 시간순이 아님(generate_time_splits: fold=0=최신). pred_df의
    # fold별 최소 날짜로 연대순 정렬해 앞쪽(이른)=cal, 뒤쪽(늦은)=test.
    fold_order = pred_df.groupby("fold")["date"].min().sort_values().index.tolist()
    n_cal = max(1, int(len(fold_order) * cal_fold_frac))
    cal_folds, test_folds = set(fold_order[:n_cal]), set(fold_order[n_cal:])
    cal = pred_df[pred_df["fold"].isin(cal_folds)]
    test = pred_df[pred_df["fold"].isin(test_folds)].copy()

    def _scale_of(items: pd.Series) -> np.ndarray:
        return items.astype(str).map(scale).fillna(1.0).to_numpy()

    cal_scale = _scale_of(cal["item_id"])
    scores = ((cal["adjusted_demand"].to_numpy() - cal["yhat"].to_numpy()) / cal_scale)
    calib = ConformalOrderCalibrator().fit(scores, service_level)
    test["our_order"] = calib.apply(test["yhat"].to_numpy(), _scale_of(test["item_id"]))
    return test[["item_id", "date", "fold", "our_order"]].reset_index(drop=True)


def _median_base_fold_predictions(
    daily: pd.DataFrame, *, val_weeks: int, n_folds: int,
) -> tuple[pd.DataFrame, list]:
    """v2 LGBM q0.5(median) base로 expanding backtest. pred_df에 actual+yhat+fold 보존."""
    windows = generate_time_splits(
        daily["date"], n_splits=n_folds,
        val_horizon_days=val_weeks * 7, step_days=val_weeks * 7,
    )
    forecaster = GlobalLGBM(
        feature_set="v2", y_col="adjusted_demand",
        params=LGBMParams(objective="quantile", alpha=0.5),
    )
    _, pred_df = run_backtest(daily, [forecaster], windows, y_col="adjusted_demand")
    pred_df["item_id"] = pred_df["item_id"].astype(str)
    return pred_df[["item_id", "date", "fold", "adjusted_demand", "yhat"]], windows


def _conformal_order_predictions(
    store_id: str, *, service_level: float = DEFAULT_SERVICE_LEVEL,
    val_weeks: int = 8, n_folds: int = 8, cal_fold_frac: float = 0.5,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """base median 발주 + cross-fold half-split conformal 보정 → test folds our_order."""
    if n_folds < 2:
        raise ValueError(
            f"--calibrate needs n_folds >= 2 (cal/test half-split); got {n_folds}"
        )
    ds = _load_dataset("real", None)
    daily = _enrich_if_needed(ds, ["v2"])
    daily = build_item_adjusted_demand(daily, alpha=alpha)
    pred_df, windows = _median_base_fold_predictions(daily, val_weeks=val_weeks, n_folds=n_folds)
    first_val_start = min(w.val_start for w in windows)
    scale = compute_item_scale(daily, before_date=first_val_start, y_col="adjusted_demand")
    out = _apply_conformal_to_folds(
        pred_df, scale, service_level=service_level, cal_fold_frac=cal_fold_frac
    )
    console.print(
        f"[cyan]conformal our_order[/] {n_folds} fold(s)(cal {int(n_folds*cal_fold_frac)}/"
        f"test {n_folds-int(n_folds*cal_fold_frac)}), s={service_level}, "
        f"{out['date'].nunique()} dates × {out['item_id'].nunique()} items"
    )
    return out


def _fill_our_order(rows: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """rows(Task B, 전체 기간)를 our_order 예측이 존재하는 backtest val 기간(들)로
    제한한다. 예측 없는 item-day는 평가셋에서 제외 — 개수를 log로 명시(무언 축소 금지)."""
    preds = predictions.copy()
    preds["item_id"] = preds["item_id"].astype(str)
    before = len(rows)
    merged = rows.merge(preds, on=["item_id", "date"], how="inner")
    dropped = before - len(merged)
    console.print(
        f"[cyan]our_order[/] scored item-days: {len(merged):,} / {before:,} "
        f"(dropped {dropped:,} outside backtest val window)"
    )
    if "fold" in merged.columns:
        folds = sorted(int(f) for f in merged["fold"].unique())
        console.print(f"[cyan]our_order[/] fold 컬럼 보존됨: {folds}")
    return merged


def _artisee_fold_order(
    store_id: str, daily: pd.DataFrame, hourly_all: pd.DataFrame, fold_rows: pd.DataFrame,
) -> pd.Series:
    """단일 fold 발주: cutoff=fold_rows 타깃 최소일 이전 데이터로만 fit → 이 fold만 예측.

    Leakage 방지: train_daily/hourly 모두 cutoff 미만으로 절단 — fold_rows 기간 내
    미래 실측이 곡선/배수에 섞이지 않는다.
    M1: daily의 is_holiday(calendar 유래, 날짜 단위)를 target에 병합 — 명절/건물휴장
    타깃일을 ArtiseeBaseline.predict가 주말로 취급하도록 신호 전달.
    """
    cutoff = pd.to_datetime(fold_rows["date"]).min()
    train_daily = daily[daily["date"] < cutoff]
    hourly = hourly_all[hourly_all["item_id"].isin(set(train_daily["item_id"]))]
    hourly = hourly[hourly["date"] < cutoff]
    model = ArtiseeBaseline().fit(train_daily, hourly)
    holiday_by_date = daily.drop_duplicates("date").set_index("date")["is_holiday"]
    target = fold_rows[["item_id", "date"]].copy()
    target["store_id"] = store_id
    target["is_holiday"] = target["date"].map(holiday_by_date).fillna(False)
    order = model.predict(target)
    return pd.Series(order.to_numpy(), index=fold_rows.index)


def _artisee_baseline_order(store_id: str, rows: pd.DataFrame) -> pd.Series:
    """ArtiseeBaseline 재구현 발주(real 소스 전용).

    daily는 bonavi_daily(is_holiday 결측) + calendar의 is_public_holiday를
    is_holiday로 대체 부착. hourly는 receipts 재사용(item_id/date/hour/qty 스키마
    호환, build_item_residual_curve가 요구하는 컬럼과 일치).

    I1(fold별 재fit): rows에 our_order와 동일한 `fold` 컬럼이 있으면(_fill_our_order가
    real 경로에서 항상 부착) fold별로 별도 재fit한다 — cutoff=그 fold 타깃 최소일.
    고객사는 매주 월요일 새벽 trailing 3주로 재계산하는데, 옛 구현(전체 rows 기간에
    대해 cutoff=rows["date"].min() 단일 fit)은 뒤쪽 fold일수록 아띠제baseline을 실제
    보다 낡은 정보로 묶어 Δ를 우리 쪽으로 유리하게 편향시켰다. fold 컬럼이 없으면
    (단일 호출·leakage 단위테스트) 기존과 동일하게 전체 rows를 한 fold로 취급 — 하위호환.
    """
    ds = _load_dataset("real", None)
    daily = _load_real_daily(store_id)
    daily = add_calendar_features(daily, ds.calendar)
    daily["is_holiday"] = daily["is_public_holiday"].astype(bool)
    hourly_all = _load_real_receipts(set(daily["item_id"]))
    fold_groups = list(rows.groupby("fold")) if "fold" in rows.columns else [(0, rows)]
    parts = [_artisee_fold_order(store_id, daily, hourly_all, fold_rows)
             for _, fold_rows in fold_groups]
    return pd.concat(parts).reindex(rows.index).rename("artisee_order")


def _real_prospective_inputs(
    store_id: str, *, production_quantile: float = 0.85, val_weeks: int = 8, n_folds: int = 1,
    order_level: str = "item", alpha: float = DEFAULT_ALPHA,
    calibrate: bool = False, service_level: float = DEFAULT_SERVICE_LEVEL,
    cal_fold_frac: float = 0.5,
    category_margin: str = "quantile", nk_mult: float = 1.0, nk_add: float = 0.0,
    total_model: str = "lightgbm", event_prior: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """실데이터 조립: bonavi_daily + 재고정보(생산량/폐기량) join +
    our_order=production-quantile backtest 예측(최근 n_folds×val_weeks만 채움, Task C).
    order_level="category"면 v4 카테고리 총합→배분 경로(_category_order_predictions) 사용
    (총량 마진 = category_margin: quantile|nk|conformal).
    calibrate=True(+order_level="item")면 conformal 보정 발주(_conformal_order_predictions) 사용."""
    daily = _load_real_daily(store_id)
    daily = build_item_adjusted_demand(daily, alpha=alpha)
    inventory = load_inventory(REAL_INVENTORY_XLSX_PATH, store_id)
    inventory, waste_report = handle_negative_waste(inventory, policy="clip")
    console.print(f"[cyan]negative waste[/] clipped: {waste_report}")
    rows = _assemble_real_rows(daily, inventory, bulk_qty=_real_bulk_qty())
    if order_level == "category":
        predictions = _category_order_predictions(
            store_id, production_quantile=production_quantile, val_weeks=val_weeks,
            n_folds=n_folds, alpha=alpha, margin_method=category_margin,
            nk_mult=nk_mult, nk_add=nk_add,
            service_level=service_level, cal_fold_frac=cal_fold_frac,
            total_model=total_model, event_prior=event_prior,
        )
    elif calibrate:
        predictions = _conformal_order_predictions(
            store_id, service_level=service_level, val_weeks=val_weeks,
            n_folds=n_folds, cal_fold_frac=cal_fold_frac, alpha=alpha,
        )
    else:
        predictions = _our_order_predictions(
            store_id, production_quantile=production_quantile, val_weeks=val_weeks,
            n_folds=n_folds, alpha=alpha,
        )
    rows = _fill_our_order(rows, predictions)
    receipts = _load_real_receipts(set(rows["item_id"]))
    unit_prices = _load_unit_prices(REAL_INVENTORY_XLSX_PATH)
    return rows, receipts, unit_prices


def _load_prospective_inputs(
    source: str, store_id: str, *,
    production_quantile: float = 0.85, val_weeks: int = 8, n_folds: int = 1,
    order_level: str = "item", alpha: float = DEFAULT_ALPHA,
    calibrate: bool = False, service_level: float = DEFAULT_SERVICE_LEVEL,
    cal_fold_frac: float = 0.5,
    category_margin: str = "quantile", nk_mult: float = 1.0, nk_add: float = 0.0,
    total_model: str = "lightgbm", event_prior: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """(rows, receipts, unit_prices) 반환. production_quantile/val_weeks/n_folds/order_level/alpha/
    calibrate/service_level/cal_fold_frac/category_margin/nk_*/total_model/event_prior은 real 소스만 사용."""
    if source == "synthetic":
        return _synthetic_prospective_inputs()
    if source == "real":
        return _real_prospective_inputs(
            store_id, production_quantile=production_quantile, val_weeks=val_weeks,
            n_folds=n_folds, order_level=order_level, alpha=alpha,
            calibrate=calibrate, service_level=service_level, cal_fold_frac=cal_fold_frac,
            category_margin=category_margin, nk_mult=nk_mult, nk_add=nk_add,
            total_model=total_model, event_prior=event_prior,
        )
    raise ValueError(f"unknown source: {source!r} (expected 'synthetic' or 'real')")


def _stockout_item_days(rows: pd.DataFrame) -> set:
    """is_stockout==True로 관측된 (item_id, date) 키 집합 — arrival profile에서 제외."""
    if "is_stockout" not in rows.columns:
        return set()
    observed = rows[rows["is_stockout"].astype(bool)]
    return set(zip(observed["item_id"].astype(str), observed["date"].astype(str)))


def _decoupling_by_category(rows: pd.DataFrame) -> dict[str, float]:
    """카테고리별 ρ_DS (Decoupling Score) 산출.

    각 카테고리 내 item-day들에 대해 (adjusted_demand, is_stockout) 상관을 계산.
    반환: {category_id: score}
    """
    scores = {}
    for cat_id, group in rows.groupby("category_id"):
        demand = group["adjusted_demand"].to_numpy()
        stockout = group["is_stockout"].astype(float).to_numpy()
        score = decoupling_score(demand, stockout)
        scores[cat_id] = score
    return scores


@app.command("prospective-eval")
def cmd_prospective_eval(
    source: str = typer.Option("synthetic", help="synthetic | real"),
    store_id: str = typer.Option("store_gw01", help="real 소스는 store_mapping의 store_gw01 등 코드 사용"),
    open_hour: int = typer.Option(8),
    close_hour: int = typer.Option(22),
    production_quantile: float = typer.Option(
        0.85, help="our_order production quantile α (real 소스만 사용)"
    ),
    our_order_val_weeks: int = typer.Option(
        8, help="our_order backtest 검증(최근) 기간(주) — real 소스만 사용, 5년 전체 대신 최근 구간으로 제한"
    ),
    n_folds: int = typer.Option(1, help="full-window 회고 fold 수(real 소스). 1=단일창(기존)"),
    order_level: str = typer.Option(
        "item", help="item(기존 v2 LGBM) | category(v4 총합→배분)",
        click_type=click.Choice(["item", "category"]),
    ),
    alpha: float = typer.Option(
        DEFAULT_ALPHA, help="adjusted_demand의 마감할인 실수요 비율 α (real 소스만 사용)"
    ),
    calibrate: bool = typer.Option(
        False, help="conformal 보정 발주 사용(real+item 경로). base median + cross-fold half-split"
    ),
    target_service_level: float = typer.Option(
        DEFAULT_SERVICE_LEVEL, help="conformal 목표 서비스레벨 s (초과율 목표=1−s, 기본 0.74)"
    ),
    cal_fold_frac: float = typer.Option(
        0.5, help="앞쪽 folds 중 calibration 비율(나머지=test). conformal 계열(item calibrate·category conformal) 공용"
    ),
    category_margin: str = typer.Option(
        "quantile",
        help="order_level=category 총량 마진: quantile(production-q 직접) | nk(base_prod×mult+add 수동버퍼) | "
             "conformal(base median+q_s×scale, 목표 s=--target-service-level). real+category 전용",
        click_type=click.Choice(["quantile", "nk", "conformal"]),
    ),
    nk_mult: float = typer.Option(1.0, help="category-margin=nk의 곱셈 계수 K (order=base_prod×K+N)"),
    nk_add: float = typer.Option(0.0, help="category-margin=nk의 덧셈 버퍼 N (order=base_prod×K+N)"),
    total_model: str = typer.Option(
        "lightgbm",
        help="order_level=category 총량 base 학습 모델: lightgbm(CategoryTotalModel, 기존) | "
             "distributional(NGBoost LogNormal, σ(x) 학습). distributional의 spread 이득은 "
             "category-margin=quantile에서만 흐른다(nk/conformal은 median 기반). order_level=category 전용.",
        click_type=click.Choice(["lightgbm", "distributional"]),
    ),
    event_prior: bool = typer.Option(
        True, "--event-prior/--no-event-prior",
        help="order_level=category에서 sharp 캘린더 이벤트(xmas/설/추석 등) 레벨-앵커 블렌드 적용. "
             "모델 무관(lightgbm·distributional 공통) post-model 레이어. 기본 on. item 경로는 미적용.",
    ),
    baseline: str = typer.Option(
        "proxy",
        help="proxy(기존 base_order=production_qty proxy) | artisee(ArtiseeBaseline 재구현 제시량). "
             "artisee는 real 소스만 지원 — synthetic 입력엔 daily/hourly 스키마(sold_units/"
             "is_holiday/stockout_time)가 없음.",
        click_type=click.Choice(["proxy", "artisee"]),
    ),
    out_csv: str = typer.Option("reports/prospective_kpi.csv"),
) -> None:
    """우리 발주 추천 vs 현행 발주를 KPI(폐기/매진시각/매진률)로 비교.

    실 데이터(--source real)의 경우 카테고리별 ρ_DS(Decoupling Score) 진단도 출력.
    합성 데이터는 category_id가 없어 ρ_DS 미계산. --n-folds>1이면 fold별 Δ KPI +
    95%CI 집계도 출력·저장한다. --baseline artisee면 현행 baseline 대신 ArtiseeBaseline
    재구현 제시량을 base_order로 교체해 비교한다(real 소스 전용).
    """
    if baseline == "artisee" and source != "real":
        raise typer.BadParameter(
            "--baseline artisee는 --source real 전용이다 — synthetic 입력(_synthetic_prospective_inputs)"
            "엔 ArtiseeBaseline.fit이 요구하는 daily/hourly 스키마(store_id/sold_units/is_holiday/"
            "stockout_time, item_id/date/hour/qty)가 없다."
        )
    if total_model == "distributional" and order_level != "category":
        raise typer.BadParameter(
            "--total-model distributional은 --order-level category 전용이다 — "
            "DistributionalTotalModel은 카테고리 총량 base(CategoryTotalModel drop-in)로만 배선돼 있다. "
            "item 경로(v2 LGBM)는 미지원."
        )
    rows, receipts, unit_prices = _load_prospective_inputs(
        source, store_id,
        production_quantile=production_quantile, val_weeks=our_order_val_weeks, n_folds=n_folds,
        order_level=order_level, alpha=alpha,
        calibrate=calibrate, service_level=target_service_level, cal_fold_frac=cal_fold_frac,
        category_margin=category_margin, nk_mult=nk_mult, nk_add=nk_add,
        total_model=total_model, event_prior=event_prior,
    )
    if baseline == "artisee":
        rows = rows.assign(base_order=_artisee_baseline_order(store_id, rows))
    profiles = build_arrival_profile(
        receipts, group_cols=["item_id"],
        exclude_keys=_stockout_item_days(rows), exclude_cols=["item_id", "date"],
    )
    sh = StoreHours(store_id, open_hour, close_hour)
    # 평가 잣대: real은 마감할인 실수요 adjusted_demand, synthetic은 실 closing 데이터가
    # 없어 potential_demand 유지(Task 5 — 발주는 이미 Task 4에서 adjusted 학습).
    demand_col = "adjusted_demand" if source == "real" else "potential_demand"
    our = simulate_item_day_kpis(rows, profiles, order_col="our_order",
                                 store_hours=sh, group_cols=["item_id"],
                                 unit_prices=unit_prices, demand_col=demand_col)
    base = simulate_item_day_kpis(rows, profiles, order_col="base_order",
                                  store_hours=sh, group_cols=["item_id"],
                                  unit_prices=unit_prices, demand_col=demand_col)
    table = compare_policies(our, base)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    console.print(table.to_string(index=False))
    console.print(
        f"[cyan]예측 편향 WPE="
        f"{wpe(rows[demand_col].to_numpy(), rows['our_order'].to_numpy()):.3f}[/]"
    )
    if source == "real" and "category_id" in rows.columns:
        rho_ds = _decoupling_by_category(rows)
        console.print("[cyan]카테고리별 ρ_DS (Decoupling Score)[/]")
        for cat_id in sorted(rho_ds.keys()):
            score = rho_ds[cat_id]
            console.print(f"  {cat_id}: {score:.4f}")
    if source == "real" and n_folds > 1 and "fold" in our.columns:
        per_fold = compare_policies_by_fold(our, base)
        metric_cols = ["waste_cost_krw", "lost_margin_krw", "stockout_rate", "soldout_median_h"]
        agg = aggregate_fold_kpis(per_fold, metric_cols)
        console.print(per_fold.to_string(index=False))
        console.print(agg.to_string(index=False))
        per_fold.to_csv(out_path.with_name("prospective_kpi_per_fold.csv"), index=False)
        agg.to_csv(out_path.with_name("prospective_kpi_agg.csv"), index=False)
    if source == "real":
        exceed = quantile_exceedance_rate(
            rows["adjusted_demand"].to_numpy(), rows["our_order"].to_numpy()
        )
        nominal_label, nominal_value = (
            ("s", 1 - target_service_level) if calibrate else ("α", 1 - production_quantile)
        )
        console.print(f"[cyan]calibration[/] 초과율 P(demand>order)={exceed:.3f} "
                      f"(nominal 1−{nominal_label}={nominal_value:.2f})")
        console.print(f"[cyan]waste sanity[/] {compare_actual_vs_simulated_waste(rows, base)}")
    if source == "real" and order_level == "category":
        by_date = rows.groupby("date").agg(
            pd_sum=("adjusted_demand", "sum"), order_sum=("our_order", "sum"),
        )
        cat_exceed = float((by_date["pd_sum"] > by_date["order_sum"]).mean())
        cat_wape = wape(by_date["pd_sum"].to_numpy(), by_date["order_sum"].to_numpy())
        cat_wpe = wpe(by_date["pd_sum"].to_numpy(), by_date["order_sum"].to_numpy())
        console.print(
            f"[cyan]category calibration[/] 초과율 P(Σdemand>Σorder)={cat_exceed:.3f} "
            f"(nominal 1−q={1 - production_quantile:.2f}) | WAPE={cat_wape:.3f} WPE={cat_wpe:+.3f}, "
            f"{len(by_date)} dates"
        )
    console.print(f"[green]wrote[/] {out_path}")


if __name__ == "__main__":
    app()
