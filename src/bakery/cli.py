"""bakery CLI — generate-data / backtest / predict-next-week."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

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
from .evaluation.classifier_metrics import base_rate, precision_at_k, recall_at_k, roc_auc
from .evaluation.split import apply_split, generate_time_splits
from .features.calendar_features import add_calendar_features
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
from .ingest.store_mapping import load_store_mapping
from .models.lightgbm_regressor import VALID_FEATURE_SETS, GlobalLGBM, LGBMParams
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
) -> None:
    """Compare baselines + LightGBM variants on the same rolling folds."""
    variant_list = _parse_variants(variants)
    ds = _load_dataset(source, data_dir)
    daily = _enrich_if_needed(ds, variant_list)
    windows = generate_time_splits(
        daily["date"], n_splits=n_splits, val_horizon_days=horizon_days, step_days=step_days
    )
    forecasters = _build_forecasters(variant_list, include_production=include_production)
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
) -> None:
    """Train on all history; emit demand prediction + recommended production.

    v2: trains two LightGBMs — median (regression) for demand and a
    `production_quantile` (default 0.85) for recommended production. Newsvendor
    intuition: higher quantile = safer against stockouts, more potential waste.

    v1/v0 (legacy): fall back to the demand-times-margin heuristic with
    `recommended_production = yhat * (1 + base_safety_margin + risk_bonus * stockout_prob)`.
    """
    feature_set = _model_to_feature_set(model)
    ds = _load_dataset(source, data_dir)
    daily = _enrich_if_needed(ds, [feature_set]) if feature_set else ds.daily
    last = daily["date"].max()
    horizon = pd.date_range(last + pd.Timedelta(days=1), periods=7, freq="D")
    forecaster = _pick_model(model)
    forecaster.fit(daily)
    pairs = daily[["store_id", "item_id", "category_id"]].drop_duplicates()
    target = pairs.merge(pd.DataFrame({"date": horizon}), how="cross")
    if feature_set in {"v1", "v2", "v3"}:
        forecast_weather = _load_forecast_weather(horizon) if use_forecast else None
        target = _enrich_target(
            target, ds, forecast_weather=forecast_weather, include_external=(feature_set == "v3"),
        )
    yhat = forecaster.predict(target)
    demand_col = "yhat_potential_demand" if feature_set in {"v2", "v3"} else "yhat_sold_units"
    target = target.assign(**{demand_col: yhat.round(2).to_numpy(), "model": forecaster.name})

    if feature_set in {"v2", "v3"}:
        prod_params = LGBMParams(objective="quantile", alpha=production_quantile)
        prod_model = GlobalLGBM(feature_set=feature_set, params=prod_params).fit(daily)
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
) -> tuple[pd.DataFrame, str]:
    """Per-item demand point estimates for the next 7 days (v6 point estimate)."""
    feature_set = _model_to_feature_set(model)
    daily = _enrich_if_needed(ds, [feature_set]) if feature_set else ds.daily
    last = daily["date"].max()
    horizon = pd.date_range(last + pd.Timedelta(days=1), periods=7, freq="D")
    forecaster = _pick_model(model)
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
) -> None:
    """v6 산출물: 점추정 + 발주량 + 매진/폐기 위험 수치 + 결정 lineage.

    Forecast(LightGBM 점추정) → 결정론 정책(안전마진·반올림) → Monte-Carlo 위험.
    예측은 학습 코어, 이 레이어는 예측 *이후*의 결정/위험/lineage 껍질이다
    (docs/kinetic_layer_fit_analysis.md §8·§10). 예측 이후 단계라 leakage 없음.
    """
    ds = _load_dataset(source, data_dir)
    items, model_name = _demand_points_next_week(ds, model, use_forecast)
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
                       production_quantile: float = 0.85):
    """Build baseline + LightGBM-per-variant list, optionally adding quantile
    production models for v2/v3 (lightgbm_v2_q85 etc.)."""
    forecasters = [SeasonalNaive(n_weeks=4), MovingAverage(window=28)]
    for v in variants:
        forecasters.append(GlobalLGBM(feature_set=v))  # demand (median) model
        if include_production and v in {"v2", "v3"}:
            prod_params = LGBMParams(objective="quantile", alpha=production_quantile)
            forecasters.append(GlobalLGBM(feature_set=v, params=prod_params))
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
) -> None:
    """Production-model quantile α sweep — net_profit-최대 α 찾기.

    α 0.50 (median) ~ 0.95 범위 backtest + 사업 KPI 비교. 매장·카테고리별 최적 α 결정.
    """
    import warnings
    import numpy as np
    from .evaluation.business_metrics import (
        CostParams, asymmetric_loss, simulate_profit,
    )

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
    windows = generate_time_splits(
        daily["date"], n_splits=n_splits, val_horizon_days=horizon_days, step_days=step_days
    )

    forecasters = []
    for a in alpha_list:
        if a == 0.5:
            forecasters.append(GlobalLGBM(feature_set=variant))  # median (regression)
        else:
            params = LGBMParams(objective="quantile", alpha=a)
            forecasters.append(GlobalLGBM(feature_set=variant, params=params))

    console.print(
        f"[cyan]alpha-sweep[/] variant={variant} αs={alpha_list} "
        f"folds={len(windows)} horizon={horizon_days}d"
    )
    fold_df, pred_df = run_backtest(daily, forecasters, windows)

    # Inject potential_demand for true-demand-aware profit simulation
    if "potential_demand" in daily.columns:
        pd_lookup = daily.set_index(["store_id", "item_id", "date"])["potential_demand"]
        pred_df = pred_df.copy()
        pred_df["potential_demand"] = pred_df.set_index(
            ["store_id", "item_id", "date"]
        ).index.map(pd_lookup)

    rows = []
    for model, sub in pred_df.groupby("model"):
        asym = asymmetric_loss(sub["yhat"], sub["sold_units"], params=cost_params)
        profit = simulate_profit(sub, unit_prices=unit_prices, params=cost_params)
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
        simulate_profit,
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
        variant_list, include_production=True, production_quantile=production_quantile
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
    # Inject potential_demand into pred_df from the enriched daily so simulate_profit
    # can see the censoring-corrected target.
    if "potential_demand" in daily.columns:
        pd_lookup = daily.set_index(["store_id", "item_id", "date"])["potential_demand"]
        pred_df = pred_df.copy()
        pred_df["potential_demand"] = pred_df.set_index(
            ["store_id", "item_id", "date"]
        ).index.map(pd_lookup)
    biz_rows = []
    for model, sub in pred_df.groupby("model"):
        asym = asymmetric_loss(sub["yhat"], sub["sold_units"], params=cost_params)
        profit = simulate_profit(sub, unit_prices=unit_prices, params=cost_params)
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


if __name__ == "__main__":
    app()
