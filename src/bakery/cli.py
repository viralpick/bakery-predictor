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
from .data.synthetic import generate_synthetic_bundle
from .data.weather import load_weather_forecast_from_local
from .evaluation.backtest import aggregate_by_model, per_category_wape, run_backtest
from .evaluation.classifier_metrics import base_rate, precision_at_k, recall_at_k, roc_auc
from .evaluation.split import apply_split, generate_time_splits
from .features.calendar_features import add_calendar_features
from .features.weather_features import add_weather_features
from .ingest import calendar_api, forecast_api, weather_api
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
    console.print(
        f"[green]wrote[/] hourly={len(bundle.hourly):,} daily={len(bundle.daily):,} "
        f"weather={len(bundle.weather):,} calendar={len(bundle.calendar):,} → {out_dir}"
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
    out_dir: Path = REPORTS_DIR,
) -> None:
    """Compare baselines + LightGBM variants on the same rolling folds."""
    variant_list = _parse_variants(variants)
    ds = _load_dataset(source, data_dir)
    daily = _enrich_if_needed(ds, variant_list)
    windows = generate_time_splits(
        daily["date"], n_splits=n_splits, val_horizon_days=horizon_days, step_days=step_days
    )
    forecasters = _build_forecasters(variant_list)
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
    if feature_set in {"v1", "v2"}:
        forecast_weather = _load_forecast_weather(horizon) if use_forecast else None
        target = _enrich_target(target, ds, forecast_weather=forecast_weather)
    yhat = forecaster.predict(target)
    demand_col = "yhat_potential_demand" if feature_set == "v2" else "yhat_sold_units"
    target = target.assign(**{demand_col: yhat.round(2).to_numpy(), "model": forecaster.name})

    if feature_set == "v2":
        prod_params = LGBMParams(objective="quantile", alpha=production_quantile)
        prod_model = GlobalLGBM(feature_set="v2", params=prod_params).fit(daily)
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
    """Return a daily frame enriched with calendar/weather iff any v1+ variant is requested."""
    if not any(v in {"v1", "v2"} for v in variants):
        return ds.daily
    enriched = add_calendar_features(ds.daily, ds.calendar)
    enriched = add_weather_features(enriched, ds.weather)
    return enriched


def _enrich_target(
    target: pd.DataFrame, ds: DailyDataset, *, forecast_weather: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Merge calendar (future-safe) and weather onto horizon dates. When a forecast
    frame is provided, use it; otherwise fall back to observed weather (PoC convenience)."""
    target = add_calendar_features(target, ds.calendar)
    weather_frame = forecast_weather if forecast_weather is not None else ds.weather
    target = add_weather_features(target, weather_frame)
    return target


def _load_forecast_weather(horizon: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Build a horizon weather frame from forecast_* parquet files.

    PoC: assumes all stores share the same grid/region (Seoul). When per-store
    differentiation lands (#4), this needs to loop over distinct (nx, ny) cells
    and return a long-form frame keyed by store_id.
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
    sample_store = next(iter(mapping.values()))
    return load_weather_forecast_from_local(
        short_daily_path=short_p,
        mid_daily_path=mid_p,
        observed_parquet_path=observed_p,
        station_id=sample_store["station_id"],
        nx=sample_store["nx"],
        ny=sample_store["ny"],
        mid_land_reg_id=sample_store["mid_land_reg_id"],
        mid_ta_reg_id=sample_store["mid_ta_reg_id"],
        horizon_start=horizon[0],
        horizon_end=horizon[-1],
    )


def _build_forecasters(variants: list[str]):
    """Build a baseline + LightGBM-per-variant list."""
    forecasters = [SeasonalNaive(n_weeks=4), MovingAverage(window=28)]
    for v in variants:
        forecasters.append(GlobalLGBM(feature_set=v))
    return forecasters


def _model_to_feature_set(model: str) -> str | None:
    """For predict-next-week: map model name → feature_set; baselines return None."""
    if model == "lightgbm":
        return "v0"
    if model == "lightgbm_v1":
        return "v1"
    if model == "lightgbm_v2":
        return "v2"
    return None


def _pick_model(name: str):
    table = {
        "seasonal_naive": SeasonalNaive(n_weeks=4),
        "moving_average": MovingAverage(window=28),
        "lightgbm": GlobalLGBM(feature_set="v0"),
        "lightgbm_v1": GlobalLGBM(feature_set="v1"),
        "lightgbm_v2": GlobalLGBM(feature_set="v2"),
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


if __name__ == "__main__":
    app()
