from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from bakery.features.category_aggregate import build_category_daily, build_features
from bakery.harness.backtest_core import metrics_from_preds, windowed_backtest
from bakery.harness.config import ExperimentSpec
from bakery.harness.event_priors import resolve_event_priors
from bakery.harness.registry import is_runnable, build_forecaster

STAGES: tuple[str, ...] = ("features", "backtest", "evaluate")


@dataclass
class RunResult:
    name: str
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    metrics: dict
    resolved: dict


@dataclass
class ExperimentResult:
    name: str
    runs: dict[str, RunResult]
    comparison: pd.DataFrame


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

    feat_key = _stage_key({"source": spec.data.source, "store": spec.data.store,
                           "target": spec.target, "alpha": spec.alpha})

    def _feat():
        cd = build_category_daily(alpha=spec.alpha)
        return build_features(cd, target_col=spec.target)

    feat = _load_or_compute("features", feat_key, cache_dir, _feat, trace)
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
        bt = windowed_backtest(
            feat, window_days=spec.window.window_days, target_col=spec.target,
            n_folds=spec.window.n_folds, horizon_days=spec.window.horizon_days,
            production_q=spec.production_q, alpha=spec.alpha,
            events=events, lunar_events=lunar, forecaster=fc,
            lead_days=spec.window.lead_days, anchor_dow=spec.window.anchor_dow,
            # gapless(dropna 이전) 프레임을 넘긴다 — AR 재계산은 위치 shift라 연속성이 필수.
            ar_history=feat[["date", spec.target]] if spec.window.align_features else None,
        )
        metrics = metrics_from_preds(bt.predictions)
        fout = out / fname
        fout.mkdir(parents=True, exist_ok=True)
        bt.predictions.to_csv(fout / "predictions.csv", index=False)
        bt.folds.to_csv(fout / "fold_results.csv", index=False)
        (fout / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        runs[fname] = RunResult(name=fname, predictions=bt.predictions,
                                fold_metrics=bt.folds, metrics=metrics, resolved=resolved)
        rows.append({"forecaster": fname, **metrics})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "comparison.csv", index=False)
    return ExperimentResult(name=spec.name, runs=runs, comparison=comparison)
