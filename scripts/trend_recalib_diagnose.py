"""추세 recalibration 진단 — window×연도 편향(WPE) 매트릭스 (광교 3년 OOS).

트랙1 가설: 광교 빵류 수요가 2022 정점 후 하락 → 730d 롤링 학습이 하락을 지연추종
→ 하락기(특히 2023) 상시 과대예측(WPE>0). 짧은 창이 이 편향을 줄이는가?

핵심: aggregate WAPE(긴 창 유리, 옛 결론 재탕)가 아니라 **연도별 signed WPE**로 판정.
predictions frame(date/actual/expected)에서 직접 연도별 지표를 계산한다.

레버(a) 창 단축 테스트 전용. 레버(c) 온라인 보정은 별도 스크립트.

실행: PYTHONPATH=scripts uv run python scripts/trend_recalib_diagnose.py
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path

import numpy as np
import pandas as pd

from store_predictive_power import (
    build_store_data,
    windowed_backtest,
    STORE_EVENT_PRIORS,
    TARGET,
)

CD_CODE, STORE_ID, LABEL = "1000000047", "store_gw01", "광교"
N_FOLDS = 160  # 160×7 ≈ 3.07년 OOS (이상치 분석과 동일)
CANDIDATE_WINDOWS = (180, 365, 730, 1095, 1825)
OOS_YEARS = (2023, 2024, 2025)  # 하락 전환기 = 2023
OUT_CSV = Path("reports/trend_recalib_window_by_year.csv")


def year_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """predictions → 연도별 WPE(=Σ(exp−act)/Σact, 부호=편향)·WAPE·n.

    WPE>0 = 과대예측(하락 지연추종의 신호). |WPE| 축소가 판정 기준.
    """
    p = preds.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["year"] = p["date"].dt.year
    rows = []
    for year, g in p.groupby("year"):
        act = g["actual"].to_numpy()
        exp = g["expected"].to_numpy()
        denom = max(np.abs(act).sum(), 1.0)
        rows.append(
            dict(
                year=int(year),
                n=int(len(g)),
                wpe=float((exp - act).sum() / denom),
                wape=float(np.abs(act - exp).sum() / denom),
            )
        )
    return pd.DataFrame(rows)


def run() -> None:
    print(f"[{LABEL}] build_store_data ...")
    sd = build_store_data(CD_CODE, STORE_ID, LABEL)
    cfg = STORE_EVENT_PRIORS[LABEL]

    all_rows = []
    for window_days in CANDIDATE_WINDOWS:
        print(f"[{LABEL}] windowed_backtest window={window_days}d n_folds={N_FOLDS} ...")
        res = windowed_backtest(
            sd.feat,
            window_days=window_days,
            n_folds=N_FOLDS,
            target_col=TARGET,
            events=cfg.get("events"),
            lunar_events=cfg.get("lunar_events"),
        )
        ym = year_metrics(res.predictions)
        ym["window_days"] = window_days
        all_rows.append(ym)

    result = pd.concat(all_rows, ignore_index=True)
    result = result[result["year"].isin(OOS_YEARS)].copy()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    _print_matrix(result, "WPE (편향, +=과대예측)", "wpe")
    _print_matrix(result, "WAPE (정확도)", "wape")
    print(f"\n저장: {OUT_CSV}")


def _print_matrix(df: pd.DataFrame, title: str, col: str) -> None:
    pivot = df.pivot(index="window_days", columns="year", values=col) * 100
    print(f"\n=== {title} (%) — 행=window(d), 열=연도 ===")
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    run()
