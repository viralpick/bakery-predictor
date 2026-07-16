"""모델이 커버 못 하는 이상치 탐지 — 우리 예측 모델의 OOS 잔차 기반 (원본 통계 아님).

목적: 캘린더·날씨·외부데이터를 다 보는 우리 총량 예측 모델(LightGBM + 특수일 prior)이
그럼에도 **크게 빗나간 날/주/월**을 찾아 고객사에 "이 날 무슨 일이 있었나" 문의한다.
→ 남은 큰 잔차 = 모델 피처로 설명 안 되는 외부 사건(휴업/공사/지역이벤트/POS오류 등) 후보.

방법:
- windowed_backtest(광교, 730d rolling, HORIZON=7)로 과거 3년 OOS 예측(date,actual,expected).
  actual = adjusted_demand_unit(정상+0.8×마감, bulk 제외) = 모델 target.
- 잔차 resid = actual − expected. robust z = 0.6745·(resid−median)/MAD.
- 일별: |z| ≥ Z_STRONG 튀는 날 (LOW=판매 저조 / HIGH=급등).
- 주/월별: Σresid/Σactual (WPE) robust z → "특정 주·월 전체적으로 낮음/높음".
- 휴일(calendar_raw)·요일 enrich → 모델이 이미 아는 효과(휴일·요일)는 걸러 판단.

실행: PYTHONPATH=scripts uv run --with matplotlib python scripts/anomaly_detect_model_residuals.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from store_predictive_power import (
    build_store_data, windowed_backtest, STORE_EVENT_PRIORS,
    DEFAULT_WINDOW_DAYS, HORIZON,
)

CD_CODE, STORE_ID, LABEL = "1000000047", "store_gw01", "광교"
YEARS = 3
N_FOLDS = 160                 # 160×7 ≈ 3.07년 OOS test span (여유분, 아래서 3년으로 컷)
Z_STRONG = 3.5                # 강한 이상치 (robust z, p≈0.0005)
Z_WEAK = 3.0                  # 참고 tier
CALENDAR_PARQUET = "data/external/calendar_raw.parquet"
DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]


def _robust_z(x: np.ndarray) -> np.ndarray:
    """MAD 기반 robust z-score. 이상치가 스케일을 부풀리지 않도록 median/MAD 사용."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        sd = x.std()
        return (x - med) / sd if sd > 0 else np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def _load_holidays() -> pd.DataFrame:
    cal = pd.read_parquet(CALENDAR_PARQUET)
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    return cal[["date", "name", "is_holiday"]].drop_duplicates("date")


def _predictions_3y() -> pd.DataFrame:
    """광교 headline 모델(events=XMAS+CHUSEOK prior)로 3년 OOS 예측."""
    sd = build_store_data(CD_CODE, STORE_ID, LABEL)
    cfg = STORE_EVENT_PRIORS.get(LABEL, {})
    res = windowed_backtest(
        sd.feat, window_days=DEFAULT_WINDOW_DAYS, n_folds=N_FOLDS,
        horizon_days=HORIZON, events=cfg.get("events"), lunar_events=cfg.get("lunar_events"),
    )
    p = res.predictions.copy()
    p["date"] = pd.to_datetime(p["date"])
    cutoff = p["date"].max() - pd.DateOffset(years=YEARS)
    p = p[p["date"] > cutoff].sort_values("date").reset_index(drop=True)
    p["resid"] = p["actual"] - p["expected"]          # 음수 = 실제가 예측보다 낮음(판매 저조)
    p["pct"] = p["resid"] / p["expected"].clip(lower=1) * 100
    p["z"] = _robust_z(p["resid"].to_numpy())
    p.to_csv("reports/anomaly_full_preds.csv", index=False)  # 전체 OOS preds(재run 없이 재분석용)
    return p


def _day_anomalies(p: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    d = p[p["z"].abs() >= Z_WEAK].copy()
    d = d.merge(holidays, on="date", how="left")
    d["is_holiday"] = d["is_holiday"].fillna(False)
    d["dow"] = d["date"].dt.dayofweek.map(lambda i: DOW_KR[i])
    d["direction"] = np.where(d["resid"] < 0, "LOW(저조)", "HIGH(급등)")
    d["tier"] = np.where(d["z"].abs() >= Z_STRONG, "강", "참고")
    return d.sort_values("z", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def _period_anomalies(p: pd.DataFrame, freq: str) -> pd.DataFrame:
    """주(freq='W-MON')/월(freq='MS') 단위 Σresid/Σactual(WPE) robust z."""
    key = p["date"].dt.to_period("W" if freq == "W" else "M")
    g = p.groupby(key).agg(
        start=("date", "min"), end=("date", "max"), n_days=("date", "size"),
        actual=("actual", "sum"), expected=("expected", "sum"), resid=("resid", "sum"),
    ).reset_index(drop=True)
    g = g[g["n_days"] >= (4 if freq == "W" else 20)]  # 불완전 기간 제외
    g["wpe_pct"] = g["resid"] / g["actual"].clip(lower=1) * 100
    g["z"] = _robust_z(g["wpe_pct"].to_numpy())
    g["direction"] = np.where(g["resid"] < 0, "LOW(저조)", "HIGH(급등)")
    out = g[g["z"].abs() >= Z_WEAK].copy()
    return out.sort_values("z", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def main() -> None:
    p = _predictions_3y()
    holidays = _load_holidays()
    span = f"{p['date'].min().date()} ~ {p['date'].max().date()}"
    wape = float(p["resid"].abs().sum() / max(p["actual"].abs().sum(), 1)) * 100
    print(f"[광교 3년 OOS] {len(p)}일 ({span}), 모델 WAPE={wape:.2f}%  "
          f"잔차 median={p['resid'].median():.1f} MAD-scale로 z 산출")

    day = _day_anomalies(p, holidays)
    wk = _period_anomalies(p, "W")
    mo = _period_anomalies(p, "M")

    day.to_csv("reports/anomaly_days.csv", index=False)
    wk.to_csv("reports/anomaly_weeks.csv", index=False)
    mo.to_csv("reports/anomaly_months.csv", index=False)

    n_strong = int((day["tier"] == "강").sum())
    print(f"\n일별 이상치: |z|≥{Z_WEAK} {len(day)}일 (그중 강 |z|≥{Z_STRONG}: {n_strong}일) · "
          f"LOW {int((day['resid']<0).sum())} / HIGH {int((day['resid']>0).sum())}")
    print(day.head(25)[["date", "dow", "actual", "expected", "pct", "z", "direction", "tier", "is_holiday", "name"]].to_string(index=False))

    print(f"\n주별 이상치: {len(wk)}주")
    print(wk[["start", "end", "n_days", "actual", "expected", "wpe_pct", "z", "direction"]].to_string(index=False))

    print(f"\n월별 이상치: {len(mo)}개월")
    print(mo[["start", "end", "n_days", "actual", "expected", "wpe_pct", "z", "direction"]].to_string(index=False))

    print("\nwrote reports/anomaly_{days,weeks,months}.csv")
    print("※ actual=adjusted_demand(모델 target) · pct=(실제−예측)/예측% · z=robust(MAD).")
    print("※ 휴일/요일 enrich: 모델이 이미 아는 효과(is_holiday=True 등)는 문의 우선순위 낮춤.")


if __name__ == "__main__":
    main()
