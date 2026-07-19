"""트랙4 진단 — 극한날씨에서 모델이 nonlinear 꺾임을 못 잡아 체계적 편향이 남는가.

배경: 이상치 분석 4트랙 중 4번(극한날씨 nonlinear). 신호: 큰 잔차 이상치 20일 중 극한날씨
2건(2024-08-11 폭염34도·매진, 2024-01-22 눈). LightGBM은 트리라 온도/강수를 nonlinear로
쓸 수 있고 feature importance서 날씨(습도·운량·풍속·온도)가 이미 상위(트랙3 확인). 그럼에도
**극한 구간**(폭염/한파/폭우)에 체계적 잔차 편향이 남으면 nonlinear 보강 여지.

★게이트0(표본): 극한 빈도 확인 완료 — 폭염 maxTa≥33 61일·한파 minTa≤-10 27일·강한비 sumRn≥30
  45일(진단 가능). 강풍 avgWs≥5 2일 → 표본 부족 제외.

★교란 주의: 폭염=여름·한파=겨울에만 발생. 트랙3서 여름 WPE≈0였으므로 폭염을 "전체 나머지"와
  비교하면 계절 교락 → 반드시 **같은 계절 비극한일**과 비교(폭염 vs 여름 비폭염 등).

2레이어(원신호가 폭염 "매진"이라 발주층도 반드시 CI 검정):
  - 예측층: WPE by 온도/강수 bin (극한 bin만 튀나=nonlinear 미스) + 사전등록 대비(동계절 통제, CI)
  - 발주층: 매진율(actual>production) 극한 vs 동계절비극한 CI (raw 매진%는 계절 교락 → 통제 필수)
  - spike(robust |z|≥3) 분리: 산발 극한일 vs 체계적 극한 편향
게이트2(편향 유의 시): 극한 조건부 마진 vs 전역 iso-waste. 본 진단서 두 층 다 신호 없어 미실행.

WPE = (expected−actual)/Σ|actual|. 음수=과소예측(발주부족 방향).

실행: PYTHONPATH=scripts uv run python scripts/track4_weather_diagnose.py
      (track3_fresh_preds.parquet 필요 — track3_seasonal_diagnose.py가 생성)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PREDS = Path("reports/track3_fresh_preds.parquet")
WEATHER = Path("data/external/weather_observed.parquet")
STATION = 119                        # 수원 = 광교
N_BOOT = 2000
SPIKE_Z = 3.0
SEED = 42
# 사전등록 극한 정의 (게이트0 빈도 확인 완료)
HEAT_MAXTA, COLD_MINTA, RAIN_SUMRN = 33.0, -10.0, 30.0
SUMMER, WINTER = (6, 7, 8, 9), (12, 1, 2)   # 폭염·한파 동계절 비교군 정의


def _load() -> pd.DataFrame:
    w = pd.read_parquet(WEATHER)
    w = w[w["station_id"] == STATION].copy()
    w["date"] = pd.to_datetime(w["date"])
    for c in ["maxTa", "minTa", "avgTa", "sumRn", "avgWs"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    p = pd.read_parquet(PREDS)
    p["date"] = pd.to_datetime(p["date"])
    d = p.merge(w[["date", "maxTa", "minTa", "avgTa", "sumRn", "avgWs"]], on="date", how="left")
    d["month"] = d["date"].dt.month
    d["resid"] = d["actual"] - d["expected"]
    med = np.median(d["resid"])
    mad = np.median(np.abs(d["resid"] - med)) or d["resid"].std()
    d["resid_z"] = 0.6745 * (d["resid"] - med) / mad if mad else 0.0
    return d


def _wpe(sub: pd.DataFrame) -> float:
    denom = sub["actual"].abs().sum()
    return float((sub["expected"] - sub["actual"]).sum() / denom * 100) if denom else 0.0


def _stockout_rate(sub: pd.DataFrame) -> float:
    """q0.85 버퍼발주(production) 뚫린 전체매진 비율 %."""
    return float((sub["actual"] > sub["production"]).mean() * 100)


def _boot_contrast(a: pd.DataFrame, b: pd.DataFrame, metric, rng: np.random.Generator) -> tuple:
    """metric(a)−metric(b) day-level 부트스트랩 95%CI. a=극한 세그먼트, b=동계절 비극한."""
    ai, bi = a.index.to_numpy(), b.index.to_numpy()
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ra = a.loc[rng.choice(ai, len(ai), replace=True)]
        rb = b.loc[rng.choice(bi, len(bi), replace=True)]
        diffs[i] = metric(ra) - metric(rb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return metric(a) - metric(b), float(lo), float(hi)


def _bin_wpe(d: pd.DataFrame, col: str, bins: list, labels: list[str]) -> None:
    d = d.copy()
    d["_bin"] = pd.cut(d[col], bins=bins, labels=labels)
    rows = []
    for name, sub in d.groupby("_bin", observed=True):
        rows.append(dict(구간=str(name), n=len(sub), WPE_pct=round(_wpe(sub), 2),
                         매진pct=round((sub["actual"] > sub["production"]).mean() * 100, 1)))
    print(pd.DataFrame(rows).to_string(index=False))


def _extreme_contrast(d: pd.DataFrame, seg_mask, comp_mask, name: str,
                      rng: np.random.Generator) -> None:
    """극한 세그먼트 vs 동계절 비극한 비교군. 예측층(WPE)·발주층(매진율) 둘 다 CI + spike 분리.
    (원신호 = 폭염 매진 이벤트라 발주층도 동계절 통제 CI로 검정 — raw 매진%는 계절 교락됨.)"""
    seg = d[seg_mask]
    comp = d[comp_mask]
    if len(seg) < 5:
        print(f"\n[{name}] n={len(seg)} <5 → 표본 부족, skip")
        return

    def _report(metric, unit: str, label: str) -> None:
        diff, lo, hi = _boot_contrast(seg, comp, metric, rng)
        verdict = "★ CI 0배제 (real)" if (lo > 0 or hi < 0) else "0포함 (noise)"
        print(f"   {label}: 극한={metric(seg):.2f}{unit} vs 동계절비극한={metric(comp):.2f}{unit}  "
              f"diff={diff:+.2f}  95%CI=[{lo:+.2f}, {hi:+.2f}]  → {verdict}")

    print(f"\n[{name}] n={len(seg)} (비교군 n={len(comp)})")
    _report(_wpe, "%", "예측층 WPE   ")
    _report(_stockout_rate, "%", "발주층 매진율")
    calm = seg[seg["resid_z"].abs() < SPIKE_Z]
    print(f"   WPE spike분리: 전체={_wpe(seg):.2f}% → spike({len(seg)-len(calm)}일) 제거 후={_wpe(calm):.2f}%")


def main() -> None:
    d = _load()
    rng = np.random.default_rng(SEED)
    print(f"[광교 3년 OOS] {len(d)}일  전체 WPE={_wpe(d):.2f}%")

    print("\n=== 예측층 WPE by maxTa bin (극한 bin만 튀면 nonlinear 미스) ===")
    _bin_wpe(d, "maxTa", [-100, 0, 10, 20, 28, 31, 33, 100],
             ["<0", "0-10", "10-20", "20-28", "28-31", "31-33", "≥33(폭염)"])
    print("\n=== 예측층 WPE by sumRn bin (비 안온날/약/강/폭우) ===")
    _bin_wpe(d, "sumRn", [-1, 0, 5, 20, 30, 1e9], ["0(무강수)", "0-5", "5-20", "20-30", "≥30(강한비)"])

    print("\n=== 사전등록 극한 대비 (동계절 통제, 부트스트랩 CI) ===")
    heat, summer = d["maxTa"] >= HEAT_MAXTA, d["month"].isin(SUMMER)
    cold, winter = d["minTa"] <= COLD_MINTA, d["month"].isin(WINTER)
    rain = d["sumRn"] >= RAIN_SUMRN
    _extreme_contrast(d, heat, summer & ~heat, f"폭염(maxTa≥{HEAT_MAXTA:.0f}) vs 여름비폭염", rng)
    _extreme_contrast(d, cold, winter & ~cold, f"한파(minTa≤{COLD_MINTA:.0f}) vs 겨울비한파", rng)
    _extreme_contrast(d, rain, (d["sumRn"] > 0) & ~rain, f"강한비(sumRn≥{RAIN_SUMRN:.0f}) vs 약한비", rng)

    print("\n※ WPE 음수=과소예측(발주부족 방향). 극한 bin만 튀고 CI 0배제+spike후 잔존 = nonlinear 보강 여지.")
    print("※ 게이트0: 강풍 avgWs≥5 2일 표본부족 제외.")


if __name__ == "__main__":
    main()
