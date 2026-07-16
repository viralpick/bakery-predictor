"""안전마진(품절회피 버퍼) 발주로도 전체 매진이 난 날 — "그날 정보 부족" 후보.

앞의 잔차 이상치와 다른 렌즈: 여기선 우리 **버퍼 발주(production=q0.85 예측)**가
실제 카테고리 총수요보다 작았던 날(actual > production) = 안전마진을 뚫은 **전체 매진**.
버퍼로도 못 막았다 = 그날 수요가 우리가 가진 정보로 예측 가능한 범위를 넘었다.

- 전체 매진 = actual(카테고리 총 adjusted_demand) > production(q0.85 버퍼 발주). 1090일 OOS.
- 초과율 = (actual − production)/production. robust z(우측 tail)로 "큰 매진"(surprise) 판정.
- 2버킷: 명절(모델 prior 확장으로 커버 가능) vs 미설명(고객 문의 = 진짜 정보 부족).

"어쩔 수 없나?" 답:
- 얕은 매진(초과율 소): q0.85 설계상 ~15% 날은 초과 — 버퍼 상향으로 줄이나 폐기 trade-off(불가피 영역).
- 큰 매진·명절: prior 확장으로 커버(고칠 수 있음).
- 큰 매진·미설명: 그날 외부 정보 부족 = 고객 문의 대상.
- ⚠️ censoring: actual(adjusted)은 매진일 실수요 하한 → 실제 매진은 더 잦고 더 큼(여긴 하한).

실행: uv run python scripts/order_shortfall_days.py  (先 anomaly_detect_model_residuals.py로 full_preds 생성)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from reclassify_anomalies import _holiday_lookup, _nearest_holiday

FULL_PREDS = "reports/anomaly_full_preds.csv"
Z_SURPRISE = 3.0          # 우측 tail robust z (큰 매진)
DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        sd = x.std()
        return (x - med) / sd if sd > 0 else np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def main() -> None:
    p = pd.read_csv(FULL_PREDS)
    p["date"] = pd.to_datetime(p["date"])
    gap = p["actual"] - p["production"]                     # >0 = 발주<수요 = 전체 매진
    p["shortfall"] = gap
    p["exceed_pct"] = gap / p["production"].clip(lower=1) * 100
    p["gap_z"] = _robust_z(gap.to_numpy())                  # 우측 tail = surprise 매진

    n = len(p)
    stockout = p[p["shortfall"] > 0].copy()
    print(f"[광교 3년 OOS {n}일] 버퍼발주(q0.85) 기준 전체매진(actual>production): "
          f"{len(stockout)}일 ({len(stockout)/n*100:.1f}%)  "
          f"— q0.85 설계상 ~15% 초과는 예정된 얕은 매진")

    # 큰 매진(surprise): 우측 tail robust z
    big = stockout[stockout["gap_z"] >= Z_SURPRISE].copy()
    holidays = _holiday_lookup()
    flags = big["date"].map(lambda d: _nearest_holiday(d, holidays))
    big["holiday_related"] = [f[0] for f in flags]
    big["holiday_tag"] = [f[1] for f in flags]
    big["dow"] = big["date"].dt.dayofweek.map(lambda i: DOW_KR[i])
    big["bucket"] = big["holiday_related"].map(lambda x: "A_명절(prior확장)" if x else "B_미설명(고객문의)")
    big = big.sort_values("gap_z", ascending=False).reset_index(drop=True)
    big.to_csv("reports/order_shortfall_big.csv", index=False)

    cols = ["date", "dow", "actual", "production", "exceed_pct", "gap_z", "bucket", "holiday_tag"]
    print(f"\n=== 큰 매진(surprise, gap_z≥{Z_SURPRISE}): {len(big)}일 ===")
    print(big[cols].to_string(index=False))

    a = big[big["bucket"].str.startswith("A")]
    b = big[big["bucket"].str.startswith("B")]
    print(f"\n버킷 A 명절(prior 확장으로 커버): {len(a)}일 · 버킷 B 미설명(고객 문의): {len(b)}일")
    print("wrote reports/order_shortfall_big.csv")
    print("※ actual=adjusted_demand(매진일 실수요 하한) → 실제 매진은 더 잦고 큼(여긴 하한).")


if __name__ == "__main__":
    main()
