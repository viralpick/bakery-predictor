# 트랙4 (극한날씨 nonlinear) 검증 — DROP (2026-07-18, 광교 3년 OOS fixed 모델)

이상치 분석([[project_anomaly_detection]]) 도출 4트랙 중 4번(마지막). 신호: 큰 잔차 이상치 20일 중
극한날씨 2건(2024-08-11 폭염34도·매진, 2024-01-22 눈1cm). 가설: LightGBM이 온도/강수를 봤음에도
**극한 구간**에서 꺾임(nonlinear)을 못 잡아 체계적 잔차 편향이 남으면 nonlinear 보강 여지.

**검증 모델**: 현재 fixed 모델 fresh 백테스트 캐시 `reports/track3_fresh_preds.parquet`(3년 OOS
1090일) + 수원(119) 날씨 join. 모델 feature: avgTa/maxTa/minTa/sumRn/avgRhm/avgTca/avgWs +
rain_level + heavy_rain_in_biz_hours + apparent_temp (트리라 nonlinear 학습 가능, 트랙3서 날씨
feature 상위 중요도 확인).

## 결론: 트랙4 = drop (극한 구간 체계적 편향 없음)

### 게이트0 — 표본: 진단 가능
폭염 maxTa≥33 61일·한파 minTa≤−10 27일·강한비 sumRn≥30 45일·폭우 sumRn≥50 20일. (anomaly의
"극한 2건"은 큰 잔차 이상치 20일 중 날씨발이 2건이란 뜻; 전체 OOS 극한 빈도는 충분.)
**강풍 avgWs≥5 = 2일 → 표본 부족 제외.**

### 게이트1 — 예측층: 극한 bin이 튀지 않음
**WPE by maxTa bin**: 폭염 ≥33 WPE=**0.14%**(가장 낮음/무편향). 중간온도(10-28°C) 2.5~2.8%가
오히려 높음 → nonlinear 미스의 서명(극한 bin만 편향)이 **없음**.
**WPE by sumRn bin**: 강한비 ≥30 WPE=**0.52%**(무편향). (20-30 bin 2.32%나 n=20.)

**사전등록 극한 대비(2레이어 — 예측층 WPE·발주층 매진율 둘 다 동계절 통제 CI. 원신호가 폭염
"매진"이라 발주층도 반드시 검정)**: 6개 대비 전부 CI 0 포함.

| 극한 | n | 예측층 WPE diff | 95%CI | 발주층 매진율 diff | 95%CI | 판정 |
|---|---|---|---|---|---|---|
| 폭염(≥33) | 61 | +0.33pp | [−2.58,+3.38] | +0.66pp | [−10.17,+13.11] | noise |
| 한파(≤−10) | 27 | −1.86pp | [−4.80,+1.46] | +0.00pp | [−16.87,+17.28] | noise(검정력약) |
| 강한비(≥30) | 45 | −0.79pp | [−3.61,+2.23] | +4.00pp | [−9.26,+18.42] | noise |

- 예측층·발주층 모두 극한 3종 체계적 신호 없음. WPE diff 방향도 제각각 → 일관성 없음.
- **★원신호(폭염 매진)에 직접 답**: 폭염일 매진율 22.95% = 여름 비폭염 22.30%(diff +0.66pp, CI 0포함).
  폭염이 매진을 늘리는 게 아니라 **여름 매진율이 원래 높은 것**. `2024-08-11 폭염34도 매진`은
  극한날씨 효과가 아니라 여름 baseline 사건.
- **raw 매진% 오독 주의**: 한파 매진 25.9%·강한비 26.7%는 높아 보이나 **각자 계절 baseline 이하**
  (겨울 25.9%·트랙3 월별표 Jan 27.2%/Feb 30.6%). 동계절 통제하면 gap 소멸(한파 diff +0.00pp).
- **한파 예측층만 약한 under 방향**(WPE −1.06%, spike 없음)이나 **n=27로 검정력 약함** — "무편향
  확정" 아니라 판정 불가에 가까움. 발주층 매진율은 diff +0.00pp라 행동 근거 없음.

### 게이트2 — 미실행 (예측·발주 두 층 모두 극한 신호 없음)
게이트1에서 예측층·발주층 다 극한 체계적 신호가 없으므로 극한 조건부 마진 A/B는 겨냥 대상 없음.
설령 한파 예측층 약신호를 쫓아도 n=27 조건부 마진은 과적합 확실 + 트랙3 precedent(확정된 계절
매진 신호조차 iso-waste서 global 마진에 패 [[project_margin_buffer_optimization]])로 열위 예상.

## 시사점
- 모델이 극한날씨를 이미 잘 처리(트리 nonlinear + 날씨 feature 상위). nonlinear 보강 불필요.
- ★인과 주의(트랙3 교훈): 트랙4 신호 원출처도 이상치 리스트(구모델)이므로 "target/feature 수리가
  흡수"라 단정 불가 — 검증한 건 "현 모델에 극한 편향 없음"뿐.
- 한파 표본 부족은 데이터 한계(3년). PoC 광교 단독에선 재현 불가, production 다년 축적 시 재점검.

## 이상치분석 4트랙 종합
1(추세 recalib)=drop, 2(명절 prior)=공휴일 feature 버그수리+어린이날/추석/xmas prior 채택,
3(주말·여름 계절과소)=drop, 4(극한날씨)=drop. **실제 남은 데이터-정당 레버 = 평일 과대예측(waste)
+ 전역 under-calibration**([[project_distributional_forecasting_stack]]), 계절/날씨 세그먼트 마진 아님.

## 산출물
- `scripts/track4_weather_diagnose.py`(캐시 preds + 날씨 join, 백테스트 불필요). WPE 부호:
  (expected−actual)/Σ|actual|, 음수=과소예측.
