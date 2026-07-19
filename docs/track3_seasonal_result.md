# 트랙3 (주말·여름 계절 과소예측) 검증 — DROP (2026-07-18, 광교 3년 OOS)

이상치 분석([[project_anomaly_detection]]) 도출 4트랙 중 3번. 신호(렌즈③④): "발주부족 8일 전부
급등·주말/여름 집중, 2024 여름 매진 집중". 후보 fix = 요일×월/계절 마진.

**검증 모델**: 현재 fixed 모델 fresh 백테스트. 캐시 `reports/track3_fresh_preds.parquet`
(160-fold 730d rolling, 3년 OOS 1090일). (참고: 트랙3을 flag한 이상치분석 모델도 이미
adjusted_demand·α0.8·bulk제외·EventPrior를 썼고, 그 이후 코드 델타는 holiday feature 수리 하나뿐 —
holiday 수리는 공휴일에 작용하지 주말 전반엔 약하므로 주말·여름 신호와는 인과가 약함. 따라서 아래는
"신호가 사라졌다"가 아니라 "현재 모델에서 계절 마진이 정당한가"를 직접 검증한 것.)

## 결론: 트랙3 = drop (예측층 체계적 편향 없음 + 발주층 계절 마진 iso-waste 열위)

### 게이트1 — 예측층 계절 과소예측? = 체계적 편향 없음
전체 WPE=+1.66%(경미한 **과대**예측, under 아님). 사전등록 3대비(부트스트랩 95%CI):

| 세그먼트 | 세그 WPE | vs 나머지 diff | CI 0배제 | spike 제거 후 |
|---|---|---|---|---|
| 주말 | +0.51% | −1.75pp | ★real | −0.10% (≈0) |
| 여름(6-8월) | −0.02% | −2.25pp | ★real | −0.13% (≈0) |
| 주말×여름 | −0.83% | −2.73pp | ★real | −1.16% |

- 대비(diff)는 real이나 **방향 반대**: 주말·여름은 과소예측 아니라 WPE≈0(무편향). 대비가 real인 건
  **평일·비여름이 +2~3% 과대예측**이기 때문. "주말·여름 과소" → 실체 = "평일 과대".
- spike(robust |z|≥3) 제거하면 주말·여름 모두 ≈0 → 마진으로 고칠 계절 under-prediction 없음.
- 원래 렌즈③④ 신호는 발주/생산층(shortfall spike일 + detrend 재랭킹)이었지 expected 체계적 편향
  주장이 아니었음 → gate1은 "expected층엔 애초에 체계적 계절 편향이 없다"를 확인.

### 게이트2 — 발주층 계절 마진이 전역 마진을 iso-waste에서 이기는가? = 아니오(열위)
유일한 생존 신호 = 발주층 매진빈도가 주말·여름에 초과(주말 26.8% vs 평일 19.2%, 주말×여름 30.0%
vs 20.7%, 넓게 퍼진 신호=spike 아님). 계절 마진이 이걸 정당하게 겨냥하는지 iso-waste A/B.

**★매진 초과의 진짜 원인 = 이분산 아님(측정 반증)**: 세그먼트 상대잔차 std/IQR이 평평(타깃 9.5%
vs 비타깃 10.1%, 주말 9.8%=평일 9.8%, 여름 9.3% vs 비여름 10.0%). 주말 매진 초과는 **평일
과대예측의 뒷면** — 평일은 expected +2~3% 과대 → q0.85가 actual보다 훨씬 위 → 매진 드묾, 주말은
무편향 → q0.85가 85%ile 근처 → 매진 정상 발생. 근원은 gate1의 평일 과대예측.

**★공정한 A/B(증분)**: 공통 base 폐기율 6%(전역)에서 출발 → 추가 폐기를 전역 균일 상향 vs
주말·여름에만 추가. ("타깃일에만 buffer·비타깃 0"은 strawman=비타깃일 median 방치로 매진 폭발 →
반드시 공통 base 위 증분만 비교.) SEASONAL−GLOBAL 매진 gap, 주 단위 블록 부트스트랩 CI.

| 증분 | 매진빈도 gap | 95%CI | 매진크기 gap | 95%CI | 판정 |
|---|---|---|---|---|---|
| 6→8% | +0.64pp | [−1.37, +3.21] | +0.01pp | [−0.08, +0.11] | 0포함(무차) |
| 6→10% | +2.57pp | [+0.64, +5.13] | +0.24pp | [+0.11, +0.38] | ★**global 우위**(계절 열위) |

- 계절 마진은 전역보다 **한 번도 낫지 않고**, 폐기를 더 쓸수록 유의하게 **열위**.
- **메커니즘**(측정 정합): 세그먼트 분산이 동일하므로 버퍼를 47% 날(주말∪여름)에 집중하면 그 날들의
  한계 매진감소가 체감(diminishing returns) → 전역 분산이 최소 동등, 증분 크면 우위. 매진이 계절에
  몰린 건 분산차가 아니라 평일 과대예측이라, 계절 마진은 잘못된 레버.
  [[project_margin_buffer_optimization]] "정률 마진 OOS 과적합" 결론과 정합, 그보다 강한 실측 열위.

### 메커니즘 (참고)
expected 모델 feature importance: 달력 feature 9.6%(dow/month/is_weekend는 top15 밖, dom만 높음).
레벨은 lag/rolling·**날씨(습도 avgRhm·운량 avgTca·풍속 avgWs 상위)**가 담당. 트리가 dow×month를
직접 모델링하진 않으나 expected가 이미 무편향이라 평균은 문제없음 — 남은 건 분산뿐(마진 무해결).

## 시사점
- **실제 레버는 계절 마진이 아니라 평일 과대예측**(+2~3% WPE = 폐기 낭비). 계절 조건부 σ(x)/분포회귀도
  세그먼트 분산이 평평하므로(측정) 계절축 확장은 근거 없음 — 있다면 전역 calibration 문제.
- 전역 shortfall 21.4% ≫ q0.85 명목 15% = 전역 under-calibration 재확인([[project_prospective_derisk_retro]]).
- 남은 데이터-정당 방향은 전역 calibration(분위수/분포 스택 [[project_distributional_forecasting_stack]])
  이지 계절 세그먼트 마진이 아님. 트랙3 범위 종료.

## 산출물
- `scripts/track3_seasonal_diagnose.py`(fresh 백테스트+2레이어 진단+사전등록 대비+메커니즘),
  `scripts/track3_gate2_isowaste.py`(iso-waste 증분 A/B).
- `reports/track3_fresh_preds.parquet`(fixed 모델 3년 OOS, 유일 비싼단계 캐시), `track3_frontier_global.csv`.
- WPE 부호: (expected−actual)/Σ|actual|, 음수=과소예측. gap 음수=계절 우위.
