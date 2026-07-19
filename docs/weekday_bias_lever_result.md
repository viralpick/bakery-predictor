# 실제 레버 추격 — 평일 center 과대예측 = iso-waste 무가치, 남는 건 전역 calibration (2026-07-19)

이상치 4트랙 종료 후 advisor가 지목한 "실제 남은 레버"(평일 과대예측 + 전역 under-calibration)를
파고든 결과. 캐시 `reports/track3_fresh_preds.parquet`(광교 3년 OOS fixed 모델) 재사용.

## 결론: 요일 특정 보정 = 프론티어-중립(노이즈 내) → 유일한 데이터-정당 레버 = 전역 calibration 스칼라 하나

### 진단 1 — distributional/σ(x) branch 사망
dow별 상대잔차 (actual−exp)/exp **std 7일 전부 8.5~10.9%로 평평**(토 9.71%=평일). dow 커버리지
부족(q0.85 cov 78.6%<85%)은 spread가 아니라 **center(WPE)로 전부 설명**됨(cov=Φ((1.06−중심)/0.10),
7일 1~3pp 내 재현). [[project_distributional_forecasting_stack]] 전제("저수요일 마진↑ spread 병리")는
fix 전(q0.95·potential_demand) 산물 — flat ~6% 상단마진과 모순, **반증**. σ(x) 학습 착수 금지.

### 진단 2 — center 편향의 실체와 원인
- expected center(rel mean): **월 −2.84%·수 −3.58% = 3년 일관 과대예측**(부호 안정). 목·금·토·일은
  연도별 부호 뒤집힘(2025 무편향/역전) = 불안정.
- **원인**: 요일 레벨 편차 큼(평일 227~240 vs **주말 296~308, +30%**). rolling/ewma feature가
  dow-blind → 최근 7일 평균에 주말 고수요 2일이 섞여 저수요 평일(월·수) 예측을 상향. lag7은 요일을
  맞추나 rolling이 압도(feature importance 달력 9.6%, dow는 top15 밖).

### 게이트 — 평일 center 보정 iso-waste (advisor + 트랙1 교훈)
`scripts/weekday_bias_isowaste.py`: 발주=exp×mult, GLOBAL(전역 균일 하향) vs DOW(월·수만 트림
2~4% + 전역 보충). 동일 waste(6·8·10%)에서 DOW−GLOBAL 매진 gap, 주 블록 부트스트랩 CI.

**9개 조합(3 waste × 3 trim) 전부 매진 gap CI 0포함(프론티어-중립).** gap ±0.1~0.9pp·CI 반폭
~1.5pp라 "무가치"와 "underpowered(2차 효과라 탐지 불가)"는 구분 안 됨 — 행동은 어느 쪽이든 동일
(요일 타깃팅 안 함). 트랙1 online 시간보정이 상수하향에 진 것과 동형.

**메커니즘**: center 편향은 aggregate WPE로는 존재하나 일별로는 std 10%에 묻힘. 월·수 3% 트림은
그 요일 일별 변동 대비 작아 매진 구조가 전역 하향과 구분 안 됨. **dow 잔차 std가 평평하므로 어느
요일을 낮추든 같은 waste면 같은 매진** → 요일 특정성이 프론티어 이득 없음.

### 함의 — 모델fix(dow-aware rolling)는 디프리오리타이즈(직접 검정 안 함)
게이트가 검정한 것은 **post-hoc 평면 dow 곱셈**(per-dow 평균만 이동)이 전역 상수 하향과 프론티어-중립
이라는 것. 여기서 "모델fix도 순가치 없음"으로 넘어가는 건 비약 — 모델fix(dow-aware rolling)는 원리상
평면 배수가 못 하는 것(within-dow 잔차 σ 감소, 연휴 뒤 주말-bleed 상호작용)을 잡을 수 있다. dow축
σ가 평평한 건 "rolling 분리가 σ를 못 줄인다"를 **함의하지 않음**(현 σ 10%의 일부가 주말-bleed
노이즈면 분리로 좁혀질 여지, 미검정). **디프리오리타이즈 결정은 유지**(σ 평평 → 가치 가능성 낮음,
지금 투자 안 함) 하되 "증명됨"은 아님 — 필요 시 dow-aware rolling의 within-dow σ 감소를 별도 검정.

## 최종 착지
★통합 인사이트: **조건부 조정은 평균이동이 잔차 σ 대비 클 때만 프론티어를 옮긴다.**
- **명절(EventLevelPrior) = YES**: 1차 효과·큰 평균이동·연도 안정 → 트랙2 채택(xmas·어린이날·추석,
  프론티어 실제 이동). [[project_special_day_feature_spec]]
- **dow·계절·날씨 = NO**: 2차 효과라 σ(~10%)에 묻힘 → 트랙3(주말·여름)·트랙4(극한날씨)·본 추격
  (평일 center) 전부 iso-waste 프론티어-중립(노이즈 내). 다섯 조사가 이 한 줄로 묶임.
- 조건부 레버가 다 걸러지면 **남는 데이터-정당 조정 = 전역 스칼라 하나**.

전역 스칼라 = 두 가지로 분리해 다뤄야:
- **(실측·실행가능) q0.85 recalibration**: cov 78.6% < 명목 85% = 분위수 모델이 자기 이름값 미달 =
  **miscalibration**(정책 선택 아님). [[project_prospective_derisk_retro]]의 "전향 전 recalibration
  필요" standing item 재확인 — 분위수를 명목에 맞추는 재보정이 이 추격의 유일한 구체 산출.
- **(정책) 발주 공격성**: recalibration 후에도 남는 매진↔waste trade는 KPI 우선순위(waste 1차
  [[project_kpi_priority_framing]])로 선택. backtest 최적화 불가(보험 성격) → 비용표로 architect가
  선택([[project_margin_buffer_optimization]] 버퍼=보험 결론과 정합). 요일별 차등 아님.

## 산출물
- `scripts/weekday_bias_isowaste.py`(게이트). 진단은 인라인(캐시 preds 재분석, 재현 가능).
- rel 부호: (actual−exp)/exp, 음수=expected 과대예측. gap 음수=DOW 우위.
