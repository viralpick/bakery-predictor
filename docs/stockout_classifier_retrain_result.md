# stockout_classifier 재측정 (라벨 재정의 후) — 발주 calibration 후속 #4

- 날짜: 2026-07-10
- 백로그: 발주 calibration 후속 #4 (TODO.md, 3→4→1→2 중 둘째)
- 관련: `is_stockout` 재정의(PR#28, [[project_stockout_time_bug_and_adjusted_demand]]), classifier 소비 경로

## 배경

`StockoutClassifier`(LightGBM 이진, target=`is_stockout`)는 **옛 라벨에서 degenerate**했다. 옛 `is_stockout`은 92%(하루 1회+ 순간품절=거의 전부 True)라, 분류기가 "거의 항상 품절"이라는 상수에 가까운 확률만 학습 → P(stockout)이 변별력 없는 flat 신호였다.

PR#28에서 `is_stockout`을 **"마감 때 진짜 소진"(made>0 & waste≤0)**으로 재정의해 92%→**60.4%**로 균형화했다. 분류기 코드는 프레임의 `is_stockout`/`stockout_time`을 읽어 학습·특징 생성하므로, **데이터가 고쳐진 시점에 재학습이 자동**이다(영속 모델 아티팩트 없음, 런타임 fit). 별도 코드 변경 없음.

이 문서는 고친 라벨에서 분류기를 재측정하고 **여전히 쓸 만한지 판정**한다.

## 방법

```bash
uv run bakery stockout-risk --source real --n-splits 4
```
- rolling time split(fold=0=최신), fold별 학습→검증.
- 지표: AUC / base_rate / precision@50 / recall@50 (ranking 지표. calibration은 측정 안 함 — 아래 "범위" 참조).
- 입력: `data/internal/bonavi_daily.parquet`(PR#28 재생성물, `is_stockout` mean 0.6044).

## 결과

| fold | val 주 | AUC | base_rate | P@50 | R@50 |
|---|---|---|---|---|---|
| 3 | 2025-12-04 | 0.636 | 0.484 | 0.64 | 0.296 |
| 2 | 2025-12-11 | **0.753** | 0.481 | 0.74 | 0.333 |
| 1 | 2025-12-18 | 0.668 | 0.587 | 0.68 | 0.260 |
| 0 | 2025-12-25 (성탄주) | **0.453** | 0.552 | 0.46 | 0.187 |

- **평균 AUC ≈ 0.63** (4 fold).

### 헤드라인
1. **degenerate 해소 (재정의 성공)**: base_rate가 92%→**0.48–0.59**로 균형 회복. 분류기가 더 이상 상수 확률을 뱉지 않는다 — 문제(옛 92% 라벨 아티팩트)의 근원이 제거됐다.
2. **판별력은 약함**: 평균 AUC ~0.63 (0.5=랜덤, 0.7+가 통상 "쓸 만함"). fold 편차 큼(0.45~0.75), 특히 성탄주(fold 0)는 **0.45 = 랜덤 이하** — 연말 특수일에서 품절 패턴이 학습 분포와 어긋남.

## 판정

- **재정의는 목적 달성**: #4의 핵심(옛 degenerate 라벨 청산)은 완료. 분류기는 이제 정상 라벨에서 정상 학습된다.
- **분류기 자체는 강하지 않다**: AUC ~0.63, 불안정, 성탄주 실패. 강한 운영 신호로 쓰기엔 부족.
- **현 PoC 비주류 경로**: 이 분류기의 유일한 소비처는 `predict-next-week --model lightgbm_v1`(안전마진에 `stockout_prob` 선형 투입). 현 PoC 주류인 **v2/v3(quantile production), v6-predict, v7 ontology(risk.py Monte-Carlo)는 이 분류기를 쓰지 않는다**. 분류기 docstring도 "더 이상 유일한 품절 신호 아님"으로 명시.
- **결정: 추가 투자 없음**. 재정의로 아티팩트가 제거됐음을 확인·기록하는 것으로 #4를 종료한다. feature/튜닝으로 AUC를 끌어올리는 작업은 비주류 경로에 대한 over-engineering이라 하지 않는다. `predict-next-week v1`은 legacy 경로로 취급한다.

## 범위 / 캐비엣

- **ranking 지표만 측정** (AUC/precision@k). `predict-next-week v1`이 `stockout_prob`을 마진에 **선형** 투입하므로, 그 경로를 실사용한다면 확률 **calibration(reliability)** 측정이 별도로 필요하다 — 현재 v1은 비주류라 측정 범위에서 제외했다. v1을 되살릴 경우 calibration 체크를 먼저 해야 함.
- 광교 단독. 타 매장 품절 프로파일 다를 수 있음.
- 4 fold(2025-12)만 — 연말 구간이라 성탄주 anomaly가 과대표집됐을 수 있음. 다른 계절 fold로 확장하면 평균 AUC가 달라질 수 있음.
- 옛 92% 라벨과의 정량 before/after 비교는 하지 않음(옛 정의 데이터를 재생성하는 비용 대비 실익 없음 — 0.92 base_rate에서 확률이 상수에 수렴 = 변별력 없음은 자명).
