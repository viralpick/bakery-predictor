# 고객사(아띠제) 발주 baseline 재구현 — 설계

> 작성 2026-07-15 · 상태: 승인(설계) → 구현 계획 대기
> 관련 메모리: `project_baseline_logic_received`, `project_artisee_kickoff_meeting`

## 1. 목적

4주 전향(prospective) 실측 비교에서 우리 모델의 증분가치를 입증하려면, 비교 대상인 **고객사 현행 발주 시스템**을 재구현해야 한다. 현재 이 로직이 우리 코드에 없다.

- 기존 `evaluation/prospective.py`의 `reconstruct_baseline_order`(= 정상판매+마감판매+폐기)는 **실현 결과 역산 proxy**로, 예측력이 없는 oracle성 회고 지표다. 전향 비교의 competitor가 될 수 없다.
- 본 작업은 **과거 3주만 보고 앞을 예측하는 실제 발주 산식**을 재구현한다. 이것이 leakage 없는 공정한 경쟁 모델이다.
- 기존 단순 baseline(`SeasonalNaive`/`MovingAverage`)은 **삭제하지 않고 병존**한다. 우리 모델을 (a) 단순+흡수 하한, (b) 고객사 실제 운영 두 축 모두와 비교한다.

## 2. 재현 대상 스펙 (수령·해독본)

```
제시량 = [3주 적용수량 평균(주중/주말 분리)] × [S/O 증산배수] × [요일 스케일링] → 반올림
```

- **적용수량** = min(제시량, 판매량) + 특정일(휴일) 제외 + 30%↑·반품0 제거. 주중(월~금)/주말(토·일)로 묶어 3주 평균(주중 15개·주말 6개).
- **S/O 증산배수 (2단계)**:
  1. 시간대별 잔여수요곡선 = `(일평균합계 − h시까지 누적평균판매) / 일평균합계`, 최근 3개월 평균. (예: 올리브치아바타 07시 88% → 12시 57% → 21시 0%)
  2. 과거 각 일의 실제 매진시각에서 곡선을 읽어 일별 S/O지수(놓친 잔여%, 매진無=0). 3주 평균 → **증산배수 = 1 + 평균**. (주중 13.64% → 1.1364배, 주말 2.67% → 1.027배)
- **요일 스케일링**: 주중 pooled 평균에 요일 가중치 되돌림(정확 산식은 ERP 이미지 미명시).
- 갱신: 매주 월요일 새벽. 휴일=제시0, 건물휴장/명절=주말 취급.

## 3. 결정 사항 (브레인스토밍 확정)

1. **재현 충실도 = 충실 재현.** 3단계 전부 구현. 미수령/미명시 조각만 데이터 기반 근사 + caveat 명시.
2. **비교 레벨 = 발주(KPI) 중심.** 고객사 baseline은 수요 예측이 아니라 **발주 제시량**(안전 증산 포함)을 산출하는 발주 정책. 우리 발주량과 KPI(폐기비용/매진율/lost margin) 레벨에서 apples-to-apples 비교. (수요 WAPE 비교는 부적절 — 제시량은 의도적 과생산.)
3. **접근 = A.** `src/bakery/models/artisee_baseline.py`에 전용 클래스. 기존 competitor와 같은 위치, `fit`/`predict` 형태 재사용.

## 4. 아키텍처

### 4.1 모듈 위치
`src/bakery/models/artisee_baseline.py` — `ArtiseeBaseline` 클래스.

### 4.2 인터페이스
```python
class ArtiseeBaseline:
    name = "artisee_baseline"

    def __init__(self, *, weeks: int = 3, curve_months: int = 3,
                 rounding: str = "generic", multiple_map: dict | None = None): ...

    def fit(self, history: pd.DataFrame) -> "ArtiseeBaseline": ...
        # history: (store_id, item_id, date, sold_units, is_stockout,
        #           stockout_time, is_holiday). 시간대 곡선용 intraday는 별도 인자/소스.

    def predict(self, target: pd.DataFrame) -> pd.Series: ...
        # 반환: 제시량(발주량) Series, target.index 정렬.
        # ⚠️ sold_units 예측이 아니라 order quantity.
```

- `Forecaster` ABC는 `predict`가 "sold_units 예측"이라고 규정하므로, **ABC를 상속하지 않고** 동일한 `fit`/`predict` 시그니처만 따른다(덕타이핑). docstring에 발주량임을 명시. (대안: ABC 문서를 "예측 또는 발주량"으로 완화 — 미채택, 의미 혼선 방지.)

### 4.3 데이터 흐름
```
history (bonavi_daily, 광교 브레드, cutoff 이전)
  → [C1 적용수량]  (품목×요일그룹) 3주 평균, 휴일 제외 + 스파이크 캡
  → [C2 S/O 증산배수]  품목별 intraday 곡선(3개월) × 과거 매진시각 → 3주 평균 놓친%
  → [C3 요일 스케일]   주중 pool 대비 요일비(데이터 도출)
  → [C4 반올림]        제시량 = round(C1 × C2 × C3)
target dates → predict → 제시량 Series
  → prospective.simulate_item_day_kpis(order_col=제시량)
  → compare_policies(우리 발주 KPI, 아띠제 KPI) → Δ
```

## 5. 컴포넌트 상세

### C1. 적용수량 (base quantity)
- 입력: history의 `sold_units`(이미 bulk 제거·반품 net-out 완료 → 스펙 "30%↑·반품0 제거"에 근사 대응).
- 처리: `is_holiday` 일 제외 → `(item, dow_group)` 별 최근 `weeks`(=3) 주 평균. dow_group = 주중(월~금)/주말(토·일).
- 잔여 스파이크: 3주 창 내 개별 일이 창 중앙값의 1.3배↑면 캡(min(제시,판매)의 "30%↑→평균 대체" 근사).

### C2. S/O 증산배수
- (a) **품목별 intraday 잔여수요곡선**: 최근 `curve_months`(=3) 판매를 시간대(24)로 집계 → `1 − cumsum(hourly)/daily_total`의 일평균. `bonavi_loader.measure_hour_profile`(매장 단위)을 **품목 단위로 확장**하여 신규 빌드.
- (b) **일별 S/O지수**: 과거 각 일의 `stockout_time`(교정판: made>0 & waste≤0일 때 마지막 실판매 시각)을 시(hour)로 변환 → 곡선에서 그 시각의 잔여% 읽음. `is_stockout=False`면 0.
- 3주(주중/주말 분리) 평균 → `증산배수 = 1 + mean(놓친%)`.
- 매진시각·곡선 결측 시 증산배수=1(무증산) 폴백.

### C3. 요일 스케일링
- 정확 산식 미명시 → **데이터 도출**: 주중 pool 내 각 요일(월~금) 평균 / 주중 전체 평균 = 요일 가중. 주말도 토/일 각각. history 기준(cutoff 이전)으로만 계산.
- caveat: 원 시스템의 정확 가중과 다를 수 있음(문서화).

### C4. 반올림
- `rounding="generic"`(기본): 완제품=정수 반올림. 매장생산=일반 반올림(생지 배수 floor 휴리스틱).
- `multiple_map`(품목→배수 N) 주입 시 배수 floor/round 적용. **N 미수령 → 기본 generic, N 수령 시 교체.**

## 6. 통합 · 소비

- 신규 클래스는 제시량을 산출만 한다. KPI 비교는 기존 `prospective.py` 재사용:
  - `simulate_item_day_kpis(rows, profiles, order_col="artisee_order", ...)` → 폐기/lost/매진시각.
  - `compare_policies(our_kpis, artisee_kpis)` → Δ 표.
- CLI: `prospective-eval`(또는 신규 서브커맨드)에서 baseline 소스로 `ArtiseeBaseline` 선택 옵션 추가. (구체 배선은 구현 계획에서.)

## 7. Leakage 안전

- 절대 규칙 1 준수: C1~C3 모든 통계는 target date 이전 `weeks`/`curve_months` 창에서만 계산. 예측 시점 이후 sales/stockout 미사용.
- `fit(history)`/`predict(target)` 분리로 cutoff 강제. rolling/expanding fold에서 매 fold 재fit.
- `tests/test_features_leakage.py` 스타일 가드 추가: predict가 target date 이후 데이터에 접근하지 않음 검증.

## 8. 범위

- **포함**: 광교 브레드(데이터 존재 31품목), 3단계 전부, KPI 비교 통합, 테스트.
- **제외**: 타 매장(광교 단독 PoC), 비-브레드 카테고리, 품목별 배수 N 정밀 반올림(미수령), 신제품·프로모션 수동입력 재현(담당자 수기 영역).

## 9. 리스크 / 미해결

| 리스크 | 대응 |
|---|---|
| 품목별 배수 N 미수령 | generic 반올림, `multiple_map` 주입구 마련, N 수령 시 교체 |
| 요일 가중·적용수량 순환(min(제시,판매)) | 데이터 근사 + caveat 문서화 |
| 품목별 intraday 곡선 신규 빌드 | `measure_hour_profile` 확장, 곡선 결측 폴백(증산=1) |
| 증산배수가 potential_demand식 역산(그들의 함정) | 우리 baseline은 그들 방식 *재현*이 목적 — 우리 모델의 차별점(흡수·풀링·과잉α)은 별개로 유지. 재현 자체는 충실히 |

## 10. 테스트 계획

- C1: 3주 주중/주말 평균 정확값, 휴일 제외, 스파이크 캡 단위 테스트.
- C2: 곡선 읽기(알려진 매진시각→기대 잔여%), 매진無→증산 1.0, 3주 평균.
- C3: 요일 가중 합·정규화.
- C4: generic 반올림, multiple_map 배수 floor.
- Leakage: predict가 미래 미참조.
- 통합 스모크: 제시량 > 0, 매진 잦은 품목이 증산 반영, `compare_policies` Δ 산출.
