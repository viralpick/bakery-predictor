# KPI 평면 백본 편입 — 설계 (2026-07-30)

발주 KPI가 `scripts/` 에만 있어서 `harness-run` 으로는 정확도(WAPE/WPE)와 비율 지표
(`surplus_rate`/`stockout_risk`)까지만 나온다. 비용·매진시각·아띠제 대비 절감률은 별도
스크립트 경로이고, **정의가 축마다 갈려 같은 축에서 비교하면 오도된다.**
이 문서는 그 편입 설계다.

architect 결정(2026-07-30): ①배분 leakage는 **1단계에서 고친다** ②KPI는 **`harness-run` 에
통합하고 YAML 플래그로 on/off**.

## 1. 현재 무엇이 어디에 있나

| 관점 | 현재 위치 | 백본 |
|---|---|---|
| WAPE / WPE | `harness/backtest_core.metrics_from_preds` | ✅ |
| `surplus_rate` / `stockout_risk` | 같은 곳 | ✅ |
| offset별 진단 | `harness/panel_backtest.horizon_diagnostics` | ✅ |
| 품목별 매진 **실측** | `harness/report._soldout_stats` | ✅ (관측값, forecaster 무관) |
| 폐기비용 / 전체매진 손실 / 총비용 | `evaluation/order_cost.py` + **일회성 스크립트** | ❌ |
| 매진시각 / 조기매진율 | 같음 | ❌ |
| **아띠제 대비 절감률** | `scripts/unified_policy_kpi.py` | ❌ |

## 2. ★ 정의 충돌 5축 — 통합은 "옮기기"가 아니라 "고르기"다

`unified_policy_kpi.py` 가 `order_cost.py` 보다 헌장에 더 충실한 부분이 있다.

| 축 | `unified_policy_kpi` | `order_cost`(신규) | **확정** |
|---|---|---|---|
| **basis** | **A(아띠제 실측 QT_OUT) + B(모델 시뮬) 병기.** A/B 간극 = censoring 크기 | B만 | **A/B 병기** — 아띠제 절감률은 A 없이 계산 불가 |
| **원가율** | 0.30 전역 | **품목별 0.40 / 0.60** | **품목별** — 접두어 판별자 오분류 0으로 검증 |
| **전체매진** | A: 그날 총폐기 ≤ `WASTE_TOL=5` / B: Σ발주 < Σ실수요 | B만 | **둘 다** — A는 아띠제 비교용, B는 모델 비교용 |
| **SKU 품절율** | **날별 비율의 평균** | 행 단위 평균 | **날별 평균** — 헌장 문구가 "각 날 품절 SKU 비율의 날별 평균" |
| **매진시각** | 전체 median | **날별 median의 평균** | **병기** — 둘이 다른 질문에 답한다 |

### 중복 제거
`unified_policy_kpi` 의 `cat_shortfall_on_sellout`(전체매진일 부족률) / `cat_undersupply_rate`
(전체 부족률)은 **`order_cost` 에서 내가 다시 만든 것과 같은 지표다.** 프리미티브 하나로 합친다.

### ⚠️ fold 규약 불일치 — 옛 수치는 애초에 비교 불가였다
`unified_policy_kpi`: `N_FOLDS=8, VAL_WEEKS=8` / 헤드라인: 52 fold × 7일.
**폐기 절감 −37~45%는 8-fold 기준이고 헤드라인 52-fold와 같은 축이 아니다.**
통일하면 이 수치는 재측정된다(4단계).

## 3. ★ 배분 단계 leakage — 1단계에서 고친다

```python
compute_proportions(history, target_date)   # cutoff_date = target_date
```

배분 비율이 **대상일 직전까지의 history** 를 쓴다. 그런데 리드타임 하에서 발주 시점(원점)에는
`target_date − lead_days` 까지만 알 수 있다. **절대 규칙 #2(Time leakage 금지) 위반**이고,
C1에서 찾은 운영 horizon 불일치와 **완전히 같은 유형**이다(그때는 총량, 이번은 배분).

기존 가드 `tests/models/test_item_proportion_leakage.py` 는 *"`< cutoff` 만 쓴다"* 를 보장하지만
**cutoff 자체가 대상일로 고정**돼 있어 리드타임을 보지 못한다.

### 수정 방향 — 두 개념을 분리한다
- `target_date`: **무엇을 예측하는가** (배분 대상일)
- `cutoff_date`: **무엇을 알 수 있는가** (정보 경계)

```python
compute_proportions(history, target_date, *, cutoff_date=None)   # None → target_date (기존 동작)
distribute_total(history, total_by_date, *, lead_days=0)          # cutoff = target - lead_days
```

기본값이 기존 동작과 **정확히 같아** 모든 호출부가 무변경으로 살아난다(계약 보존).

### 호출부 전수 (착수 전 grep, 테스트 fixture 포함 — 절대 규칙 #3)
| 파일 | 위치 | 마이그레이션 |
|---|---|---|
| `forecast/forward.py` | 206, 208 | `lead_days` 전달 (운영형 forward) |
| `cli.py` | 2356 | 기본값 유지(레거시 경로) |
| `analysis/lab/handlers/stockout.py` | 90 | 기본값 유지(반사실 분석) |
| `models/item_proportion.py` | 178, 198 | 내부 |
| `tests/models/test_item_proportion_leakage.py` | — | **lead_days 케이스 추가** |
| `tests/evaluation/test_category_order.py` | 54 | 무변경 확인 |
| `tests/analysis_lab/test_handlers_stockout.py` | 111 | 무변경 확인 |
| `scripts/{compare_order_kpi,revalidate_popularity_stockout}.py` | — | 무변경 |

## 4. 단계 분할

| 단계 | 내용 | 산출 |
|---|---|---|
| **1** | 배분 leakage 수정(`cutoff_date` 분리) + **영향 실측** + 정의 5축 확정 문서화 | PR ① |
| 2 | 배분 단계를 harness에 배선 — `ExperimentSpec.order_level: category\|item` | PR ② |
| 3 | KPI 지표 편입 — `order_cost` + A/B basis + 아띠제 절감률 → harness 지표·리포트 | PR ③ |
| 4 | 옛 수치 재측정 + 차이 원인 문서화 + `unified_policy_kpi.py` wrapper화 | PR ④ |

**1단계를 독립 PR로 두는 이유**: leakage 수정은 수치를 바꾼다. 배선과 섞으면 "무엇이 수치를
바꿨나"를 식별할 수 없다 — C1 2단계에서 정확히 이 실수를 했고, gap의 92%가 "정보 지우기"
부작용이었음을 나중에야 분리해냈다.

## 5. 성공 기준

1. `harness-run` 하나로 정확도 + 비용 + 매진 + 아띠제 절감률이 전부 나온다(3단계 완료 시)
2. **엔진 동등성 hard gate 무변경** — `test_backtest_core_equivalence.py` 52-fold rtol=1e-9
3. 배분 컷오프가 `lead_days` 를 준수하고 **테스트로 고정**된다
4. A/B basis 병기 + 정의 5축이 코드 한 곳에서 확정되고 리포트에 명시된다
5. 전체 스위트 green (기준: 919 passed)
6. `unified_policy_kpi.py` 는 얇은 wrapper로 축소(중복 계산 0)
7. 옛 수치(−37~45%)를 새 기준으로 재측정하고 **차이 원인을 축별로 분해**해 문서화

## 6. 리스크

| 리스크 | 대응 |
|---|---|
| leakage 수정이 KPI 수치를 바꾼다 | 1단계에서 **before/after 실측**하고 문서화. 값이 나빠지는 것이 정상(정보를 덜 쓴다) |
| 배분 비용 — 날짜마다 `compute_proportions` | YAML 플래그로 off 기본. 헤드라인 실험 속도 보존 |
| `simulate_item_day_kpis` 계약 변경 시 기존 CLI 파손 | 계약 **변경 금지**, 새 함수로 감싼다 |
| fold 통일로 −37~45% 가 바뀐다 | 4단계에서 옛/새 병기하고 축별 차이 분해 |
| 정의 변경이 옛 문서 수치를 무효화 | 절대 규칙 #6대로 vintage·파이프라인 병기 |

## 7. 배분은 "빠뜨린" 게 아니다 — harness가 총량 층위에서 잘렸다 (경위)

architect 질문: *"하네스에 배분단계는 왜 없지? 초기 계획에서 건너뛴 건가?"* → **아니다.**

`modeling_v4.md:170` 의 3-stage 설계가 코드에 그대로 있다:
```
proportion   = raw_weight / Σ raw_weight
final_qty[i] = Stage_1_production × proportion[i]
```
→ `models/item_proportion.py`(Stage 2)이고, **`cli.py:2331` `_category_order_predictions` 가
총량 → 마진 → 배분을 완결**한다. `forecast/forward.py:206`(운영형 forward)도 같다.

| 경로 | 총량 예측 | 마진(발주량) | **품목 배분** |
|---|---|---|---|
| **`harness-run`**(헤드라인) | ✅ | ✅ | ❌ |
| `cli.py _category_order_predictions` | ✅ | ✅ (3방식) | ✅ |
| `forecast/forward.py`(운영형) | ✅ | ✅ | ✅ |

Phase 1 스파인 추출 때 harness는 **"총량 예측기 비교" 평면**으로 범위가 잡혔고, 배분은
공통 후단이라 예측기 비교에서는 상수로 취급할 수 있었다. 그 판단 자체는 합리적이었다.

**문제는 그 절단이 KPI를 불가능하게 만든 것이다** — 폐기는 품목 단위로 발생하고, 매진시각은
품목 단위가 아니면 정의되지 않는다. 실측: **품목 단위 부족량이 카테고리 단위의 7배**
(8,300 vs 1,201개) — 총량 집계가 배분 오차를 상쇄해 숨긴다.

⇒ 2단계는 **"배분을 새로 만드는 것"이 아니라 "이미 있는 배분을 harness에 배선하는 것"** 이다.
재구현 없이 `distribute_total` 호출만 붙인다.

그리고 이 절단이 leakage를 가려왔다: 배분이 CLI에만 있는 동안은 리드타임이 없어서(C1 이전)
드러나지 않았다. `cli.py:2353` 주석이 *"compute_proportions가 <date만 사용"* 이라 적어둔 게
정확히 그 지점이다.

## 8. 1단계 결과 — leakage 수정 + 영향 실측 (완료)

### 구현
- `compute_proportions(history, target_date, *, cutoff_date=None)` — 대상일과 정보 경계 분리
- `distribute_total(history, total_by_date, *, lead_days=0)` — `cutoff = target − lead_days`,
  음수는 fails-loud
- 기본값이 기존 동작과 동일 → **호출부 8곳 무변경**(계약 보존)

### ★영향 실측 (광교, 최근 364 대상일 × 품목 = 13,339행, `lead_days` 0 vs 5)

| 지표 | 값 |
|---|---|
| \|Δproportion\| 평균 | 0.001139 |
| \|Δproportion\| p95 / 최대 | 0.003948 / 0.017374 |
| 날짜별 L1 거리 평균 | **0.0417** → 배분의 **평균 2.09% 가 재배치** |
| 총량 300개 가정 시 품목별 차이 평균 | 0.342개 |
| 같은 가정 p95 / 최대 | 1.184개 / **5.212개** |
| **1개 이상 달라지는 품목-일 비율** | **6.71%** |

**해석**: leakage는 실재하지만 크기는 작다 — 배분의 약 2%가 재배치되고, 품목-일의 6.7%에서
발주가 1개 이상 달라진다. **총량 경로의 리드타임 비용(+0.536pp WAPE)보다 작은 축**이다.

⚠️ 그러나 크기가 작다는 것이 남겨둘 이유는 아니다. (a) 절대 규칙 #2는 크기 기준이 아니다
(b) 2단계에서 이 배분이 헤드라인 KPI의 입력이 되므로, 여기서 새면 폐기·매진 수치 전체가
발주자가 갖지 못한 정보에 의존하게 된다.

⚠️ 이 측정은 **비율 변화**만 본다. KPI(폐기·매진 비용)로의 전파는 총량과 곱해지고 실수요와
비교되므로 3단계에서 다시 측정한다 — 여기 수치를 KPI 변동으로 인용하면 안 된다.

## 9. 2단계 결과 — 배분을 harness에 배선 (완료)

### 설계: 가법(additive) 확장
```python
@dataclass
class BacktestResult:
    folds: pd.DataFrame
    predictions: pd.DataFrame
    item_orders: pd.DataFrame | None = None    # ← 추가만. 기존 두 필드 불변
```
`ExperimentSpec.order_level: "category" | "item"`, **기본 `category`** 로 헤드라인 속도 보존.
배분은 `models.item_proportion.distribute_total` **호출만**(재구현 0).

### 관문 통과
| 검증 | 결과 |
|---|---|
| **엔진 동등성 hard gate**(52-fold, rtol=1e-9) | **1 passed** — 카테고리 경로 무변경 |
| 배분 on/off 시 총량 예측 | **rtol=1e-12 일치** |
| 총량 보존(배분 합 == 총 발주), 52-fold 실측 | **최대 오차 1.14e-13** |
| `order_level=category` 시 `item_orders` | `None`(배분 비용 안 냄) |

### 실측 산출 (`experiments/gwangyo_item_orders.yaml`, 52-fold, `lead_days=5`)
두 arm 모두 **13,457행 / 품목 58종 / 364일**. `item_orders.csv` 로 저장된다.

### 세 가지 함정을 막았다
1. **`lead_days` 를 배분에 전달** — 총량만 리드타임을 지키고 배분이 대상일 직전을 보면
   파이프라인이 **부분 leaky**다. PR#74에서 막은 축이 harness 경로로 실제로 흐르는지
   테스트로 고정(`test_lead_days_changes_item_orders_not_shape`, 리드 구간 spike 주입).
2. **캐시 키에 `order_level` 포함** — 안 넣으면 배분 on/off가 같은 캐시를 공유해 조용히 틀린다.
3. **`panel` + `item` 조합은 fails-loud** — 패널은 fold가 원점 기준이라 배분 대상일 매핑이
   미정의다. 조용히 틀린 배분을 내는 것보다 막는다. (패널 KPI는 별도 과제)

### 테스트 9종
가법성(총량 불변) / fold 지표 불변 / 총량 보존 / 계약(컬럼·fold 집합) /
**배분 비율이 history를 따르는지**(a:b=1:3 → 0.25) / history 없으면 fails-loud /
잘못된 `order_level` 거부 / `category` 시 None / **리드타임 실효**(대조군 방식).

⚠️ **품목 수 차이 주의**: 배분 산출은 **58종**인데 타깃 카테고리 전체는 157종이다.
`compute_proportions` 가 최근 판매 실적이 있는 품목에만 배분하기 때문이다. 3단계 KPI에서
이 population 차이가 폐기·매진 분모에 영향을 주므로 그때 다시 다룬다
(PR#72에서 관측한 "강제-0 수요 행 9.4%" 와 같은 뿌리).

## 10. 3단계 전제조건 검증 (완료)

A basis(아띠제 실측)가 harness 평가 구간과 정렬되는지 확인했다:

| 항목 | 값 |
|---|---|
| 범위 | 2021-01-01 ~ 2025-12-31 |
| 품목-일 행 / 품목 | 91,619 / 157 |
| `production_qty` / `waste_qty` 결측 | **0 / 0** |
| 최근 364일 커버 | **364일 (완전)** |

→ A/B basis 병기가 데이터 측면에서 가능하다. 3단계 착수 가능.
