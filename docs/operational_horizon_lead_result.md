# 운영 리드타임 정렬 1단계 — 실측 결과 (2026-07-29)

`lead_days` / `anchor_dow` 도입 후 광교 canonical 스택으로 측정. 설계·한계는
`docs/superpowers/specs/2026-07-29-operational-horizon-design.md`.

## 1. 측정 조건

전부 **같은 코드·같은 vintage·같은 `n_test=364`**(52 fold × 7일)다. 엔진 동등성 게이트가
`lead_days=0, anchor_dow=None` 경로의 바이트 동일성을 보증하므로 헤드라인 값은 그대로 쓸 수 있다.

| # | 실험 | lead_days | anchor_dow | 블록 |
|---|---|---|---|---|
| A | `gwangyo_default` (헤드라인) | 0 | None | 목~수 |
| B | ablation (일회성, 미커밋) | 0 | 0 | 월~일 |
| C | `gwangyo_train_aged5` | 5 | 0 | 월~일 |

## 2. 결과

| 지표 | A 헤드라인 | B 앵커만 | C 앵커+lead5 |
|---|---|---|---|
| **wape** | 0.077243 | 0.076346 | **0.078303** |
| wpe | 0.006435 | 0.008139 | 0.010482 |
| stockout_risk | 0.217033 | 0.217033 | 0.208791 |
| surplus_mean_units | 20.643 | 21.172 | 21.538 |
| surplus_rate | 0.083343 | 0.085339 | 0.086815 |

## 3. 분해

| 효과 | WAPE 변화 |
|---|---|
| 월요일 앵커링 (A→B) | **−0.090pp** (0.077243 → 0.076346) |
| 리드타임 5일 순효과 (B→C) | **+0.196pp** (0.076346 → 0.078303) |
| 합계 (A→C) | +0.106pp |

**요일 앵커링만 놓으면 오히려 미세하게 좋아진다.** 즉 A→C의 +0.106pp는 두 효과가 상쇄된
값이고, **리드타임의 순효과는 +0.196pp**(상대 약 2.6%)다. 앵커링 단독 비교를 하지 않았다면
리드타임 비용을 절반으로 과소평가했을 것이다.

부수 관찰: 리드타임을 주면 예측이 위로 밀린다(wpe 0.0081 → 0.0105) → 매진 위험은 내려가고
(0.2170 → 0.2088) 잉여는 올라간다(surplus_rate 0.0853 → 0.0868). 즉 정확도 손실이
발주 정책 축에서는 "조금 더 보수적으로" 나타난다.

## 4. 해석 — 그리고 하지 말아야 할 해석

**할 수 있는 해석**: 학습 데이터가 5일 오래되는 것 자체는 이 스택에 거의 타격이 없다
(+0.196pp). 잔차가 무편향·무구조이고 점추정 헤드룸이 거의 없다는 기존 진단과 일관된다 —
모델이 안정적인 캘린더·계절 구조와 자기회귀 lag에 의존하고, **이 실험에서 lag는 그대로
남아 있다.**

<!-- 이 경고를 지우지 말 것. 이 수치가 단독 인용될 때 생기는 오독을 막는 유일한 장치다. -->
⚠️ **하지 말아야 할 해석: "운영 정렬은 거의 무료다."** 이 실험은 **모델 학습 시점만**
옮겼다. test 행의 자기회귀 feature는 프레임 전체에 대해 한 번 계산되므로 여전히 원점 이후
실측을 본다(실측: 월요일 블록 test 행의 `lag1` = 전날 일요일 = cutoff+4일).
**운영에서 진짜 잃는 것은 lag 6일치이며 그 효과는 아직 측정되지 않았다.** 따라서 위
+0.196pp는 **운영 열화의 하한(lower bound)** 이다.

## 5. 다음

1. **feature 레이어 정렬** — horizon별 lag 재구성. `forecast/forward.py` 의
   `_extend_category_features`(미래 행 target=NaN → lag가 NaN이 되어 leakage 차단)가
   재사용할 프리미티브다. 이걸 해야 운영 열화의 실제 크기가 나온다.
2. 그 수치를 보고 **헤드라인 교체 판단**(architect).
3. 폐기 KPI 축(`scripts/unified_policy_kpi.py`)에서도 같은 리드타임을 적용해야 운영 KPI가
   정합된다 — 현재 그 경로는 backbone 밖이다.

## 6. 재현

```bash
uv run bakery harness-run experiments/gwangyo_default.yaml       # A
uv run bakery harness-run experiments/gwangyo_train_aged5.yaml   # C
# B는 일회성 ablation(lead_days=0 + anchor_dow=0). 커밋하지 않았다 —
# 필요하면 train_aged5 YAML에서 lead_days만 0으로 바꿔 재현한다.
```
