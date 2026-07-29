# 운영 리드타임 정렬 2단계 (AR feature 차단) — 실측 결과 (2026-07-29)

1단계 = `docs/operational_horizon_lead_result.md`. 설계·한계는
`docs/superpowers/specs/2026-07-29-operational-horizon-design.md`.

## 1. 무엇을 측정했나

1단계(`lead_days`)는 **학습 시점만** 옮겨서 test 행의 자기회귀 feature
(`adjusted_demand_unit_{lag1,lag7,lag14,lag28,rmean*,rstd*,ewma*}`)가 여전히 원점 이후
실측을 봤다. 2단계는 `align_features: true` 로 그 경로를 끊는다 — 타깃을 cutoff 이후로
마스킹한 뒤 **AR feature만** 재계산한다(캘린더·날씨·경쟁점 feature는 타깃과 무관해
fold-invariant이므로 대상이 아니다).

## 2. 결과 (동일 코드·vintage·`n_test=364`, 52 fold 전부 월요일 시작)

| 지표 | A 헤드라인 | C 1단계 `train_aged5` | D 2단계 `lead5_ar_aligned` |
|---|---|---|---|
| **wape** | 0.077243 | 0.078303 | **0.120696** |
| **wpe** | +0.006435 | +0.010482 | **−0.092056** |
| stockout_risk | 0.217033 | 0.208791 | **0.590659** |
| surplus_mean_units | 20.643 | 21.538 | 8.120 |
| surplus_rate | 0.083343 | 0.086815 | 0.032731 |

- 헤드라인 대비 **WAPE +4.35pp**(상대 +56%)
- **wpe가 +0.010 → −0.092 로 부호가 바뀐다** = 체계적 **과소예측** 9.2%
- 그 결과 발주가 부족해져 **매진 위험 0.209 → 0.591**, 잉여는 0.087 → 0.033

## 3. ⚠️ 이 숫자를 "운영 WAPE"로 쓰면 안 된다 — 두 가지 이유

### (a) 원점 시점의 최신 실측을 버린다

타깃-날짜 기준 lag 집합은 {1, 7, 14, 28}이다. 원점(화요일)에서 다음주 월요일을 예측할 때
화요일 값은 `target − 6`인데 **집합에 6이 없다.** 그래서 "알고 있는 가장 최신 실측"을
활용하지 못하고 그냥 버린다. 실제 운영 시스템은 horizon offset별로 feature를 **원점 기준으로
재정의**해 그 정보를 쓴다 — `scripts/operational_backtest.py` 가 그렇게 했다.

### (b) ★train/test feature 가용성 불일치 (공변량 shift)

blinding은 **test 행에만** 적용된다. train 행은 전부 `date < cutoff` 라 AR feature가
완비돼 있다. 실측:

| | AR feature 결측 비율 |
|---|---|
| train 행 | **0.0%** |
| test 행 | **34.3%** |

즉 **AR이 완비된 데이터로 학습하고, AR이 3분의 1 결측인 데이터로 예측한다.** 모델은
"lag이 없을 때 어떻게 할지"를 배운 적이 없다. LightGBM이 NaN을 기본 분기로 보내면서
레벨 앵커를 잃고 아래로 쏠린 것이 wpe −0.092의 정체로 보인다(메커니즘은 미검증 —
관측된 사실은 "train 0% / test 34.3% 불일치"와 "체계적 과소예측"이다).

→ 따라서 **0.1207은 상한이지만 느슨한 상한**이며, 순수한 정보 손실량이 아니라
**정보 손실 + 공변량 shift 페널티**가 섞인 값이다.

## 4. 결론 — 운영 성능은 아직 좁혀지지 않았다

<!-- 이 구간을 단일 숫자로 바꾸지 말 것. 폭이 넓다는 사실 자체가 3단계가 필요한 근거다. -->
| | 실험 | WAPE |
|---|---|---|
| 헤드라인(리드타임 0) | `gwangyo_default` | 0.0772 |
| **하한** | `gwangyo_train_aged5` | 0.0783 |
| **느슨한 상한** | `gwangyo_lead5_ar_aligned` | 0.1207 |

구간 [0.0783, 0.1207]은 **너무 넓어서 "운영 성능"으로 보고할 수 없다.** 폭의 대부분은
위 (a)(b) 때문에 생긴 것이고, 둘 다 3단계에서 해소된다.

## 5. 3단계 (필요 작업, 미착수)

**origin-anchored feature 재정의 + 학습/예측 가용성 일치**

- horizon offset별로 feature를 원점 기준으로 정의한다: `y(origin−1)`, `y(origin−7)`,
  `y(origin−14)`, 원점 기준 rolling 등. 그러면 "가장 최신 실측"이 항상 포함된다.
- **train 행도 같은 방식으로 구성**해 가용성 불일치를 없앤다(offset별 학습셋 → offset별 모델).
  `scripts/operational_backtest.py` 가 horizon별 별도 모델을 학습한 이유가 이것이다.
- 그 로직은 `scripts/` 에 있으므로 **참고만 하고 backbone(`harness`)으로 승격**한다
  (신규 스크립트 작성 금지 원칙).
- 비용: fold × horizon = 52 × 7 = 364 모델 fit → 스위트/실행 시간 재검토 필요.

이 3단계를 마쳐야 **헤드라인 교체 판단**(architect)에 쓸 수 있는 단일 운영 수치가 나온다.

## 6. 구현 메모 — 밟은 함정 2개

1. **AR 재계산은 반드시 날짜 연속(gapless) 프레임에서** 해야 한다. `shift()`는 위치 기반인데
   backtest가 받는 프레임은 `dropna` 로 gap이 생겨 있다(광교 실측: pre-dropna 1826행 gap 0
   / post-dropna 1791행 **gap 7건**). gappy 프레임에서 재계산하면 `lag7`이 7일 전이 아니게
   되어 **leakage를 막는 게 아니라 lag 정의를 바꿔버린다.** → runner가 `dropna` 이전
   프레임을 `ar_history` 로 넘기고, `_require_gapless` 가 gap을 **fails-loud**로 거부한다.
2. **의미론 보존 앵커를 테스트로 못박았다**: gapless history로 재계산한 AR이 엔진이 원래
   만든 AR과 **rtol=1e-12로 일치**해야 한다(`test_gapless_recompute_reproduces_engine_ar`).
   이게 깨지면 blinding이 leakage를 막은 게 아니라 feature 정의를 바꾼 것이므로 즉시 드러난다.

## 7. 재현

```bash
uv run bakery harness-run experiments/gwangyo_default.yaml           # A
uv run bakery harness-run experiments/gwangyo_train_aged5.yaml       # C 하한
uv run bakery harness-run experiments/gwangyo_lead5_ar_aligned.yaml  # D 느슨한 상한
```
