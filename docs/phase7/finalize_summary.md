# Phase 7 신규 데이터 편입 마무리 — 최종 요약

Phase 7(신규 데이터 편입 마무리) Task 0~5 종료 시점 상태 기록. 코드 변경 없음(문서화 전용).

## 1. 게이트 상태 (Task 2에서 확인)

| 게이트 | 결과 |
|---|---|
| `build-data --diagnose` | 광교 canonical `max_diff=0` (재빌드 동등) |
| `check-integrity` | fail 0 (drift 2건만, 아래 §2) |
| `check-conflict` | conflicting=0 |
| 광교 canonical `bonavi_daily.parquet` md5 | 불변 |
| leakage 테스트 | 12/12 통과 |

## 2. Drift 2건 (fail 아님, 정보성)

1. **할인코드 357 미정규화** — 기존에 알려진 항목, 신규 발견 아님.
2. **비타깃 품목 1649개 마스터 미등록** — 전부 `is_target_scope=False`(음료/완제품), 타깃 품목(빵/디저트/샌드위치) 아님. 타깃 스코프 무결성에는 영향 없음.

## 3. 산출물

- **`multistore_daily.parquet` 생성**: 4매장 268,575행 (광교 75,615 / 삼성 66,445 / 메세나 73,236 / 광화문 53,279), 기간 2021~2025-12. 광교 canonical과 물리적으로 격리(별도 parquet, 광교 파이프라인 무변경).
- **헤드라인 재측정** (`docs/phase7/gwangyo_headline_remeasure.md`): 광교 총량 WAPE 8.03% → 7.72% (−0.31pp, 동일 harness 경로 클린 delta). 해석 주의: "모델이 더 정확해졌다"가 아니라 "신규 166 canonical 기준으로 헤드라인을 재설정했다"는 의미.
- **폐기 1차 KPI 재측정** (`docs/phase7/gwangyo_waste_kpi_remeasure.md`, 빵 품목 157개 스코프): 폐기 −37~45% 재확인·강화 (quantile −44.7% vs 옛 −40.5%, conformal −37.3% vs 옛 −32.9%). 방향성 유지, 신규 closing 소스로 확인(confirmed).
- **타 3매장 참조 예측** (`docs/phase7/multistore_reference.csv`): WAPE 삼성 0.0883 / 메세나 0.0858 / 광화문 0.0899. event_prior 미적용(광교 전용 로직), 참조용이며 타깃 아님.

## 4. 경계 (필수 명시)

- 모든 재측정 delta는 **146→166 품목/타깃 정의 변경 + 신규 closing 소스 변화분**이다. 모델 알고리즘 변경이 아니다.
- 데이터 기간은 **2021~2025-12**이다. 이번 재측정은 **2026 전향(prospective) 검증이 아니다**. 전향 검증은 컷오프 2026-06-30 이후 실시간 피드가 필요하며 이번 Phase 범위 밖.

## 5. 열린 결정 (범위 밖 / architect·아띠제 확인 대기)

1. **cake/sweets 스코프**: 폐기 재측정은 `TARGET_CATEGORIES`(bread/pastry/sandwich, cake 제외 — 코드 근거 "사전예약+시즌특수")로 157품목 기준. 옛 헤드라인과 동일 스코프라 like-for-like 비교는 유효하다. cake를 포함한 166품목 전체 재측정을 원하면 architect 결정이 필요하다.
2. **★harness latent bug (후속 수정 대상)**: `src/bakery/harness/runner.py:76`

   ```python
   events, lunar = resolve_event_priors(spec.event_priors) if "event_prior" in spec.layers else (None, None)
   ```

   `event_prior`가 `spec.layers`에 없는 실험은 `events=None`을 넘기는데, `EventLevelPrior`가 이를 "지정 없음"으로 받아 내부적으로 `DEFAULT_EVENTS`(xmas 포함)로 몰래 fallback한다. 현재 canonical `gwangyo_default` 실험은 event_prior가 ON이므로 이번 헤드라인 재측정에는 영향이 없으나, **event_prior를 끈 실험 결과는 의도와 달리 xmas 보정이 켜진 상태로 나온다.** Phase 7 범위 밖, 별도 수정 필요.
3. **원가율 미보유**: 폐기비용은 판매가 계열로 유지(`COST_RATE=0.30` 가정). 원가 gap 미해결.
4. **음료 마스터 부재**: T5(카페 트래픽 심화) blocked.
5. **진열시간 계획 vs 실측 미확정**.
6. **전향 4주 실측**: 실시간 피드 필요, 이번 Phase 범위 밖.

## 6. 전체 테스트 스위트 최종 확인

`uv run pytest --color=no` (controller 직접 실행, 2026-07-28): **665 passed, 0 failed** (1227s / 20분, harness 동등성 52-fold 포함). Phase 7 신규 테스트(test_category_closing_source·test_multistore_build·test_cli_build_multistore) 포함 전부 통과, 사전존재 실패 없음.
