# is_stockout/stockout_time 재정의 (데이터 fix) — 설계

작성 2026-07-08. [[project_stockout_remediation_roadmap]] 하위 프로젝트 1. stockout_time 로더 버그([[project_stockout_time_bug_and_adjusted_demand]]) 교정. **데이터만 고친다 — 재검증(하위 2)은 별도.**

## 배경 (버그)

`bonavi_loader.build_daily`가 `품절정보` 시트의 **가장 이른 품절 이벤트**를 stockout_time으로, `is_stockout = stockout_time.notna()`로 채운다(`bonavi_loader.py:264 first_so`). 그러나 베이커리는 하루 중 재입고(리필)로 같은 품목이 여러 번 순간품절(광교 item-day의 53.9%가 2+ 이벤트, 시각 spread 중앙 ~10h)한다. 첫 이벤트만 취해:
- stockout_time이 실제 최종소진보다 훨씬 이르게 찍힘(is_stockout 품목-일의 43%가 실판매가 2h+ 늦음, 전-품목 09:05 동일 아티팩트).
- is_stockout = 92% (= 하루 1회+ 순간품절 = 정상 상태).
- **potential_demand가 이 stockout_time으로 배수를 계산 → 과대복원.**

물리 검증: 광교는 97% 날에 폐기(leftover)>0 + 마감할인 판매 → 마감까지 재고 있음. 진짜 조기 전체소진 0.2%.

## 목표

`bonavi_daily`의 is_stockout/stockout_time을 **물리적 leftover(폐기) 기반 진짜 최종소진**으로 재정의하고 bonavi_daily.parquet 재생성. `품절정보` 시트 의존 제거.

## 정의 (실 로더 경로만; synthetic 불변)

- **is_stockout = (made > 0) AND (waste ≤ 0)** — 생산했고 폐기 남은 게 없음 = 다 팔림/진짜 소진. 음수 폐기(반품·보정 ~3.3%)는 ≤0로 묶어 "leftover 없음"으로 처리. (기존 92% → 예상 ~3%)
- **stockout_time = 그날 마지막 실판매 timestamp** (`load_receipts`의 max), is_stockout일 때만. 아니면 NaT.
- **재고정보(made/waste) 없는 item-day → is_stockout = False, stockout_time = NaT** (판단 불가 시 품절 주장 안 함).

## 데이터 흐름 (`build_daily` 수정)

```
판매정보(load_sales/load_receipts) → sold_units + 마지막판매 timestamp(item-day별 max)
재고정보(load_inventory) → made(QT_MADE), waste(QT_OUT; 음수 clip)      [신규 조인]
  → is_stockout = (made>0)&(waste≤0)
  → stockout_time = is_stockout ? 마지막판매시각 : NaT
품절정보(load_stockouts) → build_daily에서 미사용 (함수는 남기되 호출 제거)
attach_potential_demand(신규 is_stockout/stockout_time) → potential_demand
```

## potential_demand 처리 (컬럼 유지, 재계산)

- 컬럼은 **유지**(schema 호환 + 향후). 교정된 is_stockout/stockout_time으로 자동 재계산 → **~97% 날은 sold_units와 동일**(is_stockout False라 보정 없음), ~3% item-소진일만 마지막판매시각 기준 복원.
- **명시 문서화**: 흡수(W0) 하에서 **item-level 복원은 수요 target으로 부적합**(품목 소진 시 수요가 같은 카테고리 다른 품목으로 이동 → 복원은 double-count). 진짜 lost는 **카테고리 전체 조기소진 ~0.2%뿐**. **수요 target은 adjusted_demand**([[project_stockout_time_bug_and_adjusted_demand]]). potential_demand는 여기서 그 이상 손대지 않음(폐기/전환은 별건).

## 로더 데이터 의존 (신규)

- build_daily가 **재고정보**(load_inventory: date/item_id/production_qty/waste_qty)에 의존하게 됨. store-aware 유지(현재 광교 store_gw01 단일).
- 음수 폐기 clip은 기존 `handle_negative_waste`(ingest/inventory.py) 재사용.

## 재생성 + 검증

- 재생성: `format-bonavi` CLI로 bonavi_daily.parquet 갱신.
- **검증(무언 축소 금지)**:
  - is_stockout 비율 92% → **~3%** 확인.
  - 전-품목-동일시각(09:05) 아티팩트 소멸: is_stockout 품목-일에서 stockout_time이 실판매 마지막과 정합(2h+ gap 비율 ≈ 0).
  - is_stockout==True인 item-day는 waste≤0 (정의상 자명, sanity).
  - potential_demand: ~97% == sold_units, 나머지 ≥ sold.

## 명시적 비범위 (YAGNI)

- **재검증(W0/substitution/timing KPI 등) = 하위 2**, 여기 없음.
- potential_demand 컬럼 제거·v2/v3 target 변경 없음(adjusted_demand 전환은 별건).
- synthetic.py 경로 불변.
- 타매장 미대상.
- 카테고리 전체 조기소진(~0.2%) 상방보정 flag 미구현(불필요 판정).

## 검증/테스트 (착수 전 회귀 확인)

- **절대 규칙 1 (leakage)**: 신규 정의도 당일 관측치(폐기·당일 마지막판매)라 미래 누수 없음. `test_split_leakage`/`test_features_leakage` + `assert_no_leakage` 그린 유지가 완료 조건.
- **is_stockout 값에 의존하는 테스트 색출**: 재정의로 값이 의도적으로 바뀌므로, 실데이터 daily를 만들거나 stockout 카운트/비율을 단언하는 테스트는 기대값 갱신. 착수 전 `grep is_stockout tests/`로 열거.
- 신규 로직 단위 테스트(합성 fixture, 정확값): made/waste 조합 → is_stockout 진리표, 마지막판매시각 선택, 재고정보 결측 → False.

## 성공 기준

재생성된 bonavi_daily.parquet에서 is_stockout ≈ 3%, stockout_time이 실제 마지막판매와 정합(조기 아티팩트 소멸), potential_demand ≈ sold(97%), leakage 그린. 재검증(하위 2) 착수 가능 상태.
