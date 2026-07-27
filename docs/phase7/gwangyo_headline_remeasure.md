# 광교 헤드라인 KPI 재측정 — 새 166 canonical + 신규 closing 소스 (Phase 7 Task 3)

## 측정 경로 (필수 명시)

이 리포트의 모든 수치는 **고객 지표 경로**(`bakery harness-run experiments/gwangyo_default.yaml`
→ `forecaster=[category_total]`, `layers=[event_prior]`, `event_priors=gwangyo`,
`target=adjusted_demand_unit`, expanding window 52-fold, `production_q=0.85`,
`alpha=0.8`)에서 나온 **카테고리 총량(category_total) + event_prior** 수치다.
**item-level daily WAPE가 아니다.**

closing 소스는 Task 0(커밋 `6ef3138`)이 재배선한 신규 클린 parquet
(`data/interim/sales_lines_clean.parquet`)이며, 실행 전 아래로 존재를 재확인했다:

```
uv run python -c "from bakery.analysis.discount import CLEAN_PARQUET_DEFAULT as p; print(p, p.exists())"
→ data/interim/sales_lines_clean.parquet True
```

## 실행

```
uv run bakery harness-run experiments/gwangyo_default.yaml
```

출력: `reports/gwangyo_default/config_resolved.yaml`,
`reports/gwangyo_default/category_total/{predictions.csv,fold_results.csv,metrics.json}`,
`reports/gwangyo_default/comparison.csv`, `reports/gwangyo_default/report.html`
(모두 `reports/`, gitignored — 원본 그대로 보존, 본 문서만 커밋).

## 새로 관측한 raw 값

`reports/gwangyo_default/category_total/metrics.json` (n_test=364, 52-fold × horizon 7일):

| metric | value |
|---|---|
| n_test | 364 |
| wape | 0.07724308250273677 |
| wpe | 0.006434857882670159 |
| stockout_risk | 0.21703296703296704 |
| surplus_mean_units | 20.6431527226923 |
| surplus_rate | 0.08334303019849418 |

`reports/gwangyo_default/report.html` 4절(품목별 매진 실측, forecaster 무관, store_daily 관측치):

- 매진 median t = 18.0시
- 전체 매진율(완제품 검열 포함 희석값) = 0.151, n_soldout = 46439

## 지표 매핑 — 어떤 새 metric이 어떤 옛 headline 항목에 대응하는지

| 옛 headline 항목 | 새 metric 이름 | 대응 가능? | 근거 |
|---|---|---|---|
| 총량 WAPE (naive 8.19% / 우리 8.03%) | `wape` | **가능, 동일 formula/파이프라인** | n_test=364로 정확 일치(52-fold×7일). 옛 8.19/8.03도 같은 harness 경로(category_total+prior, 광교 52-fold, n=364)에서 나온 수치(`project_current_model_level` 메모리, 2026-07-15). 데이터만 146→166+구closing→신규closing으로 바뀜. |
| 폐기 −33~40% (vs 아띠제 실생산) | `surplus_rate`(참고용) | **불가 — 지표 정의가 다름** | 옛 수치는 `scripts/unified_policy_kpi.py` 계열로 **아띠제 실생산(QT_MADE)을 ground truth**로 두고 여러 발주 마진 정책(quantile/nk/conformal)을 비교한 *상대적* 폐기 절감률. 새 `surplus_rate`는 harness 자체 `production_q=0.85` 정책의 **자기참조적** 과잉생산율((production−actual)⁺/actual)이며 아띠제 실생산과 무관. 두 수치를 같은 축에 놓고 delta를 계산하면 오도됨. |
| (headline 표에 없음) | `stockout_risk` | 대응 없음 | report.py 라벨"전체매진 위험(발주<실수요)". 모델 발주가 실수요보다 부족한 fold-day 비율. 옛 메모리의 "카테매진①"(quantile q0.85=0.406)과 형태는 비슷하나 다른 alpha(0.7→0.8)·다른 품목수·다른 파이프라인(unified_policy_kpi vs harness backtest_core)이라 직접 diff 금지. |
| (headline 표에 없음) | 매진 median t / 전체 매진율(관측) | 대응 없음 | store_daily 관측치(forecaster 무관). 옛 146 headline 문서에 비교 대상 수치 없음. 0.151은 완제품 검열 희석값(메모리 `project_soldout_rate_dilution` 참조) — 생산품목 실질 매진율(≈0.605)과 다름. |

## Delta 표

| 지표 | 옛 146 headline | 새 166+신규closing | delta |
|---|---|---|---|
| 총량 WAPE — naive (seasonal_naive) | 8.19% | *(이번 실행에서 재측정 안 함 — yaml이 `category_total`만 실행)* | N/A |
| 총량 WAPE — 우리(category_total+event_prior) | 8.03% | **7.72%** (wape=0.07724) | **−0.31pp** (상대 −3.9%) |
| 폐기율 vs 아띠제 실생산 | −33~40% | *(재측정 안 함 — unified_policy_kpi.py는 harness-run 범위 밖)* | N/A |
| 폐기 self-referential proxy (surplus_rate, production_q=0.85) | (옛 수치 없음) | 8.33% (surplus_mean=20.64 units/fold-day) | N/A — 지표 정의 다름, 참고치로만 병기 |
| 카테고리 매진위험 (stockout_risk, prod<actual) | (옛 수치 없음) | 21.7% | N/A |
| 매진 median t (관측) | (옛 수치 없음) | 18.0시 | N/A |
| 전체 매진율 (관측, 완제품 희석) | (옛 수치 없음) | 0.151 | N/A |

## 경계 명시 (필수)

이 delta는 146→166 품목/타깃정의 변화 + closing 소스 교체분이며, 날짜는 여전히
2021~2025-12로 2026 전향(prospective) 검증이 아니다.

추가로: 유일하게 methodologically 클린한 delta는 **총량 WAPE(우리 모델) 8.03%→7.72%**다
(동일 harness 경로·동일 fold 구성, n_test 정확 일치로 확인). 이 값이 개선됐다는 것이
"모델이 더 정확해졌다"를 의미하지 않는다 — 166개 품목 재정의와 신규 closing 소스가
분모(actual=adjusted_demand_unit)와 잔차 구조 자체를 바꿨으므로, "새 canonical 기준
헤드라인 재설정"으로 읽어야 한다. **폐기(1차 KPI, vs 아띠제 실생산)는 이번 재측정
범위 밖**이다 — `harness-run`은 자기참조적 `surplus_rate`만 산출하며, 아띠제 실생산
대비 진짜 폐기 비교는 `unified_policy_kpi.py`류 별도 파이프라인의 재실행이 필요하다
(후속 과제로 남김). 매진(2차 KPI)도 마찬가지로 옛 헤드라인 표에 비교 대상 수치가
없어 delta를 계산하지 않고 새 관측치만 병기했다.
