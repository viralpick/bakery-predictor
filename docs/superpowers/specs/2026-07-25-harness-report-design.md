# Harness 실험 report 표면 설계 (Phase 2-2)

**날짜**: 2026-07-25
**브랜치**: feat/harness-report
**선행**: Phase 2-1 distributional 배선(#54, main `f996f74`). 마스터 로드맵 2단계.

## 목표

harness `ExperimentResult`를 사람이 볼 수 있는 **자기포함 HTML report**로 만든다(plotly). forecaster 비교 + 예측 오버레이 + fold WAPE에 더해, **품목별 매진 실측 뷰**(매진률·매진 median t)를 포함한다. `harness-run`이 자동 생성한다.

## 배경: 매진 두 층위 (검증 완료 2026-07-25)

사용자 요구 = 매진 2축: **① 품목별 매진률**, **② 매진 median t**. 그리고 **전체매진**(발주<실수요). 세 가지는 소스·층위가 다르다:

| 지표 | 정의 | 소스 | forecaster별 다름? |
|---|---|---|---|
| 전체매진 위험 | 발주(production) < 실수요(actual) 비율 = 헌장 관점① 모델정의 | `metrics_from_preds`의 `stockout_risk` (이미 계산됨) | **O** (production이 forecaster별) |
| 품목별 매진률 | 품목별 완판일/활성일. 완판 = production_qty>0 & waste_qty≤0 | store_daily(item-day 실측), `assign_stockout_fields` | X (관측, 실험 무관) |
| 매진 median t | 완판 item-day의 매진시각(=마지막 실판매) median | store_daily `stockout_time` | X (관측) |

**검증된 사실:**
- 품목별 매진시각 도출은 신뢰 가능: 완판행 stockout_time 100% 채워짐 / 비완판행 100% NaN (사용자 요구 "당일 실제 매진 품목만 매진시각" 정확 충족).
- 광교 신 데이터 item-day 매진율 **0.151**은 진짜(조인 버그 아님): `aggregate_daily`가 production/waste 정상 병합, 완판 15%=나머지 85% 폐기>0(과잉생산, 카테고리 폐기 97%와 층위 정합), 매진시각 hour 분포 현실적(저녁 피크, **median 18시**, 22시 이후 급감).
- 따라서 실패 중인 `test_store_daily_redefine`의 `0.50<rate<0.70` 기대가 **stale(구 데이터 기준)** → 0.151로 재baseline이 이 스텝 in-scope(canary).
- 품목별 매진률 raw median=0.0 (1150품목 중 다수 희소/미완판) → **분포+활성필터+top으로 표시**(raw median 무의미).

## 아키텍처

### 1. `src/bakery/harness/report.py` (신규)

```python
def build_report(result: ExperimentResult, *, out_path: Path, store: str | None = None) -> Path
```
- `ExperimentResult`(in-memory) 소비 → 자기포함 HTML(plotly CDN). build_dashboard의 `fig_to_div` 패턴 재사용(plotly→div, `include_plotlyjs='cdn'` 1회 + 이후 False).
- **plotly는 report.py에만 의존** — runner/backtest_core 코어는 viz 무의존 유지.
- `store`가 주어지면 품목별 매진 실측 섹션 포함, None이면 스킵(백테스트-only report도 유효).

**섹션 4종:**
1. **비교표 + 전체매진** — `result.comparison`을 plotly Table. `stockout_risk` 컬럼을 **"전체매진 위험(발주<실수요)"** 헤더로 relabel. forecaster당 1행.
2. **Fold WAPE** — forecaster를 series로 묶은 grouped line/bar(직접 비교). `result.runs[f].fold_metrics["wape"]`.
3. **예측 오버레이** — forecaster별 subplot: date축 actual/expected/production 3선(weekly_overlay 패턴). `result.runs[f].predictions`.
4. **품목별 매진 실측** (store 있을 때) — `_soldout_view(store)`가 store_daily 소싱:
   - **매진 median t**: 완판 item-day `stockout_time` hour의 median(단일 KPI 강조) + hour 분포 히스토그램(헌장 "매진 time median").
   - **품목별 매진률**: 활성일 ≥ N(예: 30)인 품목만 필터 → 매진률 히스토그램 + top 20 품목 바. (raw median 0.0 회피, 정직한 표시.)

### 2. store 해석 헬퍼

report.py 내부 `_store_daily_for(store_id)`:
- `scripts/store_daily.py`의 매장 dict를 역참조(`store_id="store_gw01"` → cd `"1000000047"`) 후 `build_store_daily(cd, store_id, exclude_bulk=True)` 호출.
- scripts import는 Phase 1 test들이 쓰는 `sys.path.insert(0,"scripts")` 패턴 재사용. 미등록 store → None 반환(섹션 스킵, 경고).

### 3. CLI 자동 생성

`cmd_harness_run`이 `run_experiment` 후 `build_report(result, out_path=out/<exp>/report.html, store=spec.data.store)` 호출 → report.html 산출. 콘솔에 경로 출력 1줄 추가. (runner는 순수 유지, viz는 report.py+cli에만.)

### 4. metrics 이름 정합

`config.DEFAULT_METRICS`를 실산출(`metrics_from_preds`)에 맞춤:
```python
DEFAULT_METRICS = ["wape", "wpe", "stockout_risk", "surplus_mean_units", "surplus_rate"]
```
계산 불가 3종(soldout_median/stockout_item_rate/shortfall_day_rate — 카테고리 총량 레벨에 없는 매진시각/item-level 요구) 제거. spec.metrics 기반 컬럼 선택(필터링)은 향후.

### 5. test_store_daily_redefine 재baseline

`tests/test_store_daily_redefine.py::test_build_store_daily_uses_redefinition`의
`assert 0.50 < rate < 0.70` → `assert 0.10 < rate < 0.20`(실측 0.151 반영). 주석도 "옛 92%→재정의 ~60%대"에서 "신 데이터 재정의 ~15%(폐기 85% 과잉생산)"로 갱신. 단위 test(`test_assign_stockout_fields_redefinition_exact`)와 leak test는 불변(이미 통과).

## Acceptance (검증)

1. **build_report 단위 test** (`test_report.py`, 빠름 — 백테스트 없음): 더미 `ExperimentResult`(2 forecaster, 소형 predictions/fold DataFrame 직접 구성) + `store=None`으로 호출 → report.html 존재 + forecaster명·WAPE값·"전체매진" 라벨·plotly div 포함 단언(정확 문자열).
2. **soldout view 단위 test**: 소형 store_daily 형태 DataFrame(완판/비완판 혼합) 주입 → 매진 median t = 알려진 값(정확 `==`), 품목별 매진률 필터 동작. (build_store_daily 전체 실행 회피 위해 `_soldout_view`를 DataFrame 인자로 분리 — 테스트 가능성.)
3. **CLI test 확장**: 기존 test_cli_harness(gwangyo_default, category 1종, store_gw01)에 report.html 존재 + 품목별 매진 섹션 문자열 단언 추가.
4. **config test**: DEFAULT_METRICS 정합 반영(내용 단언 추가).
5. **test_store_daily_redefine PASS**(재baseline) — 오래된 실패 해소.
6. 전체 스위트 green.

## 마이그레이션 스텝

- `DEFAULT_METRICS` 변경: 소비처 grep 완료 = config.py 정의 + __init__ re-export + test_config(내용 미단언, 안전). ExperimentSpec.metrics 기본값이 이걸 씀 → runner/report 무영향(현재 metrics 필터링 미사용).
- `build_report` 신규 심볼 → __init__ re-export 추가.
- `cmd_harness_run` 확장: 반환/기존 표 출력 유지 + report 생성 1줄 추가(후방호환).

## Open risks

1. plotly 6.7.0 정식 의존성 확인 완료. `fig_to_div`는 build_dashboard서 복사(scripts 의존 회피 위해 report.py에 최소 재구현).
2. `_soldout_view`를 DataFrame 인자로 분리해 단위 테스트 가능하게(build_store_daily 실행은 CLI/통합 경로에서만).
3. report.html 크기: plotly CDN(embed 아님)이라 작음. 예측 오버레이 3년치 3선×forecaster는 수만 점 — plotly 처리 가능하나 필요시 다운샘플(일별이라 ~1100점/forecaster, 문제없음).

## 범위 밖 (다음 로드맵 단계)

- build_dashboard 원시데이터 EDA 승격 = 6단계(가설검증+데이터분석).
- spec.metrics 기반 컬럼 선택/필터.
- point/composite forecaster 실행. event_priors 완전 단일화. STAGES 상수 정합.
