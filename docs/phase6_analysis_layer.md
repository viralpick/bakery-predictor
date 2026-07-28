# 데이터분석 + 가설검증 레이어 (Phase 6)

`bakery analysis-run <yaml>` 단일 진입점. 입력 데이터 분석 5종 + 가설 14종을 YAML로 on/off.

## 사용법

```bash
uv run bakery analysis-run experiments/analysis_gwangyo.yaml            # 광교
uv run bakery analysis-run experiments/analysis_multistore.yaml         # 4매장
```

산출: `reports/analysis/<name>/analysis_report.html` + 항목별 `<name>__<table>.csv`.

## 경계 (harness-run과의 분리)

| | harness-run | analysis-run |
|---|---|---|
| 대상 | 예측 성능 | 입력 데이터 + 가설 |
| 모델 실행 | 함 | **안 함** |
| preds | 생산 | `spec.predictions`로 **읽기만** |

preds 의존 4종(`seasonal_bias`/`weather_bias`/`weekday_bias`/`event_prior_validation`)은
artifact가 없으면 리포트에 `(preds 필요 — 미실행)`으로 표기되고 실행되지 않는다.

광교 전용 소스(`category_daily`)를 쓰는 `holiday_premium`/`month_dow_adjust`는
`needs_single_store` 게이트가 걸려 있어 `analysis_multistore.yaml`로 돌리면
`(단매장 spec 필요 — 미실행)`으로 표기되고 스킵된다 — 광교 수치가 4매장 분석으로
잘못 라벨링되는 것을 막기 위한 의도된 동작이다.

### `event_prior_validation` A/B 모드 전제조건

`reports/`는 gitignored라 clone 직후에는 `reports/gwangyo_no_prior/category_total/predictions.csv`가
없다. 이 상태에서는 A/B(baseline vs prior)가 아니라 단일 artifact 모드로 조용히 강등되고, 그 사실이
verdict에 표기된다. A/B 모드로 실행하려면 먼저 baseline artifact를 만든다:

```bash
uv run bakery harness-run experiments/gwangyo_no_prior.yaml
```

## 등록 목록 (19종)

- 데이터 분석 5종: `sales_distribution`, `category_mix`, `waste_rate`, `waste_alpha_identity`,
  `overproduction_breakdown`
- 가설 14종: `demand_absorption`, `substitution`, `stockout_lost_demand`, `closing_discount`,
  `other_discounts`, `discount_regime`, `seasonal_bias`, `weather_bias`, `weekday_bias`,
  `holiday_premium`, `month_dow_adjust`, `popularity_stockout`, `modeling_v4_assumptions`,
  `event_prior_validation`

## 이식 제외 (DEPRECATED)

`diag_anchor_gh`, `diag_chuseok_gh`, `diagnose_conformal_residual` — v5 conformal
구간예측 계열. 점추정+위험수치 전환으로 폐기됐다. spec에 쓰면 `AnalysisSpecError`.

## Frozen golden fixture

`reports/`는 gitignored이므로 예전에는 `holiday_premium`/`weekday_bias`의 회귀 게이트가 로컬에
`reports/` 산출물이 남아있는 머신에서만 통과하고, 신규 clone에서는 조용히 스킵됐다. 이를 막기 위해
아래 두 파일을 git-tracked fixture로 고정했다 — 없으면 스킵이 아니라 **fail**한다.

| 파일 | 생성일 |
|---|---|
| `tests/fixtures/frozen/raw_adjusted_series.csv` | 2026-07-16 |
| `tests/fixtures/frozen/track3_fresh_preds.parquet` | 2026-07-18 |

## 이식 대조 기록

측정일 2026-07-28, 광교 canonical vintage 기준.

| 항목 | 게이트 형태 | 출처 수치 | 핸들러 수치 | 판정 일치 |
|---|---|---|---|---|
| demand_absorption | 동일 vintage 실측(레거시 `scripts/store_daily.py` 대조) | — (출처 수치 미제공, 대조는 판정 일치 여부로만 수행) | `max\|Δβ\| = 0.1438`, 게이트 판정 불일치 2건(`pastry` @ `store_ss01`, `store_gh01`) | ⚠️ 부분 불일치(원인 규명됨, 아래 참조) |
| holiday_premium | frozen-input golden | 평일 n=71 median 1.25 | frozen golden 재계산: 평일 공휴일 n=71 median lift 1.25 IQR[1.10,1.38] / 주말 n=22 median 0.89 IQR[0.78,1.00]. streak: 고립 n=20 median 1.345475, "2" n=0(NaN), 연휴 n=51 median 1.190351. event_ranking 21행, by_holiday 93행 | ✅ 판정 라벨 동일("지지"). fresh vintage에서 수치만 미세차: 주말 q75 1.00→0.99, 주말 pct −11.3%→−10.7% |
| weekday_bias | frozen-input golden | base waste 0.047616 | frozen golden 재계산: n=1090일, 월·수 비중 0.285321, base(expected) waste 0.047616. w_target=0.06 기준 trim 0.02 → gap_freq −0.0009174311926605228 / gap_mag −0.0001567963417563184, trim 0.03 → +0.00458715596330278 / −4.3062759390345706e-05, trim 0.04 → +0.007339449541284404 / +0.0002116974752705003 | ✅ |
| closing_discount | 출처 대조 | A1/A2/A3 3방식(Task 8 스펙) | α 구간 `[nan, 1.000]`, A1 0.847 / A2 nan(degenerate) / A3 0.279 | ✅ (A2 degenerate는 출처와 동일한 알려진 현상) |
| stockout_lost_demand | 출처 대조 — ⚠️ 무영향 가설의 대체 지표 아님(아래 캐비앗 1 참조) | 광교 `lost_share_of_sold` 크기 측정 | 광교 19.6%(광화문 40.2% / 삼성 35.4% / 메세나 26.8%) | 해당 없음(크기 보고, 판정 없음) |

| other_discounts | 출처 대조 | 마감 외 할인의 존재·규모 | **마감 외 할인 272,319건 / 코드 83종**, 시각 분포는 09~10시대에 집중(09시 5,640 / 10시 4,635) | ✅ (마감할인만 있는 매장이 아님이 확인됨) |

`other_discounts`는 빈 경로가 아니라 실데이터가 있다 — 광교는 마감할인(0069/0077) 외에도
83종의 할인코드가 27만 건 이상 붙는다. 시각 분포가 저녁이 아니라 **오전(09~10시)에 몰리는 것**은
이들이 마감 성격이 아님을 시사한다(성격 분류는 후속 과제). 마감할인 α 추정(`closing_discount`)이
마감 코드만 대상으로 하는 이유가 여기 있다 — 전체 할인을 마감으로 뭉뚱그리면 α가 오염된다.

### demand_absorption 불일치 원인

`max|Δβ| = 0.1438`, 게이트 판정이 갈리는 2건은 `pastry` 카테고리 × `store_ss01`/`store_gh01`.
원인은 `src/bakery/data/bonavi_loader_v2.py:52`의 `EXCLUDE_CATEGORIES = frozenset({"salad"})`
(2026-07-23 결정, salad를 판매/폐기 양쪽에서 제외)를 canonical 경로는 적용하지만 레거시
`scripts/store_daily.py`는 적용하지 않아서다 — salad가 `other_cat_sold` 통제 변수에 섞여 들어가
계수가 흔들린다. canonical(analysis-run 핸들러)이 진실이고, 레거시 스크립트는 소비처가 17곳
걸려 있는 코드라 이번 이식 범위 밖이라 수정하지 않았다.

### 수치 등가가 **불가능한** 항목과 근거

- `sales_distribution`/`category_mix`/`waste_*`/`overproduction_breakdown` — 레거시 eda01~05는
  `data/internal/v2/` 원본 시트를 다른 필터로 읽었다(FG_ITEM=='SS', beverage/etc 포함).
  canonical 재표현이므로 게이트는 구조 불변식(비중 합=1.0, 폐기율∈[0,1], 항등식 잔차)이다.
  실측: 광교 `waste_rate` 0.125324, `zero_frac` 0.903111. 4매장 `zero_frac` 0.903~0.928(편차
  0.025), `waste_rate` 0.125~0.159. 삼성타운 `n_carry_in`=3047 / `carry_in_units`=−8026.
  `waste_alpha_identity` 항등식 잔차 0 비율 = 전체 91.83% / `out<0` 88.80%(8,108행 중 908행
  위반, |잔차| max 25.0) / `out≥0` 91.92%. 재계산 `made−(normal+closing)−out`은 저장된
  `identity_diff`와 **100% 일치**.
- `month_dow_adjust` — 출처는 레거시 직독 + α=0.5, 핸들러는 canonical + 헌장 α=0.8. 실측:
  84/84 칸 채워짐, 최악 칸 9월 수요일 −4.111%, 칸간 편차 2.638%p.
- preds 의존 4종 — 출처는 비-canonical 엔진(`store_predictive_power`) 캐시. canonical
  harness preds와 수치가 다르므로 동결 artifact에만 수치 게이트를 걸고 그 외엔 방향/판정만.
  실측:
  - `discount_regime`: `depth_invariant`, `closing_share` β=0.0022 CI95[−0.0090,0.0135],
    placebo 0건, n=21204.
  - `popularity_stockout`: ablation spearman 0.996(n=37), `n_top_changed`=0,
    `max_abs_share_delta`=0.0030, `adj_stockout` min 1.0 / max 1.1842 / std 0.0504.
  - `substitution`: nested λ pastry 0.999878 / sandwich 0.999842 / bread 0.996803 /
    cake 0.992605 / **sweets 0.715539**, NaN 0/5. DiD 1914행 β 평균 −0.0023428383871375007.
    RD 3944행 outflow 평균 0.588. 2회 독립 실행 결과가 마지막 자리까지 동일(결정적).
  - `modeling_v4_assumptions`: `1-1-b` cv비 0.8548 vs <0.70 **FAIL**(n=157) / `2-1-a` 0.0129
    vs <0.05 PASS(n=149) / `2-1-b` 0.0068 vs <0.10 PASS(n=46 실이벤트) / `basket` 0.2817 vs
    >0.30 **FAIL**(n=256,345). verdict: "부분 지지 — 4가정 중 2건 통과, 미통과:
    ['1-1-b','basket']". **현재 vintage에서 두 가정이 실패한다 — 각주가 아니라 후속 조치가
    필요한 실측 결과로 기록한다.**
  - `seasonal_bias`: 주말 WPE 차 −3.1494855109920525 CI[−5.1486187464689595,
    −0.9393701135677681] n=104 → **신호**(CI가 0을 배제). 여름 −1.0432285451348726
    CI[−2.8541313230507974, +1.0760712678132356] n=122 → noise(CI가 0 포함).
  - `weather_bias`: `is_heatwave` +1.984 CI[−3.021,+7.169] n=17 / `is_coldwave` −4.566
    CI[−9.469,+0.998] n=11 / `is_heavy_rain` −0.829 CI[−4.204,+2.710] n=14 — 전부 CI가 0을
    포함한다. ⚠️ n=11~17은 **underpowered**(검정력 부족)이지 "효과 없음"의 증거가 아니다 —
    캐비앗 2 참조.
  - `event_prior_validation`: A/B 모드(baseline artifact 존재 시) 이벤트일 WPE −2.5707
    (baseline −19.7625), 개선 −17.19%p, 매진률 0.0 vs baseline 66.67, **n=3**. 비이벤트일
    WPE 0.6775 n=361(baseline 동일). baseline artifact 364행 WAPE 0.0788;
    2025-12-25 expected 240.991106 vs prior 315.833938, actual 307.6; 364행 중 3행만 상이.
    ⚠️ n=3은 **underpowered** — 캐비앗 2 참조.

## 캐비앗 (반드시 함께 읽는다)

1. **`stockout_lost_demand`은 "매진 무영향" 가설의 대체 검정이 아니다.** 이 핸들러는
   `lost_share_of_sold` 크기만 보고한다(광교 19.6%). "매진이 매출에 영향을 주는가"의 진짜 검정은
   `scripts/verify_stockout_revenue_4stores.py`의 traffic 통제 4-layer OLS
   (`smf.ols("log_rev ~ n_stockout + C(dow) + C(month) + yr", data=d)`)이며, **이번 이식에
   포함하지 않았다 — 백로그.** 기존 결론(2026-06-03, 2026-07-10 재검증: 3/4 매장 무영향,
   메세나폴리스만 약신호)은 analysis-run과 서로 다른 것을 측정하므로 충돌이 아니다.
2. **Underpowered ≠ refuted.** `weather_bias`의 세 세그먼트(n=17/11/14)와
   `event_prior_validation`의 A/B 모드(n=3)는 표본이 작다. CI가 0을 포함한다는 것은 그
   표본 크기에서 "효과 없음을 확인했다"가 아니라 "검정력이 부족해 판별 불가"라는 뜻이다.
   위 표/실측 옆에 이 표시를 명시했다.
3. **레거시 EDA 5종·`month_dow_adjust`·preds 의존 4종은 수치 등가가 원천적으로 불가능**하다
   (읽는 원본/필터/타깃/α가 다르다). 게이트는 구조 불변식 또는 방향/판정 일치로 대체했다 — 위
   "수치 등가가 불가능한 항목" 절 참조.
4. **frozen golden fixture는 git-tracked다** — `reports/`가 gitignored라 신규 clone에서
   회귀 게이트가 조용히 스킵되던 문제를, fixture 부재 시 fail로 바꿔 막았다. 위 "Frozen golden
   fixture" 절 참조.
5. **`event_prior_validation`의 A/B 모드는 사전 실행이 필요하다** — 위 "A/B 모드 전제조건" 절
   참조. `reports/gwangyo_no_prior/category_total/predictions.csv`가 없으면 단일 artifact
   모드로 조용히 강등된다.
6. **`modeling_v4_assumptions`는 현재 vintage에서 4가정 중 2건(`1-1-b`, `basket`) 미통과다.**
   각주가 아니라 후속 조치가 필요한 실측 결과로 취급한다.
7. **DEPRECATED 3종**(`diag_anchor_gh`, `diag_chuseok_gh`, `diagnose_conformal_residual`)은
   `AnalysisSpecError`로 거부된다 — v5 conformal 폐기 계열이므로 이식하지 않았다.
