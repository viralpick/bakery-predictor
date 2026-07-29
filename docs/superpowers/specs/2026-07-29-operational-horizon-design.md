# 운영 리드타임 / 요일 앵커 백테스트 정렬 (설계)

작성 2026-07-29 · 브랜치 `feat/operational-horizon`

## 1. 문제

헤드라인 백테스트가 문서화된 운영 시나리오를 구현하지 않는다.

| | 정의 |
|---|---|
| 문서 (`docs/modeling_v4.md:10`) | "운영 시나리오: D=목요일 → D+4~D+10 (다음주 월~일) 발주 예측" |
| 실제 `harness/backtest_core.py` | 인덱스 기반 연속 7일 블록 + `train = date < test_start` → **리드타임 0**(첫 예측일 = 원점+1), 블록 시작 요일도 데이터 끝에 따라 임의 |

즉 백본 엔진은 "내일부터 7일"을 예측하는데, 운영은 "며칠 뒤 시작하는 다음 주"를 예측해야 한다. 리드타임 0은 운영보다 낙관적이다.

운영 정렬 버전이 아예 없던 것은 아니다. `scripts/operational_backtest.py` 에 horizon별(D+4~D+10) 개별 모델을 학습하는 구현이 숨어 있었다. 백본 밖 스크립트라 라우팅 표에서 보이지 않았고, 측정 축(모델·α·분위수·fold 수)이 canonical 스택과 전부 달라 그 수치는 헤드라인과 같은 축에서 비교할 수 없다. **canonical 스택(`category_total` + `event_prior`)에서 운영 horizon 효과는 아직 미측정이다.**

신규 운영 제약: 수요일 오전까지 발주 시스템에 전달 → 데이터는 **화요일까지만** 사용 가능, 대상은 **다음 주 월~일**.

## 2. 설계

`WindowSpec` / `windowed_backtest` 에 **opt-in 파라미터 2개**를 추가한다. 새 엔진을 만들지 않고 기존 단일 엔진을 확장한다.

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `lead_days: int` | `0` | train/prior cutoff = `test_start − lead_days`. 0 = 현 동작(리드타임 없음) |
| `anchor_dow: int \| None` | `None` | fold 블록 시작 요일(0=월). `None` = 현 동작(인덱스 기반 연속 블록) |

`anchor_dow` 는 0~6 범위를 벗어나면 스펙 로딩에서 거부(`SpecError`), `lead_days` 는 음수 거부.

**fold 경계**: `anchor_dow`가 주어지면 헬퍼 `_fold_starts_by_dow()` 가 데이터 마지막 날짜 이하에서 horizon 블록이 완전히 들어가는 마지막 `anchor_dow` 시작일을 찾고, 거기서 `horizon_days` 간격으로 과거 방향 `n_folds`개를 만든다. 범위를 벗어난 fold는 `test_df`가 비어 skip된다(기존 `min_train_rows` 미달 skip과 동일 방식).

**운영 실험 배선** — `experiments/gwangyo_train_aged5.yaml`:
`lead_days=5`, `anchor_dow=0`. 월요일 블록 시작 − 5일 = 수요일이 cutoff → train은 **화요일까지**, 첫 대상일 = 원점+6. 헤드라인(`gwangyo_default.yaml`)은 손대지 않았다. 헤드라인 교체는 architect 결정 사항이다.

fold 딕셔너리 키(`fold`/`n_train`/`n_test`/`test_start`/`test_end`/`wape`/`wpe`/`prod_pct_under`)와 `predictions` 컬럼(`date`/`fold`/`actual`/`expected`/`production`)은 불변 — `report.py` 및 소비처 계약.

## 3. 기본값 불변 근거 (엔진 동등성 hard gate)

`lead_days=0` 일 때 `cutoff = test_start_date − Timedelta(days=0) == test_start_date` 이고 `cutoff − window == test_start_date − window` 이므로, train slice와 prior history 필터가 **문자 그대로 기존 식과 동일**하다. Timestamp에 0 Timedelta를 더하는 연산은 항등이라 부동소수 오차가 개입할 여지가 없다.

`anchor_dow=None` 분기는 원본의 인덱스 산술(`test_end = total − k*test_size` → `test_start = test_end − test_size` → `df.iloc[test_start:test_end]`)을 재작성 없이 그대로 옮겼다. 추가된 `if len(test_df) == 0: continue` 는 데이터 충분성 guard(`total > n_folds*test_size + min_train_rows`) 때문에 기본 경로에서 항상 거짓이다.

따라서 `tests/harness/test_backtest_core_equivalence.py`(52-fold, rtol=1e-9, 수정 없음)가 그대로 게이트로 작동한다.

기존 호출부(`runner.py` + `scripts/` 9개)는 모두 키워드 인자로 호출하므로 새 파라미터를 넘기지 않으면 기본값이 적용된다. features 캐시 키(`_stage_key`)는 건드리지 않았다 — features는 `lead_days`/`anchor_dow` 에 의존하지 않는다.

## 4. leakage 처리

리드타임을 넣을 때 놓치기 쉬운 구멍은 **event_prior 경로**다. 기존 코드는 `hist = df[df["date"] < test_start_date]` 로 prior를 fit했는데, 리드타임이 있으면 cutoff와 test_start 사이(원점 이후) 데이터가 prior 레벨-앵커에 들어가 leakage가 된다. train과 prior history **둘 다 동일한 `cutoff`** 를 쓰도록 바꿨다. 즉 원점 이후 데이터는 어느 경로로도 새지 않는다.

## 5. 검증

`tests/harness/test_backtest_core_lead.py` (n_folds=3, feature 프레임·lead 실행은 모듈 스코프 1회 공유):

1. `lead_days=0, anchor_dow=None` 결과가 인자 없는 호출과 `folds`·`predictions` 프레임 전체 rtol=1e-9 일치.
2. spy forecaster로 `fit`에 넘어온 train_df 캡처 → 각 fold의 train 최대 날짜 == `test_start − 6일`.
3. `EventLevelPrior.fit` 스파이로 history 캡처 → 최대 날짜 ≤ `test_start − 6일` (leakage 회귀).
4. `anchor_dow=0` 일 때 모든 fold의 `test_start.dayofweek == 0`, `test_end == test_start + 6일`.
5. `_fold_starts_by_dow` 단독: 마지막 완전 블록에서 7일 간격 역방향, 블록이 데이터 끝을 넘지 않음.

`tests/harness/test_config.py` 에 기본값 불변(0/None) + opt-in 파싱 + 범위 거부 4건 추가.

실측 확인: 광교 feature 프레임 1791행 / 2021-01-29~2025-12-31. `anchor_dow=0` 은 2025-12-22(월) 블록부터 역방향으로 잡히고, `lead_days=5` 에서 prior history 최대 날짜는 각 fold의 `test_start − 6일`(2025-12-19 / 12-12 / 12-05)로 관측됐다.

## 6. 미측정 / 후속

- canonical 스택에서 운영 horizon(`lead_days=5`, `anchor_dow=0`)이 WAPE·폐기율에 주는 영향은 **아직 측정하지 않았다.** `experiments/gwangyo_train_aged5.yaml` 52-fold 실행이 다음 단계.
- 헤드라인 교체 여부는 architect 결정.

### ⚠️ `lead_days` 는 train cutoff만 옮긴다 — test-row feature는 아직 아니다 (실측 확인)

feature 프레임(46 컬럼)에는 타깃의 자기회귀 feature가 들어 있다:
`adjusted_demand_unit_{lag1,lag7,lag14,lag28,ewma7,ewma28}`. 이 값들은 fold와 무관하게 **전체 프레임에 대해 한 번** 계산되고, backtest는 행을 train/test로 자르기만 한다.

결과적으로 `lead_days=5` 에서도 test 첫 행의 `lag1` 은 `test_start − 1일`, 즉 **cutoff + 4일**(원점 이후) 실측이다. 따라서 이번 변경으로 정렬된 것은 **모델 학습 시점**이며, **feature 가용성**은 여전히 리드타임을 무시한다.

이는 이번 변경이 만든 결함이 아니라 기존 엔진의 성질이다(현 헤드라인 `lead_days=0` 에서도 horizon 2일차 이후의 `lag1` 은 test 구간 내부 실측이다 — 매일 재관측을 암묵 가정). `scripts/operational_backtest.py` 가 horizon offset별로 **별도 모델**을 학습한 이유가 바로 이것이다.

즉 `gwangyo_train_aged5.yaml` 이 지금 측정하는 것은 "**학습 데이터가 5일 오래됐을 때의 손실**"이며, "완전한 운영 시나리오"는 아니다. 완전 정렬에는 horizon별 feature 재구성(D+offset용 lag를 cutoff 이전 값으로 고정)이 필요하고, 이는 `backtest_core` 확장이 아니라 **feature 레이어 작업**이다 — 별도 태스크로 남긴다. 리포트에서 이 실험을 인용할 때 이 한계를 반드시 병기해야 한다.

**실험 이름을 `operational` 로 쓰지 않은 이유(2026-07-29 결정)**: 파일명은 인용될 때 그대로
따라다닌다. `gwangyo_operational` 이면 "운영 성능"으로 읽히고, 위 한계가 탈락한 채 수치만
돌아다닌다 — 이 프로젝트가 반복해서 당한 측정 축 사고와 같은 형태다. 그래서 측정 대상을
이름에 박았다: **`gwangyo_train_aged5`**(학습 데이터 5일 노후화). 이 실험의 열화는
**운영 열화의 하한(lower bound)** 이다 — feature가 아직 원점 이후를 보므로 완전 정렬 시
열화는 이보다 커진다.
