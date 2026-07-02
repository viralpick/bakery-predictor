# q_order_top 재설계 (rank→explain 체인) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** grounding eval의 q_order_top을 "관측 매진 위험 1위 품목의 발주 분해"(rank→explain 2-step 체인)로 재설계하고, 관측 매진시각 기반 랭킹 함수 `rank_stockout_earliness`를 추가한다.

**Architecture:** 신규 OntologyFunction 1개(순수 pandas, 신규 모델링 0) → ToolSpec/dispatch 노출 → Q셋 수정(q_order_top 재설계 + q_rank_earliness 추가, 10→11문항) → decomposition 채점을 item_id+order_qty로 강화. 기존 MC 기반 `rank_stockout_risk`와 S5/S6/S7 closed-loop는 무접촉.

**Tech Stack:** Python 3.12, pandas, pytest, uv. 스펙: `docs/superpowers/specs/2026-07-02-q-order-top-redesign-design.md`

## Global Constraints

- 위험 점수 = 일평균 손실 영업시간: 품목별 전체 일에 대해 `mean(max(close_hour − stockout_time의 시각, 0))`, 매진 없는 날 = 0 (스펙 D1)
- `close_hour: int = 22` 라벨된 가정 (스펙 D3) — 매직 넘버 금지 규칙에 따라 모듈 상수 `DEFAULT_CLOSE_HOUR = 22`
- 정렬 키 `(lost_hours_per_day desc, item_id asc)`, 기간 내 매진 전무 시 `ValueError` (스펙 D4)
- decomposition 채점 = item_id exact match AND qty `abs ≤ 1e-6` 둘 다 (스펙 D5)
- 기존 `rank_stockout_risk`·q_rank_top1/3/5·decision layer·closed-loop 무변경 (스펙 D2)
- 함수 30줄 이내, 중첩 3단계 이하, guard clause 우선 (글로벌 code-quality 규칙)
- 검증됨: synthetic store_A(=`_ctx` 선정 매장)에 매진 4,379행(9~21시), 점수 1위 item_C03≈1.52 vs 2위 1.40 — gold 비퇴화

---

### Task 1: `rank_stockout_earliness` OntologyFunction + registry 등록

**Files:**
- Modify: `src/bakery/ontology/functions.py` (rank_stockout_risk 바로 뒤에 함수 추가, 상수 추가, FUNCTION_REGISTRY 항목 추가)
- Test: `tests/test_ontology_functions.py` (append)

**Interfaces:**
- Consumes: `_period_slice(daily, store_id, start, end)` (functions.py 기존 헬퍼)
- Produces: `rank_stockout_earliness(daily, store_id, period, k=5, *, close_hour=DEFAULT_CLOSE_HOUR) -> pd.DataFrame[item_id, lost_hours_per_day, stockout_days, days]` — Task 2(dispatch), Task 3(build_gold)이 이 시그니처·컬럼명을 그대로 사용

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_ontology_functions.py` 상단 import에 `rank_stockout_earliness` 추가:

```python
from bakery.ontology.functions import (
    FUNCTION_REGISTRY,
    demand_diff_by_condition,
    explain_order,
    rank_stockout_earliness,
    rank_stockout_risk,
    waste_cost,
    what_if,
)
```

파일 끝에 테스트 5개 추가:

```python
def _earliness_fixture() -> pd.DataFrame:
    """2일 × 4품목 손계산 fixture. close_hour=22 기준:
    early: (22-10 + 0)/2 = 6.0 / late: ((22-20)+(22-21))/2 = 1.5
    afterclose: 23시 매진 → clamp 0 / never: 매진 없음 → 0
    """
    return pd.DataFrame({
        "store_id": ["s1"] * 8,
        "item_id": ["early", "early", "late", "late",
                    "afterclose", "afterclose", "never", "never"],
        "date": ["2026-01-01", "2026-01-02"] * 4,
        "stockout_time": [
            pd.Timestamp("2026-01-01 10:00"), pd.NaT,
            pd.Timestamp("2026-01-01 20:00"), pd.Timestamp("2026-01-02 21:00"),
            pd.Timestamp("2026-01-01 23:00"), pd.NaT,
            pd.NaT, pd.NaT,
        ],
    })


def test_rank_stockout_earliness_hand_calc():
    ranked = rank_stockout_earliness(
        _earliness_fixture(), "s1", ("2026-01-01", "2026-01-02"), k=4)
    assert list(ranked.columns) == ["item_id", "lost_hours_per_day", "stockout_days", "days"]
    # 동률(0.0)인 afterclose/never는 item_id 오름차순
    assert list(ranked["item_id"]) == ["early", "late", "afterclose", "never"]
    assert ranked["lost_hours_per_day"].tolist() == pytest.approx([6.0, 1.5, 0.0, 0.0])
    assert ranked["stockout_days"].tolist() == [1, 2, 1, 0]
    assert ranked["days"].tolist() == [2, 2, 2, 2]


def test_rank_stockout_earliness_topk_cuts():
    ranked = rank_stockout_earliness(
        _earliness_fixture(), "s1", ("2026-01-01", "2026-01-02"), k=2)
    assert list(ranked["item_id"]) == ["early", "late"]


def test_rank_stockout_earliness_no_stockout_raises():
    frame = _earliness_fixture()
    frame["stockout_time"] = pd.NaT
    with pytest.raises(ValueError, match="no stockouts"):
        rank_stockout_earliness(frame, "s1", ("2026-01-01", "2026-01-02"), k=3)


def test_rank_stockout_earliness_synthetic_smoke(dataset, store_period):
    store_id, period = store_period
    ranked = rank_stockout_earliness(dataset.daily, store_id, period, k=3)
    assert len(ranked) == 3
    assert ranked["lost_hours_per_day"].is_monotonic_decreasing
    assert (ranked["lost_hours_per_day"] >= 0).all()
    # 결정론: 두 번 호출 결과 동일
    again = rank_stockout_earliness(dataset.daily, store_id, period, k=3)
    assert list(ranked["item_id"]) == list(again["item_id"])


def test_rank_stockout_earliness_registered_read_side():
    spec = FUNCTION_REGISTRY["rank_stockout_earliness"]
    assert spec.side == "read"
    assert spec.impl is rank_stockout_earliness
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_ontology_functions.py -v -k earliness`
Expected: FAIL — `ImportError: cannot import name 'rank_stockout_earliness'`

- [ ] **Step 3: 구현**

`src/bakery/ontology/functions.py`의 상수 영역(`DEMAND_PROXY_COL` 아래)에 추가:

```python
DEFAULT_CLOSE_HOUR = 22  # 라벨된 가정: bonavi loader 하드코딩·synthetic store_A와 일치 (spec D3)
```

`rank_stockout_risk` 함수 바로 뒤에 추가:

```python
def rank_stockout_earliness(
    daily: pd.DataFrame,
    store_id: str,
    period: tuple[str, str],
    k: int = 5,
    *,
    close_hour: int = DEFAULT_CLOSE_HOUR,
) -> pd.DataFrame:
    """Top-k items by observed stockout earliness: avg selling-hours lost per day.

    Score = mean over ALL the item's days of max(close_hour − stockout
    time-of-day, 0); days without a stockout contribute 0 — earliness and
    frequency in one number. Historical observation (stockout_time), NOT a
    forecast; complements the MC-based rank_stockout_risk.
    """
    sliced = _period_slice(daily, store_id, *period)
    stockout_at = pd.to_datetime(sliced["stockout_time"])
    hour_of_day = stockout_at.dt.hour + stockout_at.dt.minute / 60.0
    lost = (close_hour - hour_of_day).clip(lower=0.0).fillna(0.0)
    per_item = (
        sliced.assign(lost_hours=lost)
        .groupby("item_id", observed=True)
        .agg(lost_hours_total=("lost_hours", "sum"),
             stockout_days=("stockout_time", "count"),
             days=("lost_hours", "size"))
        .reset_index()
    )
    per_item["lost_hours_per_day"] = per_item["lost_hours_total"] / per_item["days"]
    if float(per_item["lost_hours_per_day"].max()) == 0.0:
        raise ValueError(f"no stockouts observed for store={store_id} in {period}")
    ranked = per_item.sort_values(["lost_hours_per_day", "item_id"],
                                  ascending=[False, True]).head(k)
    return ranked[["item_id", "lost_hours_per_day", "stockout_days", "days"]].reset_index(drop=True)
```

FUNCTION_REGISTRY의 `"rank_stockout_risk"` 항목 바로 뒤에 추가:

```python
    "rank_stockout_earliness": OntologyFunctionSpec(
        "rank_stockout_earliness",
        "Top-k items by observed stockout earliness (avg selling-hours lost per day).",
        ("store_id", "period", "k"),
        "table[item_id, lost_hours_per_day, stockout_days, days]", rank_stockout_earliness),
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_ontology_functions.py -v`
Expected: 전부 PASS (기존 테스트 포함 — `test_function_registry_impls_match_module`이 신규 항목 자동 검증)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/functions.py tests/test_ontology_functions.py
git commit -m "feat: rank_stockout_earliness — 관측 매진시각 기반 위험 랭킹 (일평균 손실 영업시간)"
```

---

### Task 2: ToolSpec + dispatch 노출

**Files:**
- Modify: `src/bakery/ontology/grounding/tools.py` (TOOL_SPECS에 spec 추가, `_call`에 분기 추가)
- Test: `tests/test_grounding_tools.py` (기존 name-set 테스트 수정 + dispatch 테스트 추가)

**Interfaces:**
- Consumes: `fn.rank_stockout_earliness(daily, store_id, period, k)` (Task 1)
- Produces: tool 이름 `"rank_stockout_earliness"` (인자 store_id/period/k) — grounded arm이 TOOL_SPECS 경유로 자동 노출받음

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_grounding_tools.py`의 `test_tool_specs_cover_six_functions`를 다음으로 교체:

```python
def test_tool_specs_cover_seven_functions():
    names = {t.name for t in TOOL_SPECS}
    assert names == {
        "rank_stockout_risk", "rank_stockout_earliness", "explain_order",
        "what_if", "waste_cost", "demand_diff_by_condition", "what_if_driver",
    }
    for t in TOOL_SPECS:
        assert t.parameters["type"] == "object"
        assert "properties" in t.parameters
```

파일 끝에 추가:

```python
def test_dispatch_rank_stockout_earliness_returns_json(dataset):
    store = dataset.daily["store_id"].iloc[0]
    dates = pd.to_datetime(dataset.daily.loc[dataset.daily["store_id"] == store, "date"])
    call = ToolCall(id="c9", name="rank_stockout_earliness",
                    arguments={"store_id": store,
                               "period": [str(dates.min().date()), str(dates.max().date())],
                               "k": 3})
    result = dispatch(call, dataset)
    payload = json.loads(result.content)
    assert len(payload) == 3
    assert {"item_id", "lost_hours_per_day", "stockout_days", "days"} <= set(payload[0])
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_grounding_tools.py -v`
Expected: `test_tool_specs_cover_seven_functions` FAIL (name set 불일치), `test_dispatch_rank_stockout_earliness_returns_json` FAIL (error payload — unknown tool)

- [ ] **Step 3: 구현**

`src/bakery/ontology/grounding/tools.py` TOOL_SPECS에서 `rank_stockout_risk` spec 바로 뒤에 추가:

```python
    ToolSpec("rank_stockout_earliness",
             "Top-k items by observed stockout earliness: average selling-hours lost "
             "per day to stockouts (higher = stocks out earlier/more often). "
             "Historical observation from stockout_time, NOT a forecast.",
             {"type": "object", "properties": {
                 "store_id": {"type": "string"}, "period": _PERIOD, "k": {"type": "integer"}},
              "required": ["store_id", "period", "k"], "additionalProperties": False}),
```

`_call`에서 `rank_stockout_risk` 분기 바로 뒤에 추가:

```python
    if name == "rank_stockout_earliness":
        return fn.rank_stockout_earliness(dataset.daily, a["store_id"], tuple(a["period"]), a["k"])
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_grounding_tools.py -v`
Expected: 전부 PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/grounding/tools.py tests/test_grounding_tools.py
git commit -m "feat: rank_stockout_earliness ToolSpec + dispatch — grounded arm에 관측 매진위험 랭킹 노출"
```

---

### Task 3: Q셋 재설계 — q_order_top 체인 gold + q_rank_earliness 추가

**Files:**
- Modify: `src/bakery/ontology/grounding/questions.py` (`_ctx` 축소, q_order_top 교체, q_rank_earliness 추가, build_gold 분기 수정·추가)
- Test: `tests/test_grounding_questions.py`

**Interfaces:**
- Consumes: `fn.rank_stockout_earliness` (Task 1), `fn.explain_order` (기존)
- Produces: q_order_top gold = `{"item_id": str, "order_qty": float}` (Task 4 scorer가 이 형식을 채점), QUESTIONS 11문항 (`q_rank_earliness` 포함)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_grounding_questions.py`의 `test_build_gold_is_deterministic`에서 decomposition 분기를 수정하고, 파일 끝에 체인 일치 테스트 2개 추가:

```python
        elif q.grader_type == "decomposition":
            assert "order_qty" in g1
            assert "item_id" in g1
```

```python
def test_q_order_top_gold_matches_chain(dataset):
    """gold가 rank(k=1)→explain 체인과 정확히 일치 — grounded arm이 같은 체인으로 도달 가능."""
    from bakery.ontology import functions as fn
    from bakery.ontology.grounding.questions import resolve_eval_context

    q = next(q for q in QUESTIONS if q.id == "q_order_top")
    gold = build_gold(q, dataset)
    store, period = resolve_eval_context(dataset)
    top1 = fn.rank_stockout_earliness(dataset.daily, store, period, k=1)
    assert gold["item_id"] == str(top1["item_id"].iloc[0])
    lineage = fn.explain_order(dataset.daily, store, gold["item_id"], period)
    assert gold["order_qty"] == pytest.approx(float(lineage["contribution"].sum()))


def test_q_rank_earliness_gold_top3(dataset):
    q = next(q for q in QUESTIONS if q.id == "q_rank_earliness")
    gold = build_gold(q, dataset)
    assert len(gold["top_items"]) == 3
    assert len(set(gold["top_items"])) == 3
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_grounding_questions.py -v`
Expected: FAIL — q_order_top gold에 `item_id` 없음, `q_rank_earliness` 미존재(StopIteration)

- [ ] **Step 3: 구현**

`src/bakery/ontology/grounding/questions.py` 수정 4곳:

(a) `_ctx`에서 `top_item` 제거 (참조 0 확인됨 — build_gold explain 분기가 유일한 소비자였음):

```python
def _ctx(dataset: DailyDataset):
    """Resolve a stable (store, period) from the dataset."""
    daily = dataset.daily
    store = sorted(dataset.daily["store_id"].unique())[0]
    dd = pd.to_datetime(daily.loc[daily["store_id"] == store, "date"])
    period = (str(dd.min().date()), str(dd.max().date()))
    return store, period
```

(b) QUESTIONS에서 q_order_top 항목을 교체하고, 그 바로 뒤에 q_rank_earliness 추가:

```python
    Question("q_order_top",
             "관측 매진 위험(일평균 손실 영업시간 기준)이 가장 높은 품목은 무엇이고, "
             "그 품목의 권장 발주량은?",
             DECOMPOSITION, "explain_order", {}),
    Question("q_rank_earliness",
             "일평균 손실 영업시간 기준 매진 위험 상위 3개 품목은?",
             RANKING, "rank_stockout_earliness", {"k": 3}),
```

(c) `resolve_eval_context`와 `build_gold` 첫 줄을 2-tuple 언패킹으로 수정:

```python
def resolve_eval_context(dataset: DailyDataset) -> tuple[str, tuple[str, str]]:
    """The (store_id, period) the eval targets — same basis as build_gold's gold."""
    return _ctx(dataset)
```

```python
def build_gold(question: Question, dataset: DailyDataset) -> dict:
    store, period = _ctx(dataset)
```

(d) `build_gold`의 `explain_order` 분기를 교체하고, `rank_stockout_risk` 분기 바로 뒤에 `rank_stockout_earliness` 분기 추가:

```python
    if question.source_fn == "rank_stockout_earliness":
        ranked = fn.rank_stockout_earliness(dataset.daily, store, period, k=k["k"])
        return {"top_items": list(ranked["item_id"])}
```

```python
    if question.source_fn == "explain_order":
        top_risk = fn.rank_stockout_earliness(dataset.daily, store, period, k=1)
        risk_item = str(top_risk["item_id"].iloc[0])
        lineage = fn.explain_order(dataset.daily, store, risk_item, period)
        value = float(lineage["contribution"].sum())
        if not math.isfinite(value):
            raise ValueError(f"non-finite gold for {question.id}: {value}")
        return {"item_id": risk_item, "order_qty": value}
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_grounding_questions.py tests/test_grounding_run.py -v`
Expected: 전부 PASS (run 테스트로 회귀 확인)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/grounding/questions.py tests/test_grounding_questions.py
git commit -m "feat: q_order_top 재설계(관측 매진위험 1위 rank→explain 체인) + q_rank_earliness 추가"
```

---

### Task 4: decomposition 채점 강화 — item_id + order_qty

**Files:**
- Modify: `src/bakery/ontology/grounding/scorer.py` (`grade` 분기, `_grade_qty` 교체)
- Modify: `src/bakery/ontology/grounding/arms.py` (OUTPUT_SCHEMAS[DECOMPOSITION])
- Test: `tests/test_grounding_scorer.py`

**Interfaces:**
- Consumes: gold `{"item_id": str, "order_qty": float}` (Task 3)
- Produces: decomposition 답 스키마 `{"item_id": string, "order_qty": number}` (required 전키 — strict schema 규칙)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_grounding_scorer.py`의 `test_decomposition_qty_match`를 다음 3개로 교체 (기존 테스트의 Question 생성 패턴을 그대로 따름):

```python
def _decomp_q():
    return Question("q", "t", "decomposition", "explain_order", {})


def test_decomposition_both_match():
    gold = {"item_id": "item_C03", "order_qty": 42.0}
    assert grade(_decomp_q(), {"item_id": "item_C03", "order_qty": 42.0}, gold) is True


def test_decomposition_wrong_item_right_qty_fails():
    gold = {"item_id": "item_C03", "order_qty": 42.0}
    assert grade(_decomp_q(), {"item_id": "item_B04", "order_qty": 42.0}, gold) is False


def test_decomposition_right_item_wrong_qty_fails():
    gold = {"item_id": "item_C03", "order_qty": 42.0}
    assert grade(_decomp_q(), {"item_id": "item_C03", "order_qty": 43.0}, gold) is False
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_grounding_scorer.py -v`
Expected: `test_decomposition_wrong_item_right_qty_fails` FAIL (현행 채점은 qty만 보므로 True 반환)

- [ ] **Step 3: 구현**

`src/bakery/ontology/grounding/scorer.py`에서 `_grade_qty`를 교체:

```python
def _grade_decomposition(pred_item, pred_qty, gold: dict) -> bool:
    if not isinstance(pred_item, str) or not isinstance(pred_qty, (int, float)):
        return False
    return pred_item == gold["item_id"] and abs(pred_qty - gold["order_qty"]) <= _QTY_TOL
```

`grade`의 DECOMPOSITION 분기 교체:

```python
    if question.grader_type == DECOMPOSITION:
        return _grade_decomposition(answer.get("item_id"), answer.get("order_qty"), gold)
```

`src/bakery/ontology/grounding/arms.py`의 OUTPUT_SCHEMAS[DECOMPOSITION] 교체:

```python
    DECOMPOSITION: {"type": "object",
                    "properties": {"item_id": {"type": "string"},
                                   "order_qty": {"type": "number"}},
                    "required": ["item_id", "order_qty"], "additionalProperties": False},
```

모듈 docstring(scorer.py 4행 "decomposition uses exact order-qty match")도 "decomposition requires item_id exact match AND order-qty match"로 갱신.

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest tests/test_grounding_scorer.py tests/test_grounding_arms.py -v`
Expected: 전부 PASS

- [ ] **Step 5: Commit**

```bash
git add src/bakery/ontology/grounding/scorer.py src/bakery/ontology/grounding/arms.py tests/test_grounding_scorer.py
git commit -m "feat: decomposition 채점 강화 — item_id+order_qty 둘 다 일치 (품목 오인 차단, 실패 진단 가능)"
```

---

### Task 5: 전체 회귀 + live smoke + 문서 마감

**Files:**
- Modify: `TODO.md` (19행 항목 완료 체크)
- Test: 전체 스위트 + live eval 1회

**Interfaces:**
- Consumes: Task 1~4 전체
- Produces: 머지 가능한 브랜치

- [ ] **Step 1: 전체 테스트**

Run: `uv run pytest`
Expected: 전부 PASS (기준선 279 + 신규 ~10). 실패 시 실패 테스트 단위로 원인 수정 (3회 내 미해결 시 사용자 보고).

- [ ] **Step 2: live smoke (필수 — schema 오류는 CI가 못 잡음)**

`.env`의 Azure 키 필요 (provider=auto). 실행:

```bash
uv run bakery grounding-eval
```

Expected: q_order_top·q_rank_earliness에서 **grounded arm 정답(ok=True)**, grounded arm 로그/트랜스크립트에서 q_order_top이 `rank_stockout_earliness` → `explain_order` 2-step 체인을 실제 수행했는지 확인. delta는 기존 +0.8 수준 이상 유지. 신규 tool schema가 Azure strict 모드에서 400 없이 통과하는지 확인 (required 전키 규칙).

실패 시: tool schema(400) → required/additionalProperties 점검, 체인 미수행 → ToolSpec description 보강 후 재실행.

- [ ] **Step 3: TODO.md 갱신**

19행을 완료로:

```markdown
- [x] q_order_top Q셋 재설계 (rank→explain 체인) — 관측 매진시각 기반 rank_stockout_earliness, 채점 item_id+qty.
```

- [ ] **Step 4: Commit**

```bash
git add TODO.md
git commit -m "docs: TODO — q_order_top Q셋 재설계 완료 체크"
```
