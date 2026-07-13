# 다매장 stockout 재정의 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `scripts/store_daily.build_store_daily`의 품절 산출을 옛 버그 로직(stockout.parquet first-event, is_stockout 92.7%)에서 단매장과 동일한 재정의 공식(`assign_stockout_fields`, 폐기0&완판, ~48%)으로 교체한다. 반환 7컬럼·HTML·발주 불변.

**Architecture:** store_daily가 V2 `inventory.parquet`(QT_MADE/QT_OUT)와 bulk-제외 sales의 SALES_TIME max를 준비해 `bonavi_loader.assign_stockout_fields`를 호출. `aggregate_daily`(무거운 오케스트레이터)는 안 씀. 설계 = docs/superpowers/specs/2026-07-13-multistore-stockout-redefine-design.md.

**Tech Stack:** Python, pandas, pytest.

## Global Constraints

- 공유 단위 = `bonavi_loader.assign_stockout_fields`(재정의 공식). `aggregate_daily`·단매장 경로 **손대지 않음**.
- 반환 컬럼 불변: `[date, item_id, sold_units, store_id, category_id, stockout_time, is_stockout]`.
- bulk 제외는 **sales 계열(sold_units·last_sale_ts)만**, inventory(QT_MADE/QT_OUT)는 그대로(물리수량, 예약 무관).
- last_sale_ts는 store_daily의 line-level `flag_bulk_lines` 제외본 sales에서 계산(구 `absorption_4stores._last_sale_ts`=whole-receipt bulk **안 씀**).
- HTML/발주 무영향: stockout 컬럼은 `category_total.LEAK_COLS`라 학습 feature 아님 — 회귀 가드로 고정.
- 테스트 단언 정확값. `uv run pytest`(추가 `-q` 금지).
- SALES_TIME 포맷 = `%Y%m%d%H%M%S` (예: `20210101100520`).

---

### Task 1: store_daily 재정의 교체

**Files:**
- Modify: `scripts/store_daily.py` (`build_store_daily`의 stockout 산출부 교체)
- Test: `tests/test_store_daily_redefine.py` (신규)

**Interfaces:**
- Consumes: `bakery.data.bonavi_loader.assign_stockout_fields(df)` — `df`에 `production_qty·waste_qty·last_sale_ts` 필요, `is_stockout·stockout_time` 반환.
- Produces: `build_store_daily(store_cd, store_id, exclude_bulk=True)` — 반환 컬럼 동일, is_stockout=(QT_MADE>0 & QT_OUT<=0).

- [ ] **Step 1: Write failing test (RED)**

`tests/test_store_daily_redefine.py`. 실데이터 의존이라 `build_store_daily` 통합 스모크 + `assign_stockout_fields` 공식 단위테스트 조합. 공식 단위테스트는 합성으로 정확값:

```python
import numpy as np
import pandas as pd
import pytest

from bakery.data.bonavi_loader import assign_stockout_fields


def test_assign_stockout_fields_redefinition_exact():
    # 재정의: is_stockout = (production_qty>0 & waste_qty<=0), stockout_time=last_sale_ts(완판 시)
    df = pd.DataFrame({
        "production_qty": [10, 10, 0, 10, 10],
        "waste_qty":      [0, 3, 0, -2, np.nan],   # 완판 / 잔여 / 미생산 / 음수(반품, 완판) / 결측
        "last_sale_ts": pd.to_datetime([
            "2024-01-01 20:00", "2024-01-01 21:00", "2024-01-01 19:00",
            "2024-01-01 18:30", "2024-01-01 17:00"]),
    })
    out = assign_stockout_fields(df)
    assert list(out["is_stockout"]) == [True, False, False, True, False]
    # 완판 행만 stockout_time 채워짐
    assert out["stockout_time"].iloc[0] == pd.Timestamp("2024-01-01 20:00")
    assert pd.isna(out["stockout_time"].iloc[1])
    assert pd.isna(out["stockout_time"].iloc[2])
    assert out["stockout_time"].iloc[3] == pd.Timestamp("2024-01-01 18:30")
    assert pd.isna(out["stockout_time"].iloc[4])
```

(이 테스트는 기존 단매장 코드에 대해 이미 PASS — 공식 회귀 가드. RED는 Step 2의 store_daily 통합에서 확인.)

- [ ] **Step 2: store_daily 통합 스모크 테스트 추가 (RED)**

같은 파일에 추가 — 현재 store_daily는 옛 92% 로직이라 이 단언에서 FAIL:

```python
def test_build_store_daily_uses_redefinition():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from store_daily import build_store_daily
    d = build_store_daily("1000000047", "store_gw01", exclude_bulk=True)
    # 재정의 후 광교 item-day is_stockout 비율은 옛 92%가 아니라 ~60%대여야 함
    rate = d["is_stockout"].mean()
    assert 0.50 < rate < 0.70, f"expected redefined ~0.60, got {rate:.3f}"
    # 반환 컬럼 스키마 불변
    assert set(["date","item_id","sold_units","store_id","category_id","stockout_time","is_stockout"]).issubset(d.columns)
    # 완판(is_stockout) 행은 stockout_time 있고, 아닌 행은 NaT
    so = d[d["is_stockout"]]; nso = d[~d["is_stockout"]]
    assert so["stockout_time"].notna().all()
    assert nso["stockout_time"].isna().all()
```

- [ ] **Step 3: RED 확인**

Run: `uv run pytest tests/test_store_daily_redefine.py::test_build_store_daily_uses_redefinition -v`
Expected: FAIL — 현재 rate≈0.927 (옛 notna 로직).

- [ ] **Step 4: store_daily 교체**

`scripts/store_daily.py`의 stockout 산출부(현재 `so = read stockout.parquet ... so_first = groupby first ... daily["is_stockout"]=notna()`)를 교체. import에 `from bakery.data.bonavi_loader import assign_stockout_fields` 추가. sold_units 집계 이후:

```python
    # === 재정의 stockout: 폐기0 & 완판 (단매장 assign_stockout_fields 공식 공유) ===
    # ② 생산/폐기 (bulk 무관 물리수량) → 재정의 입력 컬럼명으로
    inv = pd.read_parquet(V2 / "inventory.parquet")
    inv = inv[inv["CD_PARTNER"].astype(str) == store_cd].copy()
    inv["date"] = pd.to_datetime(inv["DT_SALE"].astype(str))
    inv["item_id"] = inv["CD_ITEM"].astype(str)
    inv = inv.rename(columns={"QT_MADE": "production_qty", "QT_OUT": "waste_qty"})
    inv = inv[["date", "item_id", "production_qty", "waste_qty"]]

    # ③ 마지막 실판매 시각 — bulk 제외본 sales에서 (sold_units와 동일 필터)
    ls = sales.copy()
    ls["date"] = pd.to_datetime(ls["DT_SALE"].astype(str))
    ls["item_id"] = ls["CD_ITEM"].astype(str)
    ls["last_sale_ts"] = pd.to_datetime(ls["SALES_TIME"].astype(str),
                                        format="%Y%m%d%H%M%S", errors="coerce")
    ls = ls.groupby(["date", "item_id"], as_index=False)["last_sale_ts"].max()

    # ④ merge → 재정의 공식
    daily = daily.merge(inv, on=["date", "item_id"], how="left")
    daily = daily.merge(ls, on=["date", "item_id"], how="left")
    daily = assign_stockout_fields(daily)
    daily = daily.drop(columns=["production_qty", "waste_qty", "last_sale_ts"])
    return daily
```

기존 `scripts/store_daily.py:61-78`의 `so = pd.read_parquet(V2/"stockout.parquet")` ~ `daily["is_stockout"]=daily["stockout_time"].notna(); return daily` 블록 전체를 위로 대체.

**확인됨(구현자 참고)**: `sales`는 bulk 제외가 적용된 변수이고 daily 집계 이후에도 살아있으며 `SALES_TIME` 원본 컬럼을 보존한다 → `ls = sales.copy()`가 그대로 동작(daily 생성 전으로 옮길 필요 없음). `stockout.parquet` 읽기는 삭제(더는 안 씀).

- [ ] **Step 5: GREEN + 기존 테스트**

Run: `uv run pytest tests/test_store_daily_redefine.py tests/test_ontology_functions.py -v`
Expected: 전부 PASS (신규 2 + ontology). is_stockout rate ~0.60.

- [ ] **Step 6: Commit**

```bash
git add scripts/store_daily.py tests/test_store_daily_redefine.py
git commit -m "fix(store_daily): 다매장 stockout 재정의 (폐기0&완판, assign_stockout_fields 공유)"
```

---

### Task 2: HTML/발주 무영향 확인 + 소비처 영향맵

**Files:**
- Test: `tests/test_store_daily_redefine.py` (무영향 가드 추가)
- Create: `docs/multistore_stockout_redefine_impactmap.md` (소비처 분류)

**Interfaces:**
- Consumes: `bakery.models.category_total.select_feature_cols`, `LEAK_COLS`.

- [ ] **Step 1: HTML 무영향 회귀 가드 테스트**

stockout 컬럼이 학습 feature에 안 들어감을 고정(store_daily 재정의가 예측에 영향 없다는 근거):

```python
def test_stockout_cols_excluded_from_training_features():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from store_daily import build_store_daily, build_store_closing_rows
    from bakery.features.category_aggregate import build_category_daily, build_features
    from bakery.models.category_total import select_feature_cols
    daily = build_store_daily("1000000047", "store_gw01", exclude_bulk=True)
    cd = build_category_daily(daily_raw=daily,
                              discount_rows=build_store_closing_rows("1000000047"), alpha=0.8)
    feat = build_features(cd, target_col="adjusted_demand_unit")
    cols = select_feature_cols(feat, "adjusted_demand_unit")
    leaked = [c for c in cols if "stockout" in c.lower()]
    assert leaked == [], f"stockout cols leaked into features: {leaked}"
```

- [ ] **Step 2: 실행 확인**

Run: `uv run pytest tests/test_store_daily_redefine.py::test_stockout_cols_excluded_from_training_features -v`
Expected: PASS (stockout 컬럼 0개 — 예측 무영향 확증).

- [ ] **Step 3: 소비처 영향맵 문서 작성**

`docs/multistore_stockout_redefine_impactmap.md`. store_daily의 `is_stockout`/`stockout_time` 소비처를 grep으로 열거하고 각각 "재정의가 결과를 바꾸나" 분류. 실제 확인:

Run: `cd /Users/taehoonkim/dev/bakery-predictor && grep -rln "is_stockout\|stockout_time" scripts/ src/ | grep -v test`

각 소비처를 (A) 품절 플래그를 분석 입력으로 실제 사용(재검증 후보) / (B) LEAK_COLS·미사용(불변)으로 분류. 표로 정리, "재검증은 사용자 선택" 명시. (예상 A: absorption_4stores·substitution_4stores·revalidate_popularity_stockout·stockout_classifier / B: store_predictive_power·verify_event_prior·발주.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_store_daily_redefine.py docs/multistore_stockout_redefine_impactmap.md
git commit -m "test(store_daily): HTML 무영향 가드 + 소비처 영향맵 문서"
```

---

### Task 3: 전체 회귀 + 재정의 정량 확인

**Files:** (검증만, 코드 변경 없음)

- [ ] **Step 1: 전체 테스트**

Run: `uv run pytest`
Expected: 신규 테스트 포함 전체 PASS (1 pre-existing 네트워크 `test_grounding_openai_adapter` 무관), 회귀 0.

- [ ] **Step 2: 4매장 재정의 정량 확인**

Run: `PYTHONPATH=scripts uv run python -c "
import warnings; warnings.filterwarnings('ignore')
from store_daily import build_store_daily
for cd,sid,nm in [('1000000047','store_gw01','광교'),('1000000009','store_ss01','삼성'),('1000000029','store_mp01','메세나'),('1000000485','store_gh01','광화문')]:
    d=build_store_daily(cd,sid); print(f'{nm}: is_stockout {d[\"is_stockout\"].mean()*100:.1f}% (n={len(d)})')
"`
Expected: 4매장 모두 옛 ~92%가 아닌 재정의 비율(매장별 상이, 광교 ~60%). 광화문은 inventory 2022-07~라 그 이전 False.

- [ ] **Step 3: (선택) HTML 재생성 후 diff 무변화 스팟체크**

시간 허용 시 `PYTHONPATH=scripts uv run --with matplotlib python scripts/store_predictive_power.py` → 광교 headline WAPE가 재생성 전과 동일한지 확인(무영향 실증). 리포트 로그 WAPE만 대조.

---

## Self-Review

**Spec coverage:** §3.1 assign_stockout_fields 공유 → Task 1 ✓. §3.2 재작성 → Task 1 ✓. §4 무영향 → Task 2 ✓. §5 영향맵 → Task 2 ✓. §6 테스트 → Task 1·2 ✓.
**Placeholder scan:** 없음. 모든 코드/명령/기대값 명시.
**Type consistency:** `assign_stockout_fields(df)->df` (production_qty·waste_qty·last_sale_ts in, is_stockout·stockout_time out). `build_store_daily->7col df`. Task 1·2 일관.
**해결됨:** `sales`는 daily 집계(store_daily.py:52) 이후에도 살아있고 `SALES_TIME` 원본 보존 확인 → `ls=sales.copy()` 그대로 동작. 교체 대상 = store_daily.py:61-78(stockout.parquet 블록 전체).
