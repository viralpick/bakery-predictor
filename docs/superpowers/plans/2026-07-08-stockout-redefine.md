# is_stockout/stockout_time 재정의 (데이터 fix) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `bonavi_loader`에서 is_stockout/stockout_time을 "물리적 leftover(폐기) 기반 진짜 최종소진"으로 재정의하고 bonavi_daily.parquet 재생성. 버그난 "첫 순간품절 이벤트" 로직 대체.

**Architecture:** 순수 헬퍼 `assign_stockout_fields`(made/waste/last_sale → is_stockout/stockout_time)를 만들어 단위 테스트, `aggregate_daily`가 이를 사용하도록 배선(stockouts 대신 inventory+last_sale 주입), `build`가 재고정보 로드 + 영수증 마지막판매시각 전달. `품절정보` 시트 의존 제거. 그 후 bonavi_daily 재생성 + 검증.

**Tech Stack:** Python, pandas. 신규 의존성 없음. 기존 `load_inventory`(ingest/inventory.py) 재사용.

## Global Constraints

- **Time leakage 금지**: 신규 정의는 당일 관측치(폐기·당일 마지막판매)라 미래 누수 없음. `test_split_leakage`/`test_features_leakage` + `assert_no_leakage` 그린 유지가 완료 조건.
- **정의 (실 로더 경로만)**: `is_stockout = (production_qty > 0) AND (waste_qty <= 0)`; 재고정보 결측(NaN) → 자동 False(`NaN>0`, `NaN<=0` 둘 다 False). `stockout_time = 그날 마지막 실판매 timestamp` (is_stockout일 때, 아니면 NaT).
- **실측 목표치**: is_stockout 92% → **~60.5%** (폐기==0 58.1% + <0 3.2%). 결측 0.1%(65건, default False).
- **synthetic.py 경로 불변** (합성 is_stockout는 PoC 조작값).
- **potential_demand 컬럼 유지** — 교정 입력으로 자동 재계산. 제거/‌v2·v3 변경 없음.
- **schema 불변**: made/waste는 is_stockout 계산에만 transient 사용, bonavi_daily에 컬럼 추가 안 함.
- **테스트 단언 정확값 `==`**; 함수 30줄 이내.

---

## File Structure

- `src/bakery/data/bonavi_loader.py` — `assign_stockout_fields`(신규 순수 헬퍼) + `aggregate_daily`(시그니처: stockouts→inventory/last_sale) + `build`(inventory 로드 + last_sale 계산 + stockouts 호출 제거).
- `tests/test_bonavi_loader_stockout.py` — 신규 (헬퍼 truth table).
- `data/internal/bonavi_daily.parquet` — 재생성 (Task 3).
- `docs/stockout_redefine_result.md` — 검증 결과 (Task 3).

---

### Task 1: 순수 헬퍼 `assign_stockout_fields`

**Files:**
- Modify: `src/bakery/data/bonavi_loader.py` (add `assign_stockout_fields`)
- Test: `tests/test_bonavi_loader_stockout.py`

**Interfaces:**
- Produces: `assign_stockout_fields(df: pd.DataFrame) -> pd.DataFrame` — `df`에 `production_qty`, `waste_qty`, `last_sale_ts`(datetime64, NaT 허용) 컬럼 필요. 반환: 원본 + `is_stockout`(bool), `stockout_time`(datetime64, NaT where not stockout). 규칙: `is_stockout = (production_qty>0) & (waste_qty<=0)` (NaN → False); `stockout_time = last_sale_ts where is_stockout else NaT`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bonavi_loader_stockout.py
import numpy as np
import pandas as pd
from bakery.data.bonavi_loader import assign_stockout_fields


def test_assign_stockout_fields_truth_table():
    df = pd.DataFrame({
        "production_qty": [10.0, 5.0, 8.0, 3.0, np.nan],
        "waste_qty":      [0.0,  2.0, -1.0, 4.0, np.nan],
        "last_sale_ts": pd.to_datetime([
            "2024-01-01 21:30", "2024-01-01 20:00", "2024-01-01 15:00",
            "2024-01-01 19:00", "2024-01-01 12:00",
        ]),
    })
    out = assign_stockout_fields(df)
    # made>0 & waste<=0 → True: row0(10,0), row2(8,-1). row1(waste2>0)=F, row3(waste4)=F, row4(nan)=F
    assert out["is_stockout"].tolist() == [True, False, True, False, False]
    assert out.loc[0, "stockout_time"] == pd.Timestamp("2024-01-01 21:30")
    assert pd.isna(out.loc[1, "stockout_time"])
    assert out.loc[2, "stockout_time"] == pd.Timestamp("2024-01-01 15:00")
    assert pd.isna(out.loc[4, "stockout_time"])


def test_assign_stockout_fields_missing_inventory_is_false():
    # 재고정보 결측(NaN made/waste) → is_stockout False, stockout_time NaT (판매 있어도)
    df = pd.DataFrame({
        "production_qty": [np.nan],
        "waste_qty": [np.nan],
        "last_sale_ts": pd.to_datetime(["2024-01-01 21:00"]),
    })
    out = assign_stockout_fields(df)
    assert out["is_stockout"].tolist() == [False]
    assert pd.isna(out.loc[0, "stockout_time"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bonavi_loader_stockout.py -v`
Expected: FAIL with `ImportError: cannot import name 'assign_stockout_fields'`

- [ ] **Step 3: Write minimal implementation**

`src/bakery/data/bonavi_loader.py`에 모듈 레벨 함수 추가(예: `aggregate_daily` 위):
```python
def assign_stockout_fields(df: pd.DataFrame) -> pd.DataFrame:
    """물리 leftover 기반 진짜 최종소진. is_stockout=(made>0 & waste<=0),
    stockout_time=마지막 실판매(is_stockout일 때). 재고정보 결측(NaN)→False (NaN 비교가
    False라 자동). 첫 순간품절 이벤트를 쓰던 버그를 대체한다."""
    made = pd.to_numeric(df["production_qty"], errors="coerce")
    waste = pd.to_numeric(df["waste_qty"], errors="coerce")
    out = df.copy()
    out["is_stockout"] = ((made > 0) & (waste <= 0)).fillna(False).astype(bool)
    out["stockout_time"] = df["last_sale_ts"].where(out["is_stockout"])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bonavi_loader_stockout.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/bakery/data/bonavi_loader.py tests/test_bonavi_loader_stockout.py
git commit -m "feat: assign_stockout_fields — leftover 기반 진짜 최종소진 헬퍼"
```

---

### Task 2: `aggregate_daily` + `build` 배선 (inventory/last_sale 주입, stockouts 제거)

**Files:**
- Modify: `src/bakery/data/bonavi_loader.py` (`aggregate_daily`, `build`)

**Interfaces:**
- Consumes: `assign_stockout_fields` (Task 1); `load_inventory(xlsx_path, store_id) -> [date,item_id,production_qty,waste_qty]` (ingest/inventory.py); `load_receipts_with_time(...) -> [receipt_id,date,item_id,hour,minute,timestamp]`.
- Produces: `aggregate_daily(sales, items, inventory, last_sale, *, measured_profiles=None) -> pd.DataFrame` — 시그니처에서 `stockouts` 제거, `inventory`(date/item_id/production_qty/waste_qty) + `last_sale`(date/item_id/last_sale_ts) 추가.

- [ ] **Step 1: `aggregate_daily` 시그니처·본문 교체**

기존 `aggregate_daily`의 stockout 블록:
```python
    # First stockout per (store, item, date)
    first_so = (
        stockouts.sort_values(["store_id", "item_id", "date", "stockout_time"])
        .groupby(["store_id", "item_id", "date"], as_index=False)
        .first()
    )
    daily = daily.merge(
        first_so[["store_id", "item_id", "date", "stockout_time"]],
        on=["store_id", "item_id", "date"], how="left",
    )
    daily["is_stockout"] = daily["stockout_time"].notna()
```
를 아래로 교체 (시그니처의 `stockouts` → `inventory`, `last_sale` 추가):
```python
    # is_stockout/stockout_time: 물리 leftover(폐기) 기반 진짜 최종소진 (첫 순간품절 이벤트 버그 대체)
    inv = inventory.copy()
    inv["item_id"] = inv["item_id"].astype(str)
    inv["date"] = pd.to_datetime(inv["date"]).dt.normalize()
    ls = last_sale.copy()
    ls["item_id"] = ls["item_id"].astype(str)
    ls["date"] = pd.to_datetime(ls["date"]).dt.normalize()
    daily["item_id"] = daily["item_id"].astype(str)
    daily = daily.merge(inv[["date", "item_id", "production_qty", "waste_qty"]],
                        on=["date", "item_id"], how="left")
    daily = daily.merge(ls[["date", "item_id", "last_sale_ts"]],
                        on=["date", "item_id"], how="left")
    daily = assign_stockout_fields(daily)
    daily = daily.drop(columns=["production_qty", "waste_qty", "last_sale_ts"])
```
시그니처: `def aggregate_daily(sales, items, inventory, last_sale, *, measured_profiles=None):` (기존 `stockouts` 파라미터 제거). docstring의 "is_stockout / stockout_time: 품절정보의 가장 이른 시각" 줄을 "물리 leftover 기반 진짜 최종소진"으로 갱신.

- [ ] **Step 2: `build`에서 inventory 로드 + last_sale 계산 + stockouts 호출 제거**

`build` 내부에서 `stockouts = load_stockouts(...)` 줄을 제거하고, `aggregate_daily` 호출 전에 아래 추가:
```python
    from ..ingest.inventory import load_inventory
    inv_store = rename_store_id or "store_gw01"
    # ⚠️ 재고정보(재고정보 시트)는 xlsx_path(0520 기본)에 없고 0526 파일에만 있다.
    # build는 items/sales/receipts를 xlsx_path(0520)에서 읽고, inventory는 별도 0526에서 읽는다
    # (cli.py의 bonavi_daily(0520)⋈inventory(REAL_INVENTORY_XLSX_PATH=0526) 패턴과 동일).
    inventory = load_inventory(str(inventory_xlsx_path), inv_store)
    # 마지막 실판매 시각 (item-day별 max) — receipts_df에서
    last_sale = (
        receipts_df.groupby(["date", "item_id"], as_index=False)["timestamp"].max()
        .rename(columns={"timestamp": "last_sale_ts"})
    )
```

**`build` 시그니처에 신규 파라미터 추가** — 상수 `INVENTORY_XLSX_DEFAULT = Path("data/internal/보나비 데이터_20260526.xlsx")`를 모듈 상단(XLSX_DEFAULT 근처)에 정의하고, `build(...)`에 `inventory_xlsx_path: Path | str = INVENTORY_XLSX_DEFAULT` 파라미터 추가. `cmd_format_bonavi`(cli.py)는 기본값 사용(인자 추가 불필요).
그리고 `aggregate_daily(sales, items, stockouts, measured_profiles=...)` 호출을
`aggregate_daily(sales, items, inventory, last_sale, measured_profiles=...)`로 교체.
`load_stockouts` 함수 정의는 남겨둠(호출만 제거).

주의: `receipts_df`는 `build`에서 이미 `load_receipts_with_time`로 만들어 위에서 사용 중(컬럼 date/item_id/timestamp 보유). `inventory`의 date는 load_inventory 계약상 YYYYMMDD 문자열 → aggregate_daily 내부에서 `pd.to_datetime(...).normalize()`로 정규화(Step 1 코드에 포함됨).

- [ ] **Step 3: import 확인 + 기존 테스트 회귀**

Run: `uv run python -c "import bakery.data.bonavi_loader"` (exit 0)
그리고: `uv run pytest tests/ -k "leakage or loader or v2_pipeline or v3_pipeline or potential_demand"` — 합성 fixture 테스트라 로더 변경에 무영향(그린 유지). aggregate_daily/build를 직접 호출하는 테스트가 있으면 새 시그니처로 갱신(착수 전 `grep -rn "aggregate_daily\|load_stockouts\|build(" tests/`로 확인).
Expected: PASS.

- [ ] **Step 4: 전체 스위트**

Run: `uv run pytest` (repo root; `-q` 추가 금지 — pyproject addopts에 이미 -q라 `-qq`되어 요약 사라짐. 필요시 `--color=no`)
Expected: 모든 테스트 PASS (합성 경로 무영향). 실패 시 aggregate_daily 시그니처 소비처 확인.

- [ ] **Step 5: Commit**

```bash
git add src/bakery/data/bonavi_loader.py
git commit -m "feat: aggregate_daily/build — inventory+last_sale로 is_stockout 재정의, 품절정보 호출 제거"
```

---

### Task 3: bonavi_daily 재생성 + 검증

**Files:**
- Modify: `data/internal/bonavi_daily.parquet` (재생성)
- Create: `docs/stockout_redefine_result.md`

- [ ] **Step 1: 재생성 (138MB xlsx 읽음 — 수 분, 완료까지 대기)**

Run:
```bash
uv run bakery format-bonavi
```
(진입점 `cli.py:1108 cmd_format_bonavi` → `build(xlsx_path, store_code, rename_store_id="store_gw01", out_path="data/internal/bonavi_daily.parquet")`. 인자 기본값은 cmd 정의 확인 후 그대로. bonavi_receipts.parquet도 재기록됨.)
Expected: 에러 없이 완료. 실패하면 STOP + 트레이스백 보고(코드 재수정은 Task 1/2로 회귀, 여기서 임의 패치 금지).

- [ ] **Step 2: 검증 스크립트 (무언 축소 금지)**

Run:
```bash
uv run python -c "
import pandas as pd
d=pd.read_parquet('data/internal/bonavi_daily.parquet')
d['item_id']=d['item_id'].astype(str)
print('is_stockout 비율:', round(100*d['is_stockout'].mean(),1),'% (목표 ~60%, 기존 92%)')
# 09:05 아티팩트 소멸: is_stockout일의 stockout_time hour 분포
so=d[d['is_stockout']].copy(); so['h']=pd.to_datetime(so['stockout_time']).dt.hour
print('stockout_time hour 분위수:', so['h'].quantile([.25,.5,.75]).round(1).to_dict(),'(마지막판매라 저녁 쏠림 기대)')
print('07~09시 이른 stockout_time 비율:', round(100*(so['h']<=9).mean(),1),'% (기존 아티팩트면 높음, 신규면 낮아야)')
print('potential_demand==sold_units 비율:', round(100*(d['potential_demand'].round(3)==d['sold_units']).mean(),1),'% (is_stockout False ~40% 기대)')
print('potential_demand >= sold_units 항상?', bool((d['potential_demand']>=d['sold_units']-1e-6).all()))
"
```
확인: is_stockout ~60%, stockout_time 저녁 쏠림(07~09시 비율 급감 = 09:05 아티팩트 소멸), potential_demand가 is_stockout False에서 sold와 동일. 수치가 목표(±)와 크게 다르면 STOP + 보고.

- [ ] **Step 3: `docs/stockout_redefine_result.md` 작성**

실제 관측 수치로 채움(형식):
```markdown
# is_stockout/stockout_time 재정의 결과 (하위1)

재생성: `uv run bakery format-bonavi`. bonavi_loader가 폐기(재고정보)+마지막판매 기반으로
is_stockout/stockout_time 산출(첫 순간품절 이벤트 버그 대체).

- is_stockout: 92% → …% (made>0 & waste≤0)
- stockout_time: 09:05 등 이른 아티팩트 소멸, 마지막 실판매와 정합 (hour 분위수 …)
- potential_demand: is_stockout False …%는 == sold_units; 나머지는 조기매진분 복원
- 재고정보 결측 …건(0.1%) → is_stockout False

## caveat
- 흡수(W0) 하 item-level 복원은 수요 target 부적합(double-count) — 수요 target=adjusted_demand.
- 광교 단독. 재검증(하위2)에서 W0/timing KPI 등 이 데이터로 재실행 예정.
```

- [ ] **Step 4: Commit**

`data/internal/`가 gitignore인지 확인: `git check-ignore data/internal/bonavi_daily.parquet`. ignored면 parquet은 커밋 안 함(로컬 재생성 산출물), 문서만 커밋.
```bash
git add docs/stockout_redefine_result.md
git commit -m "docs: is_stockout 재정의 재생성 결과 (하위1 검증)"
```

---

## Self-Review

**Spec coverage:**
- is_stockout=(made>0 & waste≤0), 결측→False → Task 1 헬퍼(NaN 자동 False) ✓
- stockout_time=마지막 실판매 → Task 1(last_sale_ts) + Task 2(receipts groupby max) ✓
- 품절정보 의존 제거 → Task 2(load_stockouts 호출 제거) ✓
- 로더 재고정보 신규 의존 → Task 2(load_inventory) ✓
- 재생성 + 검증(is_stockout ~60%, 아티팩트 소멸, potential_demand 정합) → Task 3 ✓
- potential_demand 컬럼 유지·재계산(attach_potential_demand 기존 호출 유지) → 변경 없음(자동) ✓
- 비범위(재검증/폐기/synthetic/schema) → 계획에 없음 ✓

**Placeholder scan:** Task 3 문서·검증은 실행 수치 의존(불가피). code step은 완전 코드. ✓

**Type consistency:** `assign_stockout_fields`가 요구하는 `production_qty`/`waste_qty`/`last_sale_ts`를 Task 2가 merge로 공급(inventory→production_qty/waste_qty, last_sale→last_sale_ts) ✓. `aggregate_daily` 신 시그니처(stockouts 제거, inventory+last_sale)를 build 호출부가 일치 ✓. `load_inventory` 반환 컬럼(date/item_id/production_qty/waste_qty)과 merge 키 일치 ✓.
