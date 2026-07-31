"""배수 제약 하 총량 보존 배분 — 품목별 생산량이 카테고리 총량과 일치하게 만든다.

## 왜 필요한가

현행 `decision/policy._round_up_to_unit` 은 품목마다 `ceil(3의 배수)` 로 **개별 올림**한다.
품목이 수십 가지면 올림 오차가 계통적으로 누적돼 **당일 생산 총량이 뻥튀기된다.**
카테고리 총량을 맞추려고 설계한 스택인데 마지막 단계에서 총량이 깨지는 구조였다.
(인수인계 D3 항목 — "실제 결함"으로 판정됨)

## 배수 규칙 (architect 확정)

**배수 기준 반올림**이고 **0은 금지**한다. k=3이면:

| 예측량 | 결과 |
|---|---|
| 0 ~ 4.5 | **3** (0 금지 → 최소 1배수) |
| 4.5 ~ 7.5 | **6** |
| 7.5 ~ 10.5 | **9** |

경계는 `(n+0.5)×k` 이고 그 지점은 **위로** 간다(round-half-up). 즉 4.5 → 6, 7.5 → 9.

## ★총량 보존이 항상 가능한가 — 아니다

배수 제약과 총량 일치는 수학적으로 양립하지 않을 수 있다. 전 품목이 k=3이면 배분 합은
반드시 3의 배수라 총량 100을 맞출 수 없다.

**우리 경우는 가능하다** — 배수 제약 품목이 소수(광교 실측 21품목: 마스터 4 + 추정 17)이고 나머지는 k=1이라,
k=1 품목이 1개 단위로 잔차를 흡수한다. 그래서 알고리즘은:

1. **제약 품목**(k>1): 배수 반올림 + 최소 k
2. **잔차** = 총량 − 제약 품목 합
3. **비제약 품목**(k=1): largest-remainder로 잔차를 **정확히** 흡수

총량이 최소 필요량보다 작으면 불가능하므로 **fails-loud** 한다 — 조용히 총량을 깨는 것보다
낫다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# 배수 제약이 없는 품목의 k. 1개 단위로 조정 가능해 잔차를 흡수한다.
NO_BATCH_CONSTRAINT = 1
# 아띠제 배수 마스터의 규칙 표기 → 배수
BATCH_RULE_TO_UNIT: dict[str, int] = {
    "짝수": 2,
    "3의배수": 3,
    "8의배수": 8,
}


def round_to_batch(qty: float, unit: int) -> int:
    """배수 기준 반올림. 0은 금지 — 최소 1배수를 낸다.

    unit<=1이면 일반 반올림(단 0 금지). 경계 `(n+0.5)*unit` 은 위로 간다.

    ⚠️`round()` 는 파이썬 banker's rounding(2.5→2)이라 쓰면 안 된다.
    `floor(x + 0.5)` 로 round-half-up을 명시한다 — 4.5/3=1.5 가 2가 되어야 6이 나온다.
    """
    if qty < 0:
        raise ValueError(f"qty must be >= 0, got {qty}")
    if unit <= NO_BATCH_CONSTRAINT:
        return max(1, int(math.floor(float(qty) + 0.5)))
    batches = int(math.floor(float(qty) / unit + 0.5))
    return max(1, batches) * unit


def resolve_batch_units(
    item_ids: pd.Series | list,
    unit_map: dict[str, int] | None = None,
) -> pd.Series:
    """품목별 배수. 매핑에 없으면 제약 없음(k=1)."""
    ids = pd.Series(list(item_ids), dtype="object").astype(str)
    table = unit_map or {}
    return ids.map(lambda i: int(table.get(i, NO_BATCH_CONSTRAINT))).astype(int)


def _absorb_residual(free: pd.DataFrame, residual: int) -> np.ndarray:
    """비제약 품목에 잔차를 배분 — largest remainder. 최소 1개는 지킨다.

    residual>0: 소수부 큰 순으로 +1 / residual<0: 소수부 작은 순으로 −1 (단 1 미만 금지)
    """
    # copy 필수 — to_numpy()가 read-only view를 줄 수 있다(in-place 수정 불가)
    qty = np.array(free["base"].to_numpy(dtype=int), copy=True)
    if residual == 0:
        return qty
    order = np.argsort(-free["frac"].to_numpy()) if residual > 0 \
        else np.argsort(free["frac"].to_numpy())
    step = 1 if residual > 0 else -1
    remaining = abs(residual)
    # 여러 바퀴 돌 수 있다(잔차가 품목 수보다 클 때)
    while remaining > 0:
        moved = False
        for idx in order:
            if remaining == 0:
                break
            if step < 0 and qty[idx] <= 1:      # 0 금지
                continue
            qty[idx] += step
            remaining -= 1
            moved = True
        if not moved:                            # 더 줄일 수 없다
            raise ValueError(
                f"잔차 {residual}를 흡수할 수 없다 — 비제약 품목이 모두 최소치(1)에 도달했다. "
                "총량이 최소 필요량보다 작다."
            )
    return qty


def distribute_with_batch(
    quantities: pd.Series,
    total: float,
    *,
    unit_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """품목별 목표량을 배수 제약 하에 정수화하고 **합을 total과 정확히 일치**시킨다.

    quantities: index=item_id, value=배분 목표량(실수). 합이 total과 달라도 된다.
    반환: [item_id, target, unit, qty] — `qty.sum() == round(total)` 이 보장된다.

    ★제약 품목은 배수를 지키고, 비제약 품목(k=1)이 잔차를 흡수한다. 흡수 불가면 fails-loud.
    """
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    target_total = int(math.floor(float(total) + 0.5))
    frame = pd.DataFrame({
        "item_id": [str(i) for i in quantities.index],
        "target": quantities.to_numpy(dtype=float),
    })
    frame["unit"] = resolve_batch_units(frame["item_id"], unit_map).to_numpy()

    constrained = frame[frame["unit"] > NO_BATCH_CONSTRAINT].copy()
    free = frame[frame["unit"] <= NO_BATCH_CONSTRAINT].copy()
    constrained["qty"] = [
        round_to_batch(q, u) for q, u in zip(constrained["target"], constrained["unit"], strict=True)
    ]

    if free.empty:
        # 비제약 품목이 없으면 잔차를 흡수할 수단이 없다 — 총량 불일치를 숨기지 않는다.
        got = int(constrained["qty"].sum())
        if got != target_total:
            raise ValueError(
                f"총량 보존 불가: 배수 제약 품목만 있어 합 {got} != 총량 {target_total}. "
                "비제약 품목(k=1)이 하나라도 있어야 잔차를 흡수할 수 있다."
            )
        return constrained[["item_id", "target", "unit", "qty"]].reset_index(drop=True)

    free["base"] = np.maximum(1, np.floor(free["target"].to_numpy())).astype(int)
    free["frac"] = free["target"].to_numpy() - np.floor(free["target"].to_numpy())
    residual = target_total - int(constrained["qty"].sum()) - int(free["base"].sum())
    free["qty"] = _absorb_residual(free, residual)

    out = pd.concat([constrained, free], ignore_index=True)
    out = out.set_index("item_id").loc[[str(i) for i in quantities.index]].reset_index()
    got = int(out["qty"].sum())
    if got != target_total:                      # 방어: 알고리즘 버그를 조용히 넘기지 않는다
        raise AssertionError(f"총량 보존 실패: {got} != {target_total}")
    return out[["item_id", "target", "unit", "qty"]]


def load_batch_units(xlsx_path: str, name_to_item: dict[str, str]) -> dict[str, int]:
    """아띠제 배수 마스터(`브레드 맞춤수량`) → {item_id: unit}.

    name_to_item: 정규화된 품목명 → 품목코드. 매칭 실패는 조용히 버리지 않고 반환값에서
    빠지므로, 호출부가 커버리지를 확인해야 한다(마스터 자체가 4품목뿐이다).

    ⚠️마스터는 **누락**이 있다 — 광교 실측에서 마스터 4품목 외에 **17품목**이 정렬도 0.80+로
    배수 제약을 보인다(예 `151300000566` 정렬도 1.000). 단 마스터에 있는 품목은 광교에서
    정렬도 0.894~0.993으로 **전부 유효**하다. 즉 마스터는 "틀린 게 아니라 부분집합"이다.
    """
    raw = pd.read_excel(xlsx_path, sheet_name="브레드 맞춤수량")
    raw.columns = ["item_name", "rule"][: len(raw.columns)]
    raw = raw.dropna(subset=["item_name"])
    out: dict[str, int] = {}
    for row in raw.itertuples():
        unit = BATCH_RULE_TO_UNIT.get(str(row.rule).strip())
        if unit is None:
            continue
        item_id = name_to_item.get(_normalize_name(row.item_name))
        if item_id is not None:
            out[str(item_id)] = unit
    return out


def _normalize_name(name: object) -> str:
    """품목명 정규화 — 공백·괄호·언더스코어 제거 후 소문자."""
    import re

    return re.sub(r"[\s_()]", "", str(name)).lower()


def estimate_batch_units(
    inventory: pd.DataFrame,
    *,
    item_col: str = "item_id",
    made_col: str = "production_qty",
    candidates: tuple[int, ...] = (2, 3, 4, 6, 8),
    min_rows: int = 60,
    min_rate: float = 0.80,
) -> pd.DataFrame:
    """실측 생산량에서 품목별 배수를 추정 — 마스터 누락을 보완한다.

    정렬도가 `min_rate` 이상이고 **우연 기대치(1/k)보다 뚜렷히 높은** 배수만 채택한다.
    반환: [item_id, unit, align_rate, n] (정렬도 내림차순).

    ⚠️추정이므로 마스터와 불일치할 수 있다. 광교에서는 마스터 4품목이 모두 정렬도 0.894+로
    일치했다(`151100003487`만 n=47로 `min_rows` 미달이라 추정 목록에서 빠진다 — 규칙 위반이
    아니라 표본 부족이다).
    ⚠️**매장을 반드시 확인할 것** — `1000000047`=광교 / `1000000009`=삼성타운이다.
    매장을 잘못 잡으면 정렬도가 전혀 다르게 나온다(예 `151100002803`: 광교 0.961 vs
    삼성타운 0.336). `load_inventory(..., "store_gw01")` 처럼 store_id로 접근하면 안전하다.
    """
    rows = []
    made = pd.to_numeric(inventory[made_col], errors="coerce").fillna(0)
    work = inventory.assign(_made=made)
    work = work[work["_made"] > 0]
    for item, group in work.groupby(item_col):
        if len(group) < min_rows:
            continue
        best, best_rate = None, 0.0
        for unit in candidates:
            rate = float((group["_made"] % unit == 0).mean())
            if rate >= min_rate and rate - 1.0 / unit > best_rate - 1.0 / (best or 1):
                best, best_rate = unit, rate
        if best is not None:
            rows.append({"item_id": str(item), "unit": best,
                         "align_rate": best_rate, "n": len(group)})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["item_id", "unit", "align_rate", "n"])
    return out.sort_values("align_rate", ascending=False).reset_index(drop=True)


def resolve_units_master_first(
    master_units: dict[str, int],
    estimated: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame]:
    """마스터 우선 + 데이터 추정으로 보완한 배수 맵과 대조표.

    마스터(아띠제 공식 의도)를 먼저 쓰고, 마스터에 없는 품목만 추정값으로 채운다.
    ★마스터는 누락이 있다 — 광교 실측에서 마스터 4품목 외에 **17품목**이 정렬도 0.80+로
    배수 제약을 보인다. 반대로 마스터 품목은 광교에서 정렬도 0.894~0.993으로 전부 유효하다.

    반환: (unit_map, 대조표[item_id, source, unit, master_unit, est_unit, align_rate])
    """
    est_map = {str(r.item_id): int(r.unit) for r in estimated.itertuples()}
    est_rate = {str(r.item_id): float(r.align_rate) for r in estimated.itertuples()}
    unit_map = dict(est_map)
    unit_map.update({str(k): int(v) for k, v in master_units.items()})   # 마스터가 이긴다
    rows = [
        {
            "item_id": item,
            "source": "master" if item in master_units else "estimated",
            "unit": unit,
            "master_unit": master_units.get(item),
            "est_unit": est_map.get(item),
            "align_rate": est_rate.get(item),
        }
        for item, unit in sorted(unit_map.items())
    ]
    return unit_map, pd.DataFrame(rows)
