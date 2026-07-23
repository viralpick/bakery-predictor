"""신규 라인레벨 파일 어댑터(bonavi_loader_v2) 테스트.

- maybe_swap_fg_time: 판매정보2 헤더 스왑 교정 (CI-safe 유닛).
- load_items_v2: 타깃 정의(당일폐기Y − salad) 구조 (마스터 파일 존재 시).
"""

import pandas as pd
import pytest

from bakery.data.bonavi_loader_v2 import (
    MASTER_XLSX,
    load_items_v2,
    maybe_swap_fg_time,
)

_TS = "20210101100000"  # 14자리 판매시간 예시


def test_swap_corrects_flipped_sheet():
    """헤더가 뒤바뀐 시트(SALES_FG에 타임스탬프) → 교정 후 SALES_FG가 0/1."""
    flipped = pd.DataFrame({"SALES_FG": [_TS, _TS, _TS], "SALES_TIME": ["0", "1", "0"]})
    out = maybe_swap_fg_time(flipped)
    assert out["SALES_FG"].tolist() == ["0", "1", "0"]
    assert out["SALES_TIME"].tolist() == [_TS, _TS, _TS]


def test_swap_leaves_normal_sheet_untouched():
    """정상 시트(SALES_FG가 이미 0/1) → 변경 없음."""
    normal = pd.DataFrame({"SALES_FG": ["0", "1", "0"], "SALES_TIME": [_TS, _TS, _TS]})
    out = maybe_swap_fg_time(normal)
    assert out["SALES_FG"].tolist() == ["0", "1", "0"]
    assert out["SALES_TIME"].tolist() == [_TS, _TS, _TS]


@pytest.mark.skipif(not MASTER_XLSX.exists(), reason="0526 마스터 파일 로컬 전용")
def test_load_items_v2_target_definition():
    """타깃 = 당일폐기Y AND category != salad. 구조 불변식 검증."""
    items = load_items_v2()
    # 모든 타깃은 당일폐기=Y (역: Y가 아니면 타깃 아님)
    assert bool((items.loc[items["is_target"], "discard_daily"]).all())
    # 타깃에 salad 카테고리 없음
    assert "salad" not in set(items.loc[items["is_target"], "category_id"])
    # salad 품목은 당일폐기여도 타깃 아님
    salad = items[items["category_id"] == "salad"]
    if len(salad):
        assert not bool(salad["is_target"].any())
    # 타깃이 비어있지 않음
    assert int(items["is_target"].sum()) > 0
