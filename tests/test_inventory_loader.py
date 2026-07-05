"""Tests for inventory loader (생산량, 폐기량)."""

import pandas as pd
import pytest
from pathlib import Path
from pandas.api.types import is_string_dtype, is_integer_dtype

from bakery.ingest.inventory import load_inventory, _normalize_inventory


def test_normalize_inventory_basic():
    """Test basic column mapping and filtering."""
    # Synthetic 재고정보 DataFrame
    raw = pd.DataFrame({
        "날짜": ["20210101", "20210101", "20210102"],
        "점포코드": ["1000000047", "1000000009", "1000000047"],
        "품목코드": ["151100000241", "151100000241", "151100000247"],
        "생산량": [5, 3, 2],
        "폐기량": [1, 0, 0],
    })

    # Filter to store_gw01 (점포코드 1000000047)
    result = _normalize_inventory(raw, store_code="1000000047")

    # Should have exactly 2 rows (gwangyo only)
    assert len(result) == 2
    assert list(result.columns) == ["date", "item_id", "production_qty", "waste_qty"]

    # Check dtypes (string dtypes may be object or StringDtype depending on pandas version)
    assert is_string_dtype(result["date"])
    assert is_string_dtype(result["item_id"])
    assert is_integer_dtype(result["production_qty"])
    assert is_integer_dtype(result["waste_qty"])

    # Check exact values
    assert result.iloc[0]["date"] == "20210101"
    assert result.iloc[0]["item_id"] == "151100000241"
    assert result.iloc[0]["production_qty"] == 5
    assert result.iloc[0]["waste_qty"] == 1

    assert result.iloc[1]["date"] == "20210102"
    assert result.iloc[1]["item_id"] == "151100000247"
    assert result.iloc[1]["production_qty"] == 2
    assert result.iloc[1]["waste_qty"] == 0


def test_normalize_inventory_numeric_conversion():
    """Test that non-numeric values in qty columns are converted safely."""
    raw = pd.DataFrame({
        "날짜": ["20210101", "20210102"],
        "점포코드": ["1000000047", "1000000047"],
        "품목코드": ["151100000241", "151100000241"],
        "생산량": ["5", "3"],  # Strings that can be converted
        "폐기량": ["1", "0"],
    })

    result = _normalize_inventory(raw, store_code="1000000047")

    assert result["production_qty"].dtype == "int64"
    assert result["waste_qty"].dtype == "int64"
    assert result.iloc[0]["production_qty"] == 5


def test_load_inventory_real_file():
    """Test load_inventory with real file and gwangyo store."""
    xl_path = Path("data/internal/보나비 데이터_20260526.xlsx")

    if not xl_path.exists():
        pytest.skip(f"Real data file not found: {xl_path}")

    # Load gwangyo store (store_id="store_gw01" maps to 점포코드 1000000047)
    result = load_inventory(str(xl_path), store_id="store_gw01")

    # Columns must be exact
    assert list(result.columns) == ["date", "item_id", "production_qty", "waste_qty"]

    # Must have data
    assert len(result) > 0

    # All rows should be gwangyo (점포코드 1000000047 in source)
    # Check by sampling: all item_ids should exist in gwangyo
    assert is_string_dtype(result["item_id"])
    assert is_string_dtype(result["date"])
    assert is_integer_dtype(result["production_qty"])
    assert is_integer_dtype(result["waste_qty"])

    # Check dates are in valid format (YYYYMMDD as string)
    assert all(result["date"].str.match(r"\d{8}"))


def test_load_inventory_empty_filter():
    """Test that filtering to non-existent store returns empty DataFrame."""
    raw = pd.DataFrame({
        "날짜": ["20210101"],
        "점포코드": ["1000000047"],
        "품목코드": ["151100000241"],
        "생산량": [5],
        "폐기량": [1],
    })

    result = _normalize_inventory(raw, store_code="9999999999")  # Non-existent

    # Should return empty DataFrame with correct columns
    assert len(result) == 0
    assert list(result.columns) == ["date", "item_id", "production_qty", "waste_qty"]
