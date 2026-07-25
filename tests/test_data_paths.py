import pytest
from bakery.data import paths


def test_layer_roots_under_data():
    assert paths.RAW_DIR == paths.PROJECT_ROOT / "data" / "raw"
    assert paths.INTERIM_DIR == paths.PROJECT_ROOT / "data" / "interim"
    assert paths.PROCESSED_DIR == paths.PROJECT_ROOT / "data" / "processed"


def test_dataset_resolves_internal_processed():
    assert paths.dataset("bonavi_daily") == (
        paths.PROCESSED_DIR / "internal" / "bonavi_daily.parquet"
    )
    assert paths.dataset("bonavi_receipts") == (
        paths.PROCESSED_DIR / "internal" / "bonavi_receipts.parquet"
    )


def test_dataset_resolves_interim_and_external():
    assert paths.dataset("sales_lines_clean") == (
        paths.INTERIM_DIR / "sales_lines_clean.parquet"
    )
    assert paths.dataset("weather_observed") == (
        paths.PROCESSED_DIR / "external" / "weather_observed.parquet"
    )


def test_dataset_resolves_raw_sources():
    assert paths.dataset("sales_xlsx") == (
        paths.RAW_DIR / "internal" / "보나비 판매 데이터_20260721.xlsx"
    )
    assert paths.dataset("master_xlsx") == (
        paths.RAW_DIR / "internal" / "보나비 데이터_20260526.xlsx"
    )


def test_unknown_dataset_raises_keyerror_with_known_names():
    with pytest.raises(KeyError, match="unknown dataset"):
        paths.dataset("does_not_exist")
    assert "bonavi_daily" in paths.list_datasets()


def test_dataset_resolves_waste_alpha_4stores():
    """src/bakery/cli.py:1616 CLOSING_DEMAND_WASTE_PARQUET consumer — Task 5 migration target."""
    assert paths.dataset("waste_alpha_4stores") == (
        paths.PROCESSED_DIR / "internal" / "waste_alpha_4stores.parquet"
    )


def test_living_pop_zips_dir_resolves():
    """src/bakery/ingest/living_population_csv.py:32 ZIP_DIR_DEFAULT consumer.

    Directory of dynamically-named zips, not a single .parquet — doesn't fit dataset(name),
    so it's a standalone module constant instead.
    """
    assert paths.LIVING_POP_ZIPS_DIR == paths.RAW_DIR / "external" / "living_pop_zips"
