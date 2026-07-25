"""이동 후 레이아웃/심링크/byte-identity 회귀 테스트.
데이터가 gitignored라 로컬 환경 의존 → 파일 부재 시 skip."""
from pathlib import Path

import pytest

from bakery.data import paths

_REQUIRED = [
    "bonavi_daily",
    "bonavi_receipts",
    "sales_lines_clean",
    "weather_observed",
    "calendar_raw",
    "sales_xlsx",
    "master_xlsx",
    "waste_alpha_4stores",
]


@pytest.mark.parametrize("name", _REQUIRED)
def test_dataset_exists_at_new_location(name):
    p = paths.dataset(name)
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated in this environment")
    assert p.exists(), f"{name} not at {p}"


def test_no_parquet_left_in_flat_internal_root():
    flat = paths.DATA_DIR / "internal"
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    # 옛 평면 루트엔 심링크만 허용, 실체 parquet 금지
    reals = [f for f in flat.glob("*.parquet") if f.is_file() and not f.is_symlink()]
    assert reals == [], f"real parquet still in flat root: {reals}"


def test_no_parquet_left_in_flat_internal_v2_root():
    flat_v2 = paths.DATA_DIR / "internal" / "v2"
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    reals = [f for f in flat_v2.glob("*.parquet") if f.is_file() and not f.is_symlink()]
    assert reals == [], f"real parquet still in flat v2 root: {reals}"


def test_legacy_symlink_resolves():
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    legacy = paths.DATA_DIR / "internal" / "bonavi_daily.parquet"
    assert legacy.exists()  # 심링크 통해 해석
    assert legacy.resolve() == paths.dataset("bonavi_daily").resolve()


def test_legacy_v2_waste_alpha_symlink_resolves():
    """유일한 v2/ src-consumed 파일 — old v2/ 경로 심링크가 registry 새 경로로 해석돼야 함."""
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    legacy = paths.DATA_DIR / "internal" / "v2" / "waste_alpha_4stores.parquet"
    assert legacy.exists()
    assert legacy.resolve() == paths.dataset("waste_alpha_4stores").resolve()


def test_legacy_living_pop_zips_symlink_resolves():
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    legacy = paths.DATA_DIR / "external" / "living_pop_zips"
    assert legacy.exists()
    assert legacy.resolve() == paths.LIVING_POP_ZIPS_DIR.resolve()


def test_cruft_archived_without_symlink():
    """cruft(.pre-*-bak 등)는 data/_archive/로 격리되고 옛 위치엔 아무것도 안 남는다(심링크 없음)."""
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    archive = paths.DATA_DIR / "_archive"
    assert archive.exists()
    old_cruft_paths = [
        paths.DATA_DIR / "internal" / "bonavi_daily.parquet.pre-v2-bak",
        paths.DATA_DIR / "internal" / "bonavi_receipts.parquet.pre-v2-bak",
        paths.DATA_DIR / "internal" / "v2" / "sales.parquet.pre-new-bak",
        paths.DATA_DIR / "internal" / "v2" / "item_active_stats.parquet",
    ]
    for old in old_cruft_paths:
        assert not old.exists(), f"cruft still at old location (should be archived, no symlink): {old}"
