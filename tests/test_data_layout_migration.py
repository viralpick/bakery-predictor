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


_STRAY_FILE_PATTERNS = ["*.parquet", "*.xlsx", "*.xls"]


def _stray_real_files(flat_dir: Path) -> list[Path]:
    """flat_dir 바로 아래에서, 심링크가 아닌 실체 데이터 파일(parquet/xlsx/xls)을 찾는다.
    이동 후 옛 평면 경로엔 심링크만 남아야 하므로, 실체 파일이 남아있으면 회귀다."""
    reals: list[Path] = []
    for pattern in _STRAY_FILE_PATTERNS:
        reals.extend(f for f in flat_dir.glob(pattern) if f.is_file() and not f.is_symlink())
    return reals


def test_no_parquet_left_in_flat_internal_root():
    flat = paths.DATA_DIR / "internal"
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    # 옛 평면 루트엔 심링크만 허용, 실체 parquet/xlsx/xls 금지(raw source 4종 회귀 포함 감지)
    reals = _stray_real_files(flat)
    assert reals == [], f"real data file still in flat internal root: {reals}"


def test_no_parquet_left_in_flat_internal_v2_root():
    flat_v2 = paths.DATA_DIR / "internal" / "v2"
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    reals = _stray_real_files(flat_v2)
    assert reals == [], f"real data file still in flat v2 root: {reals}"


def test_no_parquet_left_in_flat_external_root():
    flat = paths.DATA_DIR / "external"
    if not paths.RAW_DIR.exists():
        pytest.skip("data not migrated")
    # 9종 registry-moved parquet(weather_observed, calendar_raw 등)이 실체로 회귀하지 않았는지 확인
    reals = _stray_real_files(flat)
    assert reals == [], f"real data file still in flat external root: {reals}"


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
