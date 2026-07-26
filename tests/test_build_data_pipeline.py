import numpy as np
import pandas as pd
import pytest

from bakery.data import paths, pipeline


def _max_numeric_diff(regen: pd.DataFrame, disk: pd.DataFrame) -> float:
    """정렬 후 수치 컬럼 최대 절대 diff. 컬럼 계약도 함께 단언."""
    keys = [c for c in ("date", "item_id", "store_id") if c in regen.columns]
    r = regen.sort_values(keys).reset_index(drop=True)
    d = disk.sort_values(keys).reset_index(drop=True)
    assert list(r.columns) == list(d.columns)
    worst = 0.0
    for c in r.select_dtypes(include=[np.number]).columns:
        worst = max(worst, float((r[c].fillna(-9e9) - d[c].fillna(-9e9)).abs().max()))
    return worst


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory):
    """clean→daily/receipts 1회 재생성(module 스코프, 두 테이블 공유)."""
    if not paths.dataset("sales_lines_clean").exists():
        pytest.skip("interim clean parquet 부재")
    out_root = tmp_path_factory.mktemp("build_internal")
    return pipeline.build_internal(reconvert=False, out_root=out_root)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["bonavi_daily", "bonavi_receipts"])
def test_build_internal_reproduces_disk(rebuilt, name):
    """결정적 재생성이 on-disk와 rtol=1e-9 일치 (재배치 전제 실증, 2026-07-25 진단)."""
    regen = pd.read_parquet(rebuilt[name])
    disk = pd.read_parquet(paths.dataset(name))
    assert _max_numeric_diff(regen, disk) <= 1e-9
