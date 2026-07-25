import numpy as np, pandas as pd, pytest
from bakery.data import pipeline, paths


@pytest.mark.slow
def test_build_internal_reproduces_bonavi_daily(tmp_path):
    """clean→daily 결정적 재생성이 on-disk와 rtol=1e-9 일치 (2026-07-25 진단으로 확인됨)."""
    if not paths.dataset("sales_lines_clean").exists():
        pytest.skip("interim clean parquet 부재")
    rebuilt = pipeline.build_internal(reconvert=False, out_root=tmp_path)
    regen = pd.read_parquet(rebuilt["bonavi_daily"])
    disk = pd.read_parquet(paths.dataset("bonavi_daily"))
    keys = ["date", "item_id", "store_id"]
    r = regen.sort_values(keys).reset_index(drop=True)
    d = disk.sort_values(keys).reset_index(drop=True)
    assert list(r.columns) == list(d.columns)
    for c in r.select_dtypes(include=[np.number]).columns:
        max_diff = (r[c].fillna(-9e9) - d[c].fillna(-9e9)).abs().max()
        assert max_diff <= 1e-9, f"{c} diverged by {max_diff}"
