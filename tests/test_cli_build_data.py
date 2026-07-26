import pytest

from bakery.data import paths, pipeline


@pytest.mark.slow
def test_build_internal_is_idempotent(tmp_path):
    if not paths.dataset("sales_lines_clean").exists():
        pytest.skip("interim 부재")
    a = pipeline.build_internal(out_root=tmp_path / "a")
    b = pipeline.build_internal(out_root=tmp_path / "b")
    diff = pipeline.equivalence_diff(a, {k: v for k, v in b.items()})
    assert all(v == 0.0 for v in diff.values()), diff
