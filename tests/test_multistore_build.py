# tests/test_multistore_build.py
"""build_multistore: 4매장 daily를 별도 parquet로. 광교 파트는 canonical과 정합."""
import numpy as np
import pandas as pd
import pytest
from bakery.data import bonavi_loader_v2 as v2
from bakery.data import paths
from bakery.data.schema import DAILY_COLUMNS, validate_daily


@pytest.fixture(scope="module")
def multistore(tmp_path_factory):
    out = tmp_path_factory.mktemp("ms") / "multistore_daily.parquet"
    v2.build_multistore(out_path=out)
    return pd.read_parquet(out)


def test_multistore_has_four_stores(multistore):
    counts = multistore.groupby("store_id").size()
    assert (counts > 0).all()
    assert set(counts.index) == {
        "store_gw01", "store_ss01", "store_gh01", "store_mp01"}


def test_multistore_schema(multistore):
    assert list(multistore.columns) == list(DAILY_COLUMNS.keys())
    validate_daily(multistore)   # raise 없으면 통과


def test_multistore_gwangyo_matches_canonical(multistore):
    """multistore의 광교 파트 == 기존 canonical bonavi_daily (max_diff=0)."""
    canon = pd.read_parquet(paths.dataset("bonavi_daily"))
    gw = multistore[multistore["store_id"] == "store_gw01"].copy()
    keys = ["date", "item_id"]
    gw = gw.sort_values(keys).reset_index(drop=True)
    canon = canon.sort_values(keys).reset_index(drop=True)
    assert len(gw) == len(canon)
    for c in gw.select_dtypes(include=[np.number]).columns:
        max_diff = float(np.abs(gw[c].fillna(-9e9).to_numpy() - canon[c].fillna(-9e9).to_numpy()).max())
        assert max_diff == 0.0, f"{c} diff={max_diff}"
