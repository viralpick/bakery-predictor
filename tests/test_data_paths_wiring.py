"""bonavi_loader_v2 모듈 상수가 하드코딩 리터럴이 아니라 paths registry를
가리키는지 확인한다 (Task 4: 내부 loader 소비처 마이그레이션).
"""
from bakery.data import bonavi_loader_v2 as v2
from bakery.data import paths


def test_bonavi_loader_v2_constants_use_registry():
    assert v2.NEW_SALES_XLSX == paths.dataset("sales_xlsx")
    assert v2.MASTER_XLSX == paths.dataset("master_xlsx")
    assert v2.CLEAN_PARQUET == paths.dataset("sales_lines_clean")
    assert v2.OUT_DEFAULT == paths.dataset("bonavi_daily")
    assert v2.RECEIPTS_DEFAULT == paths.dataset("bonavi_receipts")
