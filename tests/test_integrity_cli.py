import pandas as pd

from bakery.data import integrity


def test_run_all_aggregates_and_exit_severity():
    # 합성: 정상 sales + 타깃 누락 1건 → fail 존재
    sales = pd.DataFrame({
        "NO_POS": ["1"], "SLIP_NO": ["1"], "SLIP_LINE": ["1"],
        "SALES_FG": ["0"], "SALES_TIME": ["20260101120000"],
        "DT_SALE": ["20260101"], "CD_ITEM": ["A101"], "CD_USERDEF1": [""],
    })
    violations, missing = integrity.run_all(
        sales=sales,
        master_item_codes={"A999"},           # A101 없음
        target_items={"A101"}, used_discounts=set(), master_disc={"0069"},
        schema={},
    )
    # 타깃 A101 누락 → fail
    assert any(v.severity == "fail" and v.check == "target_items_resolve" for v in violations)
    assert integrity.has_fail(violations) is True
    # A101이 missing_df에 target_scope=True로
    assert missing.set_index("code").loc["A101", "is_target_scope"] == True  # noqa: E712


def test_run_conflict_diagnostic_target_flag_flip_fails():
    # 타깃 플래그(CD_USERDEF4) Y→N 변경 = 라벨 오염 → fail
    old = pd.DataFrame({"CD_ITEM": ["A101"], "CD_USERDEF4": ["Y"]})
    new = pd.DataFrame({"CD_ITEM": ["A101"], "CD_USERDEF4": ["N"]})
    v, conflicting = integrity.run_conflict_diagnostic(
        old, new, fields=["CD_USERDEF4"], scope_codes={"A101"})
    assert integrity.has_fail(v) is True
    assert conflicting.iloc[0]["old_value"] == "Y" and conflicting.iloc[0]["new_value"] == "N"


def test_has_fail_false_when_only_drift():
    v = [integrity.Violation("return_ratio", "drift", "x", 1)]
    assert integrity.has_fail(v) is False
