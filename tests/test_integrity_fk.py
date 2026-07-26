from bakery.data import integrity


def test_target_item_missing_from_master_fails():
    # 타깃 품목 A103이 마스터에 없음 → fail
    v = integrity.check_target_items_resolve(
        sale_item_codes={"A101", "A102", "A103"},
        master_item_codes={"A101", "A102"},   # A103 없음
        target_items={"A101", "A103"},         # A103은 타깃
    )
    assert len(v) == 1 and v[0].severity == "fail" and v[0].count == 1
    assert "A103" in v[0].detail


def test_non_target_orphan_does_not_fail():
    # A103 orphan이지만 타깃 아님 → fail 없음 (drift는 CSV에서 처리)
    v = integrity.check_target_items_resolve(
        sale_item_codes={"A101", "A103"}, master_item_codes={"A101"},
        target_items={"A101"},
    )
    assert v == []


def test_used_discount_missing_is_drift_not_fail():
    # ★코드 정규화 규칙 미확정(3/4자리 혼재)이라 fail 아닌 drift (advisor #1 실측)
    v = integrity.check_used_discounts_resolve(
        used_disc={"0069", "9999"}, master_disc={"0069"})  # 9999 사용됐는데 마스터 없음
    assert len(v) == 1 and v[0].severity == "drift" and "9999" in v[0].detail


def test_find_missing_codes_marks_target_scope():
    df = integrity.find_missing_codes(
        sale_codes={"A101", "A102", "A103"}, master_codes={"A101"},
        kind="item", target_codes={"A102"}, used_codes=set())
    # A102, A103 누락. A102는 타깃 → is_target_scope True
    miss = df.set_index("code")["is_target_scope"].to_dict()
    assert miss == {"A102": True, "A103": False}
    assert set(df["kind"]) == {"item"}
