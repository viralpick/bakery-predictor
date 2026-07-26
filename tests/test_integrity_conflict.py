import pandas as pd

from bakery.data import integrity


def _master(codes, names):
    return pd.DataFrame({"CD_ITEM": codes, "NM_ITEM": names})


def test_conflict_detects_changed_value():
    old = _master(["A1", "A2"], ["빵", "케이크"])
    new = _master(["A1", "A2"], ["빵", "케잌"])   # A2 이름 변경
    df = integrity.find_conflicting_codes(old, new, key="CD_ITEM", fields=["NM_ITEM"],
                                          kind="item", scope_codes={"A2"})
    assert list(df["code"]) == ["A2"]
    assert df.iloc[0]["old_value"] == "케이크" and df.iloc[0]["new_value"] == "케잌"
    assert bool(df.iloc[0]["is_target_scope"]) is True


def test_conflict_ignores_missing_codes():
    # 한쪽에만 있는 코드는 conflict 아님(누락은 Task 3 소관)
    old = _master(["A1"], ["빵"])
    new = _master(["A1", "A2"], ["빵", "쿠키"])
    df = integrity.find_conflicting_codes(old, new, key="CD_ITEM", fields=["NM_ITEM"],
                                          kind="item", scope_codes=set())
    assert df.empty


def test_scope_conflict_fails_else_drift():
    conflicts = pd.DataFrame({"code": ["A2", "A3"], "is_target_scope": [True, False]})
    v = integrity.check_scope_conflicts(conflicts)
    assert len(v) == 1 and v[0].severity == "fail" and v[0].count == 1  # A2만 fail
