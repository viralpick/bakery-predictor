import pandas as pd

from bakery.data import coverage


def test_render_flags_missing_cell():
    cells = pd.DataFrame({
        "source": ["0721", "0520"],
        "store": ["광교", "광교"],
        "field": ["sold", "production"],
        "month": ["2026-06", "2026-06"],
        "present": [True, False],  # 2026-06 생산라벨 없음(§5 갭)
        "rows": [1234, 0],
    })
    html = coverage.render_coverage_matrix(cells)
    assert "2026-06" in html
    assert "production" in html
    # 결측 셀이 시각적으로 표시되는지 (class="missing")
    assert html.count("missing") == 1


def test_render_lists_known_conflicts():
    cells = pd.DataFrame({"source": ["0520"], "store": ["광교"], "field": ["closing"],
                          "month": ["2025-12"], "present": [True], "rows": [10]})
    html = coverage.render_coverage_matrix(cells, conflicts=["closing 소스 불일치: category=0520 vs item=clean"])
    assert "closing 소스 불일치" in html


def test_render_marks_structurally_absent_combo_as_na_not_present():
    # 서로 다른 (source,store,field) 행이 서로 다른 month 값을 쓰면(예: 날짜축 없는
    # 정적 품목표 vs 월별 시계열) pivot_table의 교차곱에 NaN 셀이 생긴다.
    # bool(NaN) == True이므로 이를 "present"로 오판하면 안 된다.
    cells = pd.DataFrame({
        "source": ["display_xls", "0721_sales"],
        "store": ["광교", "광교"],
        "field": ["display", "sold"],
        "month": ["static", "2026-01"],
        "present": [True, True],
        "rows": [37, 500],
    })
    html = coverage.render_coverage_matrix(cells)
    assert html.count('class="na"') == 2
    assert html.count("missing") == 0
