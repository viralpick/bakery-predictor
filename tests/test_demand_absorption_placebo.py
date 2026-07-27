import pandas as pd

from bakery.analysis.demand_absorption import (
    GATE_CATEGORIES,
    PLACEBO_HORIZON_DAYS,
    build_absorption_panel,
    fit_absorption,
    placebo_absorption,
)


def _daily():
    """2카테고리 × 200일. 품절강도가 총량에 미치는 영향을 회귀할 수 있는 최소 패널."""
    rows = []
    for day in range(200):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for category, items in (("bread", ["b1", "b2"]), ("pastry", ["p1", "p2"])):
            for index, item in enumerate(items):
                is_stockout = (day + index) % 3 == 0
                # ★ sold_units에 (day % 11)을 더하는 이유: 11은 7과 서로소라서 dow 주기와 어긋난다.
                # 이게 없으면 2024-01-01이 월요일이라 day%7 == dow가 되어 cat_baseline이 dow 더미의
                # 정확한 선형결합이 된다(cond(X)≈2.4e18) → fit_absorption이 전부 None을 반환하고
                # 양쪽 arm이 빈 리스트가 되어 등가 테스트가 무증상 통과한다.
                rows.append({
                    "store_id": "store_gw01", "item_id": item, "category_id": category,
                    "date": date, "sold_units": 10 + (day % 7) + index * 2 + (day % 11),
                    "is_stockout": is_stockout,
                    "stockout_time": date + pd.Timedelta(hours=19) if is_stockout else pd.NaT,
                })
    return pd.DataFrame(rows)


def test_gate_categories_are_bread_and_pastry():
    assert GATE_CATEGORIES == ("bread", "pastry")
    assert PLACEBO_HORIZON_DAYS == 7


def test_placebo_shifts_treatment_forward_by_horizon():
    """placebo = 미래 d+7 품절강도로 회귀 — 허위상관 크기의 하한."""
    daily = _daily()
    results = placebo_absorption(daily)
    # 참조 계산: 패널을 만들고 처치변수만 -7 shift 후 같은 fitter를 돌린다
    panel = build_absorption_panel(daily).sort_values("date")
    panel["stockout_hours"] = (panel.groupby(["store_id", "category_id"])["stockout_hours"]
                               .shift(-PLACEBO_HORIZON_DAYS))
    panel = panel.dropna(subset=["stockout_hours"])
    expected = [fit_absorption(panel, s, c)
                for s, c in panel[["store_id", "category_id"]]
                .drop_duplicates().itertuples(index=False)]
    expected = [r for r in expected if r is not None]
    assert len(results) == 2          # bread/pastry 둘 다 fit 성공 — 빈 리스트 비교로 무증상 통과 방지
    assert [r.category_id for r in results] == [r.category_id for r in expected]
    assert [r.n for r in results] == [r.n for r in expected]
    assert [r.beta for r in results] == [r.beta for r in expected]
    assert [r.verdict for r in results] == [r.verdict for r in expected]


def test_placebo_has_fewer_rows_than_real():
    daily = _daily()
    real_n = build_absorption_panel(daily).groupby("category_id").size().min()
    placebo_n = min(r.n for r in placebo_absorption(daily))
    assert placebo_n == real_n - PLACEBO_HORIZON_DAYS


def test_script_delegates_to_primitive():
    """scripts/absorption_4stores.py는 얇은 wrapper여야 한다(로직 중복 금지)."""
    import sys
    sys.path.insert(0, "scripts")
    import absorption_4stores

    assert absorption_4stores.placebo_results is placebo_absorption
