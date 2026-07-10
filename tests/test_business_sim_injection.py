import pandas as pd
import pytest
from typer.testing import CliRunner

from bakery.cli import app
from bakery.data.loader import DailyDataset


def _dataset(daily):
    empty = pd.DataFrame()
    return DailyDataset(daily=daily, weather=empty, calendar=empty, competitor=empty,
                        living_population=empty, population=empty, consumption=empty)


def _real_like_dataset(n_days=140):
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        rows.append({
            "store_id": "store_A", "item_id": "A", "category_id": "bread",
            "date": d, "sold_units": 12, "is_stockout": False,
            "stockout_time": pd.NaT, "open_hours": 13, "capacity": 100,
            "potential_demand": 12.0,
        })
    return _dataset(pd.DataFrame(rows))


@pytest.fixture
def capture_profit(monkeypatch):
    """simulate_profit가 받은 potential_col을 기록. 실 모델·xlsx·discount 우회."""
    seen = {}

    def fake_profit(pred_df, *, potential_col=None, **kw):
        seen["potential_col"] = potential_col
        out = pred_df.copy()
        for c in ("revenue_krw", "waste_cost_krw", "lost_margin_krw", "net_profit_krw"):
            out[c] = 0.0
        return out

    monkeypatch.setattr("bakery.cli.simulate_profit", fake_profit)
    monkeypatch.setattr("bakery.cli._load_dataset",
                        lambda source, data_dir: _real_like_dataset())
    # adjusted_demand 부착 (real closing 파일 우회)
    monkeypatch.setattr(
        "bakery.cli.build_item_adjusted_demand",
        lambda daily, discount_rows=None, alpha=0.5: daily.assign(
            adjusted_demand=daily["sold_units"].astype(float)),
    )
    monkeypatch.setattr("bakery.cli.pd.read_excel",
                        lambda *a, **k: pd.DataFrame({"품목코드": ["A"], "상품구분": ["SS"],
                                                      "판매단가": [3000], "POS메뉴명": ["빵A"]}))
    return seen


def test_alpha_sweep_real_uses_adjusted_demand(capture_profit, tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, [
        "alpha-sweep", "--source", "real", "--variant", "v0",
        "--alphas", "0.5", "--n-splits", "2", "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert capture_profit["potential_col"] == "adjusted_demand"


def test_business_report_real_uses_adjusted_demand(capture_profit, tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, [
        "business-report", "--source", "real", "--variants", "v0",
        "--n-splits", "2", "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert capture_profit["potential_col"] == "adjusted_demand"
