import pandas as pd
import pytest
from typer.testing import CliRunner

from bakery.cli import app
from bakery.data.loader import DailyDataset


def _dataset(daily):
    """DailyDataset은 7개 프레임 전부 필수(frozen dataclass). 미사용은 빈 프레임."""
    empty = pd.DataFrame()
    return DailyDataset(daily=daily, weather=empty, calendar=empty, competitor=empty,
                        living_population=empty, population=empty, consumption=empty)


def _synthetic_dataset(n_days=140):
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        for item in ("A", "B"):
            rows.append({
                "store_id": "store_A", "item_id": item, "category_id": "bread",
                "date": d, "sold_units": 10 + (item == "B") * 5,
                "is_stockout": False, "stockout_time": pd.NaT,
                "open_hours": 13, "capacity": 100,
                "potential_demand": 10.0 + (item == "B") * 5,
            })
    return _dataset(pd.DataFrame(rows))


@pytest.fixture
def patch_v0_dataset(monkeypatch):
    monkeypatch.setattr("bakery.cli._load_dataset",
                        lambda source, data_dir: _synthetic_dataset())


def test_predict_v0_synthetic_writes_sold_units_col(patch_v0_dataset, tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, [
        "predict-next-week", "--source", "synthetic",
        "--model", "lightgbm", "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    out = pd.read_csv(tmp_path / "next_week_predictions.csv")
    assert "yhat_sold_units" in out.columns
