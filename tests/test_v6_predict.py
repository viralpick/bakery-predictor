import pandas as pd
import pytest
from typer.testing import CliRunner

from bakery.cli import app
from bakery.data.calendar import build_calendar_daily
from bakery.data.loader import DailyDataset
from bakery.data.weather import build_synthetic_weather


def _dataset(daily, calendar, weather):
    """DailyDataset은 7개 프레임 전부 필수(frozen dataclass). 미사용은 빈 프레임."""
    empty = pd.DataFrame()
    return DailyDataset(daily=daily, weather=weather, calendar=calendar, competitor=empty,
                        living_population=empty, population=empty, consumption=empty)


def _synthetic(n_days=140):
    """v2 feature_set은 calendar/weather 병합이 필요 — horizon(+7일)까지 커버해야 한다."""
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
    horizon_end = dates.max() + pd.Timedelta(days=7)
    calendar = build_calendar_daily(dates.min(), horizon_end)
    weather = build_synthetic_weather(dates.min(), horizon_end, store_ids=["store_A"], seed=7)
    return _dataset(pd.DataFrame(rows), calendar, weather)


@pytest.fixture
def patch_ds(monkeypatch):
    monkeypatch.setattr("bakery.cli._load_dataset", lambda source, data_dir: _synthetic())


def test_v6_predict_synthetic_smoke(patch_ds, tmp_path):
    """synthetic v6-predict가 v2 기본모델로 정상 실행(회귀 가드)."""
    runner = CliRunner()
    result = runner.invoke(app, [
        "v6-predict", "--source", "synthetic", "--model", "lightgbm_v2",
        "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    rec = pd.read_csv(tmp_path / "v6_recommendations.csv")
    assert len(rec) > 0
