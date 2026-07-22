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


def test_predict_category_rejects_synthetic_source(tmp_path):
    """order_level=category는 real 전용 — synthetic이면 BadParameter로 종료(비-0)."""
    runner = CliRunner()
    result = runner.invoke(app, [
        "predict-next-week", "--source", "synthetic",
        "--order-level", "category", "--out-dir", str(tmp_path),
    ])
    assert result.exit_code != 0
    assert not (tmp_path / "next_week_predictions.csv").exists()


def _stub_future_orders():
    """_category_future_order_predictions 반환 스키마(6 cols)의 최소 2행."""
    return pd.DataFrame({
        "store_id": ["store_gw01", "store_gw01"],
        "item_id": ["A", "B"],
        "category_id": ["bread", "pastry"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        "demand_point": [8.7, 14.2],
        "our_order": [9.7, 14.9],
    })


def test_predict_category_handler_schema(monkeypatch, tmp_path):
    """category 모드 출력이 item 경로와 동일 8-col 스키마 + 규약을 지키는지 pin.

    real 데이터 없이 c-1 함수를 스텁 — category 분기는 _load_dataset 전에 반환한다."""
    monkeypatch.setattr(
        "bakery.cli._category_future_order_predictions",
        lambda *a, **k: _stub_future_orders(),
    )
    result = CliRunner().invoke(app, [
        "predict-next-week", "--source", "real", "--order-level", "category",
        "--total-model", "lightgbm", "--out-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    out = pd.read_csv(tmp_path / "next_week_predictions.csv")
    assert list(out.columns) == [
        "store_id", "item_id", "category_id", "date",
        "yhat_adjusted_demand_unit", "stockout_prob", "recommended_production", "model",
    ]
    assert out["recommended_production"].tolist() == [10.0, 15.0]   # our_order.round(0)
    assert out["yhat_adjusted_demand_unit"].tolist() == [8.7, 14.2]  # demand_point
    assert out["stockout_prob"].isna().all()
    assert out["model"].unique().tolist() == ["category_total:lightgbm"]


def test_category_base_predict_returns_none_sigma_for_lightgbm():
    """_category_base_predict는 (median, prod, sigma) 3-tuple. lightgbm은 sigma=None."""
    import numpy as np
    from bakery.cli import _category_base_predict

    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "adjusted_demand_unit": rng.uniform(100, 300, n),
        "dow": rng.integers(0, 7, n).astype(float),
        "is_holiday": rng.integers(0, 2, n).astype(float),
    })
    train, test = df.iloc[:50], df.iloc[50:]
    median, prod, sigma = _category_base_predict(
        train, test, target_col="adjusted_demand_unit",
        total_model="lightgbm", production_quantile=0.85,
    )
    assert len(median) == len(test)
    assert len(prod) == len(test)
    assert sigma is None
