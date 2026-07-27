
import pandas as pd
import pytest

from bakery.analysis.lab.inputs import STORE_CODES, STORE_NAMES, AnalysisInputs
from bakery.analysis.lab.spec import AnalysisSpec


def _spec(**over):
    body = {"name": "t", "data": {"source": "real"}}
    body.update(over)
    return AnalysisSpec(**body)


def test_store_prior_keys_are_english_labels():
    from bakery.analysis.lab.inputs import STORE_PRIOR_KEYS
    from bakery.harness.event_priors import STORE_EVENT_PRIORS

    assert STORE_PRIOR_KEYS == {"store_gw01": "gwangyo", "store_ss01": "samsung",
                                "store_mp01": "mecenatpolis", "store_gh01": "gwanghwamun"}
    assert set(STORE_PRIOR_KEYS.values()) == set(STORE_EVENT_PRIORS)


def test_prior_key_falls_back_to_gwangyo_for_multistore():
    assert AnalysisInputs.from_spec(_spec()).prior_key == "gwangyo"
    assert AnalysisInputs.from_spec(
        _spec(data={"source": "real", "store": "multistore"})).prior_key == "gwangyo"
    assert AnalysisInputs.from_spec(
        _spec(data={"source": "real", "store": "store_mp01"})).prior_key == "mecenatpolis"


def test_store_codes_cover_four_stores():
    assert STORE_CODES == {
        "store_gw01": "1000000047",
        "store_ss01": "1000000009",
        "store_mp01": "1000000029",
        "store_gh01": "1000000485",
    }
    assert STORE_NAMES["store_gw01"] == "광교"


def test_is_multistore_flag():
    assert AnalysisInputs.from_spec(_spec()).is_multistore is False
    assert AnalysisInputs.from_spec(
        _spec(data={"source": "real", "store": "multistore"})).is_multistore is True


def test_has_predictions_false_when_unset():
    inputs = AnalysisInputs.from_spec(_spec())
    assert inputs.has_predictions is False
    assert inputs.predictions is None


def test_has_predictions_false_when_path_missing(tmp_path):
    inputs = AnalysisInputs.from_spec(_spec(predictions=tmp_path / "nope.csv"))
    assert inputs.has_predictions is False


def test_predictions_loaded_with_parsed_dates(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("date,fold,actual,expected,production\n"
                    "2025-12-25,0,307.6,315.8,356.7\n", encoding="utf-8")
    inputs = AnalysisInputs.from_spec(_spec(predictions=path))
    assert inputs.has_predictions is True
    preds = inputs.predictions
    assert preds["date"].tolist() == [pd.Timestamp("2025-12-25")]
    assert preds["expected"].tolist() == [315.8]


def test_params_for_returns_empty_dict_when_absent():
    inputs = AnalysisInputs.from_spec(_spec())
    assert inputs.params_for("demand_absorption") == {}


def test_params_for_returns_declared_params():
    inputs = AnalysisInputs.from_spec(_spec(params={"demand_absorption": {"close_hour": 21}}))
    assert inputs.params_for("demand_absorption") == {"close_hour": 21}


@pytest.mark.slow
def test_daily_drops_potential_demand_and_filters_store():
    # 측정 헌장: potential_demand는 오염 소스 — 로더에서 아예 제거해 소비 불가로 만든다
    inputs = AnalysisInputs.from_spec(_spec())
    daily = inputs.daily
    assert "potential_demand" not in daily.columns
    assert daily["store_id"].unique().tolist() == ["store_gw01"]
    assert daily["is_stockout"].dtype == bool


@pytest.mark.slow
def test_multistore_daily_has_four_stores():
    inputs = AnalysisInputs.from_spec(_spec(data={"source": "real", "store": "multistore"}))
    assert sorted(inputs.daily["store_id"].unique()) == [
        "store_gh01", "store_gw01", "store_mp01", "store_ss01"]


@pytest.mark.slow
def test_daily_is_cached_single_read():
    inputs = AnalysisInputs.from_spec(_spec())
    assert inputs.daily is inputs.daily        # cached_property 동일 객체


@pytest.mark.slow
def test_waste_frame_renames_without_transforming_values():
    from bakery.data import paths

    inputs = AnalysisInputs.from_spec(_spec())
    waste = inputs.waste
    assert set(waste.columns) >= {"item_id", "date", "production_qty", "waste_qty"}
    assert waste["cd"].unique().tolist() == ["1000000047"]
    # 순수 rename 계약: made→production_qty, out→waste_qty 값 변형 없음.
    # 음수 waste_qty는 전일 재고 이월(carry-in)로 판매가 당일 생산을 초과한 실제 신호이며
    # (해당 행에서 identity_diff==0), clip하면 항등식이 깨지고 폐기율이 부풀려진다.
    raw = pd.read_parquet(paths.dataset("waste_alpha_4stores"))
    raw = raw[raw["cd"].astype(str) == "1000000047"].reset_index(drop=True)
    assert waste["production_qty"].tolist() == raw["made"].tolist()
    assert waste["waste_qty"].tolist() == raw["out"].tolist()


@pytest.mark.slow
def test_item_to_category_maps_bread():
    inputs = AnalysisInputs.from_spec(_spec())
    mapping = inputs.item_to_category
    daily = inputs.daily
    first = daily.iloc[0]
    assert mapping[first["item_id"]] == first["category_id"]
