import pandas as pd
from bakery.ingest.inventory import handle_negative_waste


def test_clip_negative_waste_clips_and_reports():
    inv = pd.DataFrame({
        "date": ["20240101", "20240102", "20240103"],
        "item_id": ["a", "b", "c"],
        "production_qty": [10, 5, 8],
        "waste_qty": [-3, 2, -1],
    })
    cleaned, report = handle_negative_waste(inv, policy="clip")
    assert cleaned["waste_qty"].tolist() == [0, 2, 0]
    assert report == {"policy": "clip", "n_negative": 2, "n_total": 3, "min_value": -3.0}


def test_clip_no_negatives_reports_zero():
    inv = pd.DataFrame({"waste_qty": [0, 2, 5]})
    cleaned, report = handle_negative_waste(inv, policy="clip")
    assert cleaned["waste_qty"].tolist() == [0, 2, 5]
    assert report == {"policy": "clip", "n_negative": 0, "n_total": 3, "min_value": 0.0}
