import pandas as pd
import pytest

from src.models.market.data_prep.aggregate import build_local_rows


def test_build_local_rows_produces_blended_and_per_size_rows():
    cleaned = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "Value_EUR": 100.0, "Volume_Kg": 50.0,
         "Price_EUR_kg": 2.0, "Size": "1"},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "Value_EUR": 60.0, "Volume_Kg": 20.0,
         "Price_EUR_kg": 3.0, "Size": "2"},
    ])

    result = build_local_rows(cleaned)

    blended = result[result["size"].isna()]
    assert len(blended) == 1
    assert blended.iloc[0]["landings_kg"] == 70.0
    assert blended.iloc[0]["price"] == pytest.approx(160.0 / 70.0)

    size_1 = result[result["size"] == "1"]
    assert size_1.iloc[0]["landings_kg"] == 50.0
    assert size_1.iloc[0]["price"] == 2.0

    size_2 = result[result["size"] == "2"]
    assert size_2.iloc[0]["landings_kg"] == 20.0
    assert size_2.iloc[0]["price"] == 3.0


def test_build_local_rows_handles_rows_with_no_recorded_size():
    cleaned = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "SAA",
         "category_code": "Fresh - Whole", "Value_EUR": 30.0, "Volume_Kg": 10.0,
         "Price_EUR_kg": 3.0, "Size": None},
    ])

    result = build_local_rows(cleaned)

    assert len(result) == 1
    assert result.iloc[0]["size"] is None or pd.isna(result.iloc[0]["size"])
    assert result.iloc[0]["landings_kg"] == 10.0
