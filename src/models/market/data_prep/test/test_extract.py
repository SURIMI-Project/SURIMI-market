import pandas as pd

from src.models.market.data_prep.extract import filter_and_clean


def _sample_raw_rows():
    return pd.DataFrame([
        {"Date": pd.Timestamp("2020-01-01"), "Locode": "ESALC", "ERS_Code": "ANE",
         "Preservation": "Fresh", "Presentation": "Whole",
         "Value_EUR": 100.0, "Volume_Kg": 50.0, "Price_EUR_kg": 2.0, "Size": "1"},
        {"Date": pd.Timestamp("2020-01-01"), "Locode": "ESALC", "ERS_Code": "ANE",
         "Preservation": "Frozen", "Presentation": "Whole",
         "Value_EUR": 40.0, "Volume_Kg": 20.0, "Price_EUR_kg": 2.0, "Size": "1"},
        {"Date": pd.Timestamp("2020-01-01"), "Locode": "ESXXX", "ERS_Code": "ANE",
         "Preservation": "Fresh", "Presentation": "Whole",
         "Value_EUR": 10.0, "Volume_Kg": 5.0, "Price_EUR_kg": 2.0, "Size": "1"},
    ])


def test_filter_and_clean_restricts_to_fresh_whole_and_known_markets():
    raw = _sample_raw_rows()
    result = filter_and_clean(raw, market_codes=["ESALC"])

    assert len(result) == 1
    assert result.iloc[0]["market_code"] == "ESALC"
    assert result.iloc[0]["category_code"] == "Fresh - Whole"


def test_filter_and_clean_renames_columns():
    raw = _sample_raw_rows()
    result = filter_and_clean(raw, market_codes=["ESALC", "ESXXX"])
    expected_columns = {
        "date", "market_code", "species_code", "category_code",
        "Value_EUR", "Volume_Kg", "Price_EUR_kg", "Size",
    }
    assert expected_columns.issubset(set(result.columns))
