import pandas as pd
import pytest

from src.models.market.data_prep.size_ratios import compute_size_ratios


def test_compute_size_ratios_averages_ratio_over_time():
    combined_rows = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 100.0, "price": 2.0},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": "1", "landings_kg": 40.0, "price": 2.2},
        {"date": pd.Timestamp("2020-02-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 90.0, "price": 2.5},
        {"date": pd.Timestamp("2020-02-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": "1", "landings_kg": 30.0, "price": 2.75},
    ])

    result = compute_size_ratios(combined_rows)

    row = result[
        (result["market_code"] == "ES")
        & (result["species_code"] == "ANE")
        & (result["size"] == "1")
    ]
    assert row.iloc[0]["ratio"] == pytest.approx((2.2 / 2.0 + 2.75 / 2.5) / 2)


def test_compute_size_ratios_excludes_months_missing_a_blended_row():
    combined_rows = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": "1", "landings_kg": 40.0, "price": 2.2},
    ])

    result = compute_size_ratios(combined_rows)

    assert len(result) == 0
