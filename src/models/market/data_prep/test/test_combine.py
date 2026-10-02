import pandas as pd
import pytest

from src.models.market.data_prep.combine import build_combined_rows


def test_build_combined_rows_sums_local_ports_under_shared_prefix():
    local_rows = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 50.0, "price": 2.0},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESBCN", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 30.0, "price": 3.0},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "FRAGD", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 10.0, "price": 5.0},
    ])
    prefix_to_locals = {"ES": ["ESALC", "ESBCN"], "FR": ["FRAGD"]}

    result = build_combined_rows(local_rows, prefix_to_locals)

    es_row = result[result["market_code"] == "ES"].iloc[0]
    assert es_row["landings_kg"] == 80.0
    assert es_row["price"] == pytest.approx((50.0 * 2.0 + 30.0 * 3.0) / 80.0)

    fr_row = result[result["market_code"] == "FR"].iloc[0]
    assert fr_row["landings_kg"] == 10.0
    assert fr_row["price"] == 5.0


def test_build_combined_rows_preserves_size_grouping():
    local_rows = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": "1", "landings_kg": 20.0, "price": 2.5},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESBCN", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": "1", "landings_kg": 10.0, "price": 2.0},
    ])
    prefix_to_locals = {"ES": ["ESALC", "ESBCN"]}

    result = build_combined_rows(local_rows, prefix_to_locals)

    assert len(result) == 1
    assert result.iloc[0]["size"] == "1"
    assert result.iloc[0]["landings_kg"] == 30.0
