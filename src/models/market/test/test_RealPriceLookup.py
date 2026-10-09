import pandas as pd

from src.models.market.RealPriceLookup import RealPriceLookup


def _prices_df():
    return pd.DataFrame({
        "market_code": ["ESALC", "ESALC", "ESBCN"],
        "species_code": ["ANE", "ANE", "ANE"],
        "category_code": ["Fresh - Whole"] * 3,
        "date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-01-01"]),
        "price": [2.0, 2.1, 3.0],
        "size": [None, None, None],
    })


def test_load_then_get_returns_the_matching_markets_rows_only():
    lookup = RealPriceLookup()
    lookup.load(_prices_df(), market_code="ESALC")

    assert lookup.get("ANE", "Fresh - Whole", pd.Timestamp("2022-01-01")) == 2.0
    assert lookup.get("ANE", "Fresh - Whole", pd.Timestamp("2022-02-01")) == 2.1


def test_get_returns_none_for_a_different_market_or_unknown_date():
    lookup = RealPriceLookup()
    lookup.load(_prices_df(), market_code="ESALC")

    assert lookup.get("ANE", "Fresh - Whole", pd.Timestamp("2022-03-01")) is None
    assert lookup.get("ANE", "Fresh - Whole", pd.Timestamp("2022-01-01")) != 3.0  # that row belongs to ESBCN
