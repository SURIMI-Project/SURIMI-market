from unittest.mock import patch

import pandas as pd

from src.models.market.data_prep.run import build_prefix_to_locals, run


def test_build_prefix_to_locals_groups_by_prefix(tmp_path):
    markets_csv = tmp_path / "markets.csv"
    pd.DataFrame({
        "market_code": ["ES", "FR", "ESALC", "ESBCN", "FRAGD"],
        "market_name": ["Spain", "France", "Alicante", "Barcelona", "Agde"],
        "lon": [None, None, -0.5, 2.2, None],
        "lat": [None, None, 38.3, 41.4, None],
    }).to_csv(markets_csv, index=False)

    result = build_prefix_to_locals(markets_csv)

    assert set(result["ES"]) == {"ESALC", "ESBCN"}
    assert set(result["FR"]) == {"FRAGD"}


def test_build_prefix_to_locals_infers_prefix_with_no_explicit_combined_row(tmp_path):
    markets_csv = tmp_path / "markets.csv"
    pd.DataFrame({
        "market_code": ["ESALC"],
        "market_name": ["Alicante"],
        "lon": [-0.5],
        "lat": [38.3],
    }).to_csv(markets_csv, index=False)

    result = build_prefix_to_locals(markets_csv)

    assert set(result["ES"]) == {"ESALC"}


def test_run_builds_prices_and_size_ratios_csv(tmp_path):
    markets_csv = tmp_path / "markets.csv"
    pd.DataFrame({
        "market_code": ["ES", "ESALC"],
        "market_name": ["Spain", "Alicante"],
        "lon": [None, -0.5],
        "lat": [None, 38.3],
    }).to_csv(markets_csv, index=False)

    fake_raw = pd.DataFrame([
        {"Date": pd.Timestamp("2020-01-01"), "Locode": "ESALC", "ERS_Code": "ANE",
         "Preservation": "Fresh", "Presentation": "Whole",
         "Value_EUR": 100.0, "Volume_Kg": 50.0, "Price_EUR_kg": 2.0, "Size": "1"},
    ])

    output_dir = tmp_path / "out"
    with patch("src.models.market.data_prep.run.load_fish_sales_data", return_value=fake_raw):
        run(markets_csv, output_dir, gdb_path=tmp_path / "fake.gdb")

    prices = pd.read_csv(output_dir / "prices.csv")
    assert len(prices) > 0
    assert "ES" in prices["market_code"].values
    assert "ESALC" in prices["market_code"].values

    ratios_path = output_dir / "size_ratios.csv"
    assert ratios_path.exists()
