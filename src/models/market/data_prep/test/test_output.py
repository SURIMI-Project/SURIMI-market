import pandas as pd

from src.models.market.data_prep.output import build_prices_csv_rows, write_prices_csv, write_size_ratios_csv


def test_build_prices_csv_rows_adds_constant_columns():
    local = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 50.0, "price": 2.0},
    ])
    combined = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 50.0, "price": 2.0},
    ])

    result = build_prices_csv_rows(local, combined)

    assert len(result) == 2
    assert (result["currency"] == "EUR").all()
    assert (result["measurement_unit"] == "kg").all()


def test_write_prices_csv_round_trips(tmp_path):
    rows = pd.DataFrame([
        {"date": "2020-01-01", "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "landings_kg": 50.0, "price": 2.0,
         "currency": "EUR", "measurement_unit": "kg", "size": float("nan")},
    ])
    path = tmp_path / "prices.csv"

    write_prices_csv(rows, path)
    reloaded = pd.read_csv(path)

    assert reloaded.iloc[0]["market_code"] == "ESALC"
    assert pd.isna(reloaded.iloc[0]["size"])


def test_write_size_ratios_csv_round_trips(tmp_path):
    ratios = pd.DataFrame([{"market_code": "ES", "species_code": "ANE", "size": "1", "ratio": 1.1}])
    path = tmp_path / "size_ratios.csv"

    write_size_ratios_csv(ratios, path)
    reloaded = pd.read_csv(path)

    assert reloaded.iloc[0]["ratio"] == 1.1
