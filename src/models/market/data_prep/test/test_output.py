import pandas as pd

from src.models.market.data_prep.combine import build_combined_rows
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


def test_write_prices_csv_orders_blended_row_last_within_group(tmp_path):
    # Rows deliberately out of order (per-size rows before the blended
    # row) to simulate whatever arbitrary order pd.concat happens to
    # produce. The blended row (size is null) must always land last
    # within its (date, market_code, species_code, category_code) group,
    # since Elsa's load_local_market_price_history does last-write-wins
    # keyed on (species_code, category_code, date).
    rows = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "landings_kg": 10.0, "price": 2.0,
         "currency": "EUR", "measurement_unit": "kg", "size": "2"},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "landings_kg": 50.0, "price": 2.5,
         "currency": "EUR", "measurement_unit": "kg", "size": float("nan")},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "landings_kg": 5.0, "price": 3.0,
         "currency": "EUR", "measurement_unit": "kg", "size": "1"},
    ])
    path = tmp_path / "prices.csv"

    write_prices_csv(rows, path)
    reloaded = pd.read_csv(path)

    assert len(reloaded) == 3
    last_row = reloaded.iloc[-1]
    assert pd.isna(last_row["size"])


def test_build_prices_csv_rows_preserves_date_dtype_with_all_empty_combined(tmp_path):
    local = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 50.0, "price": 2.0},
    ])
    # Every prefix maps to an empty list of local codes, so
    # build_combined_rows returns only the all-empty, all-object-dtype
    # fallback frame.
    combined = build_combined_rows(local, {"FR": [], "XX": []})

    result = build_prices_csv_rows(local, combined)

    assert pd.api.types.is_datetime64_any_dtype(result["date"])
