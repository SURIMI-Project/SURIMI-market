"""Writes the final prices.csv (shared schema) and size_ratios.csv
(market-model-specific) output files."""
import pandas as pd

PRICES_CSV_COLUMNS = [
    "date", "market_code", "species_code", "category_code",
    "landings_kg", "price", "currency", "measurement_unit", "size",
]


def build_prices_csv_rows(*row_frames) -> pd.DataFrame:
    """Concatenate local + combined (blended + per-size) rows and add the
    constant currency/measurement_unit columns expected by the shared
    schema."""
    all_rows = pd.concat(row_frames, ignore_index=True)
    all_rows["currency"] = "EUR"
    all_rows["measurement_unit"] = "kg"
    return all_rows[PRICES_CSV_COLUMNS]


def write_prices_csv(all_rows: pd.DataFrame, path) -> None:
    all_rows.sort_values(["date", "market_code", "species_code", "category_code"]).to_csv(
        path, index=False
    )


def write_size_ratios_csv(ratios: pd.DataFrame, path) -> None:
    ratios.sort_values(["market_code", "species_code", "size"]).to_csv(path, index=False)
