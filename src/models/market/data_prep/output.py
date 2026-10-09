"""Writes the final prices.csv (shared schema) and size_ratios.csv
(market-model-specific) output files."""
import pandas as pd

from .. import consts

PRICES_CSV_COLUMNS = [
    "date", "market_code", "species_code", "category_code",
    "landings_kg", "price", "currency", "measurement_unit", "size",
]


def build_prices_csv_rows(*row_frames) -> pd.DataFrame:
    """Concatenate local + combined (blended + per-size) rows and add the
    constant currency/measurement_unit columns expected by the shared
    schema.

    Empty input frames (e.g. the all-empty fallback `build_combined_rows`
    returns when every combined prefix has zero local children) are
    skipped before concatenation: an empty, all-object-dtype frame would
    otherwise downcast a real frame's `date` column from datetime64 to
    object once concatenated, corrupting every row's formatted date, not
    just the empty ones."""
    non_empty_frames = [frame for frame in row_frames if not frame.empty]
    all_rows = pd.concat(non_empty_frames if non_empty_frames else row_frames, ignore_index=True)
    all_rows["currency"] = "EUR"
    all_rows["measurement_unit"] = consts.SUPPORTED_MEASUREMENT_UNIT
    return all_rows[PRICES_CSV_COLUMNS]


def write_prices_csv(all_rows: pd.DataFrame, path) -> None:
    # Sort on `size` too (blended rows last within each group, via
    # na_position="last", since the blended row's size is NaN) with a
    # stable sort so ordering is deterministic run-to-run. Elsa's
    # load_local_market_price_history reads this file row by row and does
    # last-write-wins on (species_code, category_code, date); if a
    # per-size row landed after the blended row for the same group, Elsa
    # would silently pick up an arbitrary per-size price instead of the
    # intended blended price.
    all_rows.sort_values(
        ["date", "market_code", "species_code", "category_code", "size"],
        na_position="last",
        kind="stable",
    ).to_csv(path, index=False)


def write_size_ratios_csv(ratios: pd.DataFrame, path) -> None:
    ratios.sort_values(["market_code", "species_code", "size"]).to_csv(path, index=False)
