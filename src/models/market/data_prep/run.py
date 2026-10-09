"""Orchestrates the full EMODnet -> prices.csv / size_ratios.csv pipeline."""
import argparse
from pathlib import Path

import pandas as pd

from .aggregate import build_local_rows
from .combine import build_combined_rows
from .download import download_and_extract, get_current_zip_url
from .extract import filter_and_clean, load_fish_sales_data
from .output import build_prices_csv_rows, write_prices_csv, write_size_ratios_csv
from .size_ratios import compute_size_ratios
from ..market_topology import build_prefix_to_locals as _build_prefix_to_locals_from_df


def build_prefix_to_locals(markets_csv_path) -> dict:
    """Thin, path-based wrapper around market_topology.build_prefix_to_locals
    (shared with MarketSystem.py) -- kept here under this name/signature
    since this module's own tests and callers pass a CSV path, not an
    in-memory DataFrame."""
    return _build_prefix_to_locals_from_df(pd.read_csv(markets_csv_path))


def run(markets_csv_path, output_dir, gdb_path=None, dest_dir=None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if gdb_path is None:
        zip_url = get_current_zip_url()
        gdb_path = download_and_extract(zip_url, dest_dir or output_dir / "_download")

    prefix_to_locals = build_prefix_to_locals(markets_csv_path)
    all_local_codes = [code for codes in prefix_to_locals.values() for code in codes]

    raw = load_fish_sales_data(gdb_path)
    cleaned = filter_and_clean(raw, market_codes=all_local_codes)

    local_rows = build_local_rows(cleaned)
    combined_rows = build_combined_rows(local_rows, prefix_to_locals)

    prices_rows = build_prices_csv_rows(local_rows, combined_rows)
    write_prices_csv(prices_rows, output_dir / "prices.csv")

    ratios = compute_size_ratios(combined_rows)
    write_size_ratios_csv(ratios, output_dir / "size_ratios.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Build prices.csv and size_ratios.csv from EMODnet."
    )
    parser.add_argument("--markets-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--gdb-path", default=None,
        help="Skip download, use an already-extracted .gdb (e.g. for the "
             "pinned-snapshot consistency check).",
    )
    args = parser.parse_args()
    run(args.markets_csv, args.output_dir, gdb_path=args.gdb_path)


if __name__ == "__main__":
    main()
