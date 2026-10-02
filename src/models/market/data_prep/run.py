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


def build_prefix_to_locals(markets_csv_path) -> dict:
    """Group markets.csv's 5-letter LOCODEs by their 2-letter prefix. The
    set of needed prefixes is the union of any 2-letter codes explicitly
    listed and any prefixes implied by the 5-letter codes present, so a
    combined market always exists for any country with local ports under
    it, even if its own 2-letter row is missing."""
    markets = pd.read_csv(markets_csv_path)
    codes = markets["market_code"].dropna().unique().tolist()
    local_codes = [c for c in codes if len(c) == 5]
    explicit_prefixes = {c for c in codes if len(c) == 2}
    implied_prefixes = {c[:2] for c in local_codes}
    prefixes = explicit_prefixes | implied_prefixes

    return {
        prefix: [c for c in local_codes if c.startswith(prefix)]
        for prefix in prefixes
    }


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
