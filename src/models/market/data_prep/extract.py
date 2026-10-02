"""Extracts and filters raw EMODnet First Sales data."""
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio


def find_data_layer(gdb_path) -> str:
    """Find the First Sales data layer name within the geodatabase. The
    layer name is dated (e.g. "EMODNET_HA_Fisheries_FirstSales_Data_20250319"),
    so this searches by pattern rather than hardcoding a date."""
    layers = pyogrio.list_layers(gdb_path)
    layer_names = [name for name, _ in layers]
    matches = [name for name in layer_names if "FirstSales_Data" in name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one First Sales data layer, found {len(matches)}: {matches}"
        )
    return matches[0]


def load_fish_sales_data(gdb_path) -> pd.DataFrame:
    """Load the First Sales data layer as a plain DataFrame, with a Date
    column derived from Year + Month_N."""
    layer_name = find_data_layer(gdb_path)
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    df["Date"] = pd.to_datetime(
        df[["Year", "Month_N"]].rename(columns={"Year": "year", "Month_N": "month"}).assign(day=1)
    )
    return df


def filter_and_clean(df: pd.DataFrame, market_codes: list) -> pd.DataFrame:
    """Restrict to the given market codes and Fresh+Whole sales only, and
    construct category_code from Preservation + Presentation."""
    filtered = df[
        (df["Locode"].isin(market_codes))
        & (df["Preservation"] == "Fresh")
        & (df["Presentation"] == "Whole")
    ].copy()

    filtered["category_code"] = filtered["Preservation"] + " - " + filtered["Presentation"]

    # Pin the size column's dtype regardless of what dtype the raw
    # geodatabase column arrives as (today it's already strings like "1",
    # "3a", but nothing enforces that). An integer dtype here would later
    # produce a `size` column of float64 in build_local_rows, writing
    # "1.0" instead of "1" to prices.csv and silently breaking the join
    # key size_ratios.csv uses to match per-size rows back to prices.csv.
    filtered["Size"] = filtered["Size"].astype("string")

    return filtered[
        [
            "Date", "Locode", "ERS_Code", "category_code",
            "Value_EUR", "Volume_Kg", "Price_EUR_kg", "Size",
        ]
    ].rename(columns={"Date": "date", "Locode": "market_code", "ERS_Code": "species_code"})
