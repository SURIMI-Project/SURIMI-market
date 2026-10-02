"""Aggregates cleaned per-transaction rows into blended and per-size
prices.csv rows."""
import numpy as np
import pandas as pd

GROUP_KEYS = ["date", "market_code", "species_code", "category_code"]


def _summarise(df: pd.DataFrame, group_keys: list) -> pd.DataFrame:
    grouped = df.groupby(group_keys, as_index=False).agg(
        landings_kg=("Volume_Kg", "sum"),
        value_eur=("Value_EUR", "sum"),
    )
    grouped["price"] = grouped["value_eur"] / grouped["landings_kg"]
    return grouped.drop(columns="value_eur")


def build_local_rows(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build both the blended (size=NaN, summed across all size grades)
    and the per-size rows for each (date, market_code, species_code,
    category_code) group."""
    blended = _summarise(cleaned, GROUP_KEYS)
    blended["size"] = np.nan

    with_size = cleaned.dropna(subset=["Size"])
    per_size = _summarise(with_size, GROUP_KEYS + ["Size"])
    per_size = per_size.rename(columns={"Size": "size"})

    combined = pd.concat([blended, per_size], ignore_index=True)
    return combined[GROUP_KEYS + ["size", "landings_kg", "price"]]
