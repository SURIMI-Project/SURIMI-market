"""Aggregates local rows into combined (national) rows, matching the live
model's own runtime aggregation rule."""
import logging

import pandas as pd

from .aggregate import GROUP_KEYS

logger = logging.getLogger(__name__)


def build_combined_rows(local_rows: pd.DataFrame, prefix_to_locals: dict) -> pd.DataFrame:
    """For each combined prefix (e.g. "ES", "FR"), sum landings_kg and the
    implied value (price * landings_kg) across exactly its local ports --
    for both the blended (size=NaN) and per-size rows -- deriving price
    from the summed totals, never averaging already-computed per-port
    prices directly."""
    rows = []
    for prefix, local_codes in prefix_to_locals.items():
        subset = local_rows[local_rows["market_code"].isin(local_codes)].copy()
        if subset.empty:
            logger.warning(
                "Combined prefix %r has zero matching local market rows; "
                "no combined row will be produced for it.",
                prefix,
            )
            continue
        subset["value_eur"] = subset["price"] * subset["landings_kg"]

        group_keys = ["date", "species_code", "category_code", "size"]
        grouped = subset.groupby(group_keys, as_index=False, dropna=False).agg(
            landings_kg=("landings_kg", "sum"),
            value_eur=("value_eur", "sum"),
        )
        grouped["price"] = grouped["value_eur"] / grouped["landings_kg"]
        grouped["market_code"] = prefix
        rows.append(grouped.drop(columns="value_eur"))

    if not rows:
        return pd.DataFrame(columns=GROUP_KEYS + ["size", "landings_kg", "price"])

    return pd.concat(rows, ignore_index=True)[GROUP_KEYS + ["size", "landings_kg", "price"]]
