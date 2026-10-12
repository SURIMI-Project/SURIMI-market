"""Computes the per-(market, species, size) historical price ratio used
to convert a blended prediction into a size-specific quote."""
import pandas as pd


def compute_size_ratios(combined_rows: pd.DataFrame) -> pd.DataFrame:
    blended = combined_rows[combined_rows["size"].isna()][
        ["date", "market_code", "species_code", "category_code", "price"]
    ].rename(columns={"price": "blended_price"})

    per_size = combined_rows[combined_rows["size"].notna()]

    merged = per_size.merge(
        blended, on=["date", "market_code", "species_code", "category_code"], how="inner"
    )
    merged["ratio"] = merged["price"] / merged["blended_price"]

    return (
        merged.groupby(["market_code", "species_code", "size"], as_index=False)["ratio"]
        .mean()
    )
