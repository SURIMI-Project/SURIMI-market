"""Shared topology helper: grouping 5-letter local port codes under their
2-letter combined-market prefix. Used identically by MarketSystem (the live
simulation engine) and data_prep/run.py (the offline EMODnet-to-prices.csv
pipeline) -- extracted here so neither module depends on the other."""
import pandas as pd


def build_prefix_to_locals(markets_df: pd.DataFrame) -> dict[str, list[str]]:
    """Union of explicit 2-letter combined codes and prefixes implied by the
    5-letter local codes present: group markets_df's 5-letter LOCODEs by
    their 2-letter prefix, so a combined market always exists for any
    country with local ports under it, even if its own 2-letter row is
    missing."""
    codes = markets_df["market_code"].dropna().unique().tolist()
    local_codes = [c for c in codes if len(c) == 5]
    explicit_prefixes = {c for c in codes if len(c) == 2}
    implied_prefixes = {c[:2] for c in local_codes}
    prefixes = explicit_prefixes | implied_prefixes

    return {prefix: [c for c in local_codes if c.startswith(prefix)] for prefix in prefixes}
