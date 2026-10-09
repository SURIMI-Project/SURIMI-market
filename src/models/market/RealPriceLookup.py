"""
Real recorded prices for one market_code, keyed by (species_code,
category_code, date). Wraps level_volume_model.populate_price_time_series,
which stays the free function doing the actual CSV-row filtering; this
class just gives the per-market lookup table a home instead of a raw dict
CombinedMarket and LocalMarket each built/read identically.
"""
import pandas as pd

from .level_volume_model import populate_price_time_series


class RealPriceLookup:
    def __init__(self):
        self._prices: dict[tuple[str, str, pd.Timestamp], float] = {}

    def load(self, prices_df: pd.DataFrame, market_code: str) -> None:
        populate_price_time_series(self._prices, prices_df, market_code)

    def get(self, species_code: str, category_code: str, date: pd.Timestamp) -> float | None:
        return self._prices.get((species_code, category_code, date))
