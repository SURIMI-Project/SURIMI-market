"""
One instance per 2-letter market code. Holds the frozen seasonal-baseline
+ volume-response model, keyed by (species_code, category_code). Stays
passive -- never holds or queries LocalMarket references, only receives
plain numbers.
"""
import logging

import pandas as pd

from . import consts
from .CatchAccumulator import CatchAccumulator
from .price_models import PriceModel, TrendCorrection
from .VolumeBaselineTracker import VolumeBaselineTracker, update_tracker_and_model
from .RealPriceLookup import RealPriceLookup
from .level_volume_model import fit_trend_rate_model
from .trend_confidence import compute_trend_confidence
from .window_selection import fit_price_model_with_window_selection


class CombinedMarket:
    def __init__(self, market_code: str):
        self.market_code = market_code
        self.logger = logging.getLogger(consts.COMBINED_MARKET_LOGGER_NAME)

        # Catch accumulator for sales routed straight to this 2-letter
        # prefix; same shape as LocalMarket's own accumulator.
        self._direct_catch = CatchAccumulator()

        # (species_code, category_code) -> frozen PriceModel (a
        # LevelVolumeModel, or a LevelOnlyModel on the no-volume fallback path)
        self.models: dict[tuple[str, str], PriceModel] = {}
        self.window_labels: dict[tuple[str, str], str] = {}

        # (species_code, category_code) -> VolumeBaselineTracker
        self.volume_baseline: dict[tuple[str, str], VolumeBaselineTracker] = {}

        self.real_prices = RealPriceLookup()

        self.price_history = []  # [(tick, {"prices": {...}, "volumes": {...}})]

    def fit(
        self, monthly_by_key: dict[tuple[str, str], pd.DataFrame], latest_date: pd.Timestamp
    ) -> dict[tuple[str, str], str | None]:
        """
        Fit and freeze the model for every (species_code, category_code),
        selecting the best training-window length independently per key.
        Falls back to a level-only fit if no window produces a stable
        level-volume model; excludes the key entirely if even that fails.

        Window selection and the frozen baseline are unconditional --
        identical regardless of trend. Trend confidence/rate are computed
        separately via _fit_trend for every key that gets a model, so
        get_price() can apply the trend correction unconditionally (a
        no-op ~1.0 multiplier for a non-trending key).
        """
        return fit_price_model_with_window_selection(
            monthly_by_key,
            latest_date,
            build_trend=self._fit_trend,
            models=self.models,
            window_labels=self.window_labels,
            volume_baseline=self.volume_baseline,
            logger=self.logger,
            class_label="CombinedMarket",
            market_code=self.market_code,
            window_noun="level-volume window",
        )

    def _fit_trend(self, monthly: pd.DataFrame, train_end: pd.Timestamp) -> TrendCorrection:
        trend_rate_params = fit_trend_rate_model(monthly, train_end)
        return TrendCorrection(
            confidence=compute_trend_confidence(monthly, train_end),
            rate=trend_rate_params["rate"],
            years_trained=trend_rate_params["years_trained"],
            train_end=train_end,
        )

    def update_volume_baseline(self, species_code: str, category_code: str, date: pd.Timestamp, volume: float):
        """Incremental, never-decaying running mean."""
        update_tracker_and_model(self.volume_baseline, self.models, (species_code, category_code), date, volume)

    def load_price_time_series(self, prices_df: pd.DataFrame):
        """Load real recorded prices for this market_code."""
        self.real_prices.load(prices_df, self.market_code)

    def get_price(
        self, species_code: str, category_code: str, date: pd.Timestamp, volume: float
    ) -> float | None:
        """Real-price-first: check real_prices, else predict."""
        key = (species_code, category_code)
        real_price = self.real_prices.get(species_code, category_code, date)
        if pd.notna(real_price):
            return real_price

        if key not in self.models:
            return None

        return self.models[key].predict(date, volume)

    def get_baseline_price_table(self, species_code: str, category_code: str) -> pd.Series | None:
        """
        Frozen per-month seasonal baseline, used as LocalMarket's static
        anchor. Always reindexed to the full 1..12 month range --
        baseline_price (e.g. on the level-only fallback path) may not
        cover every month, so missing months come back as NaN rather than
        being dropped or fabricated.
        """
        key = (species_code, category_code)
        if key not in self.models:
            return None
        return self.models[key].baseline_price.reindex(range(1, 13))

    def record_history(self, tick: int, prices: dict, volumes: dict):
        self.price_history.append((tick, {"prices": dict(prices), "volumes": dict(volumes)}))

    def add_direct_catch(self, species_code: str, category_code: str, quantity: float):
        self._direct_catch.add(species_code, category_code, quantity)

    def clear_direct_catch(self):
        self._direct_catch.clear()

    def get_direct_catch_snapshot(self) -> dict:
        return self._direct_catch.snapshot()
