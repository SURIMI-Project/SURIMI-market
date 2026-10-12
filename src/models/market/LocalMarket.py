"""
One instance per 5-letter LOCODE. Holds a frozen base model (own-history
only, no neighbor signal) and a frozen neighbor-informed model
(StackedModel: anchor + the port's own volume response + a distance-
weighted neighbor signal, log-space, no switch), plus the per-port catch
accumulator reused unchanged from Elsa's LocalMarket.
"""
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta
import logging

import numpy as np
import pandas as pd

from . import consts
from .market_trackers import CatchAccumulator, RealPriceLookup, VolumeBaselineTracker, update_tracker_and_model
from .price_models import PriceModel, TrendCorrection
from .StackedModel import StackedModel
from .window_selection import fit_price_model_with_window_selection


@dataclass
class NeighborFitInputs:
    """Bundles the neighbor-informed model's fit-time handoff (previously
    four loose positional arguments) into one named object."""
    port_monthly: pd.DataFrame
    anchor_series: pd.Series
    neighbor_signal: pd.Series
    latest_date: pd.Timestamp


class LocalMarket:
    def __init__(self, market_code: str, name: str = None, lon: float = None, lat: float = None):
        self.market_code = market_code
        self.market_name = name
        self.lon = lon
        self.lat = lat

        self.logger = logging.getLogger(consts.LOCAL_MARKET_LOGGER_NAME)

        # Base model state (own-history only, no neighbor signal -- same
        # model CombinedMarket uses)
        self.models: dict[tuple[str, str], PriceModel] = {}
        self.window_labels: dict[tuple[str, str], str] = {}
        self._base_predicted_series: dict[tuple[str, str], pd.Series] = {}

        # Neighbor-informed model state
        self.neighbor_models: dict[tuple[str, str], StackedModel] = {}
        self.neighbor_window_labels: dict[tuple[str, str], str] = {}

        self.volume_baseline: dict[tuple[str, str], VolumeBaselineTracker] = {}
        self.real_prices = RealPriceLookup()

        self._monthly_catch = CatchAccumulator()
        self.price_history = []  # [(tick, {"prices": {...}, "catches": {...}})]

    # --- Catch accumulation (reused unchanged from Elsa's LocalMarket) ---

    def add_monthly_catch(self, species_code, category_code, quantity):
        self._monthly_catch.add(species_code, category_code, quantity)

    def clear_monthly_catch(self):
        self._monthly_catch.clear()

    def get_monthly_catch_snapshot(self):
        return self._monthly_catch.snapshot()

    def take_monthly_catch(self):
        """Atomically snapshot and clear this tick's accumulated catch --
        see CatchAccumulator.take()."""
        return self._monthly_catch.take()

    # --- Base model fitting ---

    def fit_base_models(
        self, monthly_by_key: dict[tuple[str, str], pd.DataFrame], latest_date: pd.Timestamp
    ) -> dict[tuple[str, str], str | None]:
        # LocalMarket's own-history models never fit a trend (see
        # ARCHITECTURE.md section 6's deferred validation item) -- a single
        # identity() instance is constructed once here (it depends only on
        # latest_date, not on any per-key data) and reused for every key and
        # every branch below, instead of reconstructing it per call.
        trend = TrendCorrection.identity(latest_date)

        def on_model_fitted(key, model, monthly):
            self._base_predicted_series[key] = self._date_indexed(model.predict_series(monthly), monthly)

        return fit_price_model_with_window_selection(
            monthly_by_key,
            latest_date,
            build_trend=lambda monthly, latest_date: trend,
            models=self.models,
            window_labels=self.window_labels,
            volume_baseline=self.volume_baseline,
            logger=self.logger,
            class_label="LocalMarket",
            market_code=self.market_code,
            window_noun="base-model window",
            exclusion_suffix=" (will fall back to CombinedMarket-only at predict time)",
            on_model_fitted=on_model_fitted,
        )

    @staticmethod
    def _date_indexed(series: pd.Series, monthly: pd.DataFrame) -> pd.Series:
        """Re-index a freshly-predicted series (which inherits monthly's positional
        RangeIndex) onto monthly["date"]. Needed because other ports' neighbor-signal
        code does `.map(date)` lookups that silently produce all-NaN against a
        RangeIndex."""
        series = series.copy()
        series.index = monthly["date"]
        return series

    def update_volume_baseline(self, species_code: str, category_code: str, date: pd.Timestamp, volume: float):
        update_tracker_and_model(
            self.volume_baseline, self.models, (species_code, category_code), date, volume, self.neighbor_models
        )

    def predict_from_base_model(
        self, species_code: str, category_code: str, date: pd.Timestamp, volume: float
    ) -> float | None:
        """Evaluate the frozen base model for a given tick; always computed,
        regardless of whether this port has real price data that tick."""
        key = (species_code, category_code)
        if key not in self.models:
            return None
        return self.models[key].predict(date, volume)

    def get_base_predicted_series(self, species_code: str, category_code: str) -> pd.Series | None:
        """Full historical predicted series, used by other ports' neighbor signals."""
        return self._base_predicted_series.get((species_code, category_code))

    # --- Real-price-first ---

    def load_price_time_series(self, prices_df: pd.DataFrame):
        self.real_prices.load(prices_df, self.market_code)

    def get_real_price(self, species_code: str, category_code: str, date: pd.Timestamp) -> float | None:
        return self.real_prices.get(species_code, category_code, date)

    def record_history(self, tick: int, prices: dict, catches: dict):
        self.price_history.append((tick, {"prices": dict(prices), "catches": dict(catches)}))

    # --- Neighbor-informed model fitting ---

    def fit_neighbor_models(self, species_code: str, category_code: str, inputs: NeighborFitInputs) -> str | None:
        """Reuses this port's own BASE MODEL window (no independent
        rolling-origin search) -- the window-length decision was already
        made one step earlier against the same underlying series by
        fit_base_models, and re-deriving it independently here (as the
        previous design did, at ~2x the cost since it had to fit two
        sub-models per candidate per split) was unlikely to land anywhere
        different. Falls back to no neighbor model at all if the base model
        itself used the level-only fallback (no well-defined window to
        reuse) or if the single re-fit on the reused window fails -- there's
        no retry-the-next-candidate loop anymore since there's only ever one
        candidate now."""
        key = (species_code, category_code)
        base_window_label = self.window_labels.get(key)
        if base_window_label is None or base_window_label == "level-only-fallback":
            self.logger.warning(
                f"LocalMarket {self.market_code}: no stable base-model window to reuse for {key}, "
                f"falling back to the base model only"
            )
            return None

        years = consts.WINDOW_CANDIDATE_YEARS[base_window_label]
        train_start = inputs.latest_date - relativedelta(years=years) if years is not None else None

        try:
            model = StackedModel.fit(
                inputs.port_monthly, inputs.anchor_series, inputs.neighbor_signal, train_start, inputs.latest_date
            )
        except (ValueError, np.linalg.LinAlgError):
            self.logger.warning(
                f"LocalMarket {self.market_code}: reused base-model window {base_window_label} for {key} "
                f"failed to fit a neighbor-informed model; falling back to the base model only"
            )
            return None

        self.neighbor_models[key] = model
        self.neighbor_window_labels[key] = base_window_label
        return base_window_label

    def predict_from_neighbor_model(
        self,
        species_code: str,
        category_code: str,
        date: pd.Timestamp,
        volume: float,
        anchor_value: float,
        neighbor_value: float,
    ) -> float | None:
        """Looks up the frozen StackedModel for this key and delegates."""
        key = (species_code, category_code)
        model = self.neighbor_models.get(key)
        if model is None:
            return None
        return model.predict(date, volume, anchor_value, neighbor_value)
