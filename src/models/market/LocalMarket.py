"""
One instance per 5-letter LOCODE. Holds a frozen base model (own-history
only, no neighbor signal) and a frozen neighbor-informed model
(Stacked/DistanceOnly, picked dynamically per tick), plus the per-port catch
accumulator reused unchanged from Elsa's LocalMarket.
"""
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta
import logging

import numpy as np
import pandas as pd

from . import consts
from .CatchAccumulator import CatchAccumulator
from .PriceModel import PriceModel
from .TrendCorrection import TrendCorrection
from .LevelOnlyModel import LevelOnlyModel
from .LevelVolumeModel import LevelVolumeModel
from .VolumeBaselineTracker import VolumeBaselineTracker
from .RealPriceLookup import RealPriceLookup
from .NeighborInformedModel import NeighborInformedModel
from .StackedModel import StackedModel
from .DistanceOnlyModel import DistanceOnlyModel
from .level_volume_model import (
    fit_level_volume_model,
    predict_level_volume,
    score_mape,
    with_month_column,
)
from .stacked_model import (
    fit_stacked_model,
    predict_stacked_model,
    fit_distance_only_model,
    predict_distance_only_model,
    should_use_distance_only,
)
from .window_selection import build_rolling_splits, select_best_window, rank_windows_by_score


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
        self.neighbor_models: dict[tuple[str, str], NeighborInformedModel] = {}
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

    # --- Base model fitting ---

    def fit_base_models(
        self, monthly_by_key: dict[tuple[str, str], pd.DataFrame], latest_date: pd.Timestamp
    ) -> dict[tuple[str, str], str | None]:
        selected: dict[tuple[str, str], str | None] = {}

        for key, monthly in monthly_by_key.items():
            rolling_splits = build_rolling_splits(latest_date)

            def fit_fn(train_start, train_end, _monthly=monthly):
                return fit_level_volume_model(_monthly, train_start, train_end)

            def score_fn(params, test_start, test_end, _monthly=monthly):
                predicted = predict_level_volume(_monthly, params)
                mask = (_monthly["date"] >= test_start) & (_monthly["date"] < test_end)
                return score_mape(predicted[mask], _monthly["price"][mask])

            result = select_best_window(consts.WINDOW_CANDIDATE_YEARS, rolling_splits, fit_fn, score_fn)

            if result is not None:
                label, years = result
                train_start = latest_date - relativedelta(years=years) if years is not None else None
                try:
                    trend = TrendCorrection.identity(latest_date)
                    model = LevelVolumeModel.fit(monthly, train_start, latest_date, trend)
                    self.models[key] = model
                    self.window_labels[key] = label
                    self._base_predicted_series[key] = self._date_indexed(
                        model.predict_series(monthly), monthly
                    )
                    self._seed_volume_baseline_stats(key, monthly, train_start, latest_date)
                    selected[key] = label
                    continue
                except (ValueError, np.linalg.LinAlgError):
                    # Mirrors CombinedMarket.fit's guard: a scored-good window can still
                    # fail its final re-fit if sparse volume data yields a zero-variance
                    # series there. Fall through to the same fallback as "no window selected".
                    self.logger.warning(
                        f"LocalMarket {self.market_code}: selected window {label} for {key} "
                        f"scored acceptably in rolling-origin validation but failed to re-fit on "
                        f"the final window ending at {latest_date}; falling back to level-only fit"
                    )

            else:
                self.logger.warning(
                    f"LocalMarket {self.market_code}: no stable base-model window found for {key}, "
                    f"falling back to level-only fit"
                )
            try:
                trend = TrendCorrection.identity(latest_date)
                model = LevelOnlyModel.fit(monthly, None, latest_date, trend)
                self.models[key] = model
                self.window_labels[key] = "level-only-fallback"
                self._base_predicted_series[key] = self._date_indexed(
                    model.predict_series(monthly), monthly
                )
                selected[key] = "level-only-fallback"
            except ValueError:
                self.logger.warning(
                    f"LocalMarket {self.market_code}: no usable price history at all for {key}, "
                    f"excluding from output (will fall back to CombinedMarket-only at predict time)"
                )
                selected[key] = None

        return selected

    def _seed_volume_baseline_stats(self, key, monthly, train_start, train_end):
        tracker = VolumeBaselineTracker()
        tracker.seed(monthly, train_start, train_end)
        self.volume_baseline[key] = tracker

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
        key = (species_code, category_code)
        tracker = self.volume_baseline.get(key)
        if tracker is None:
            return
        result = tracker.update(date, volume)
        if result is None:
            return
        month, new_mean = result
        if key in self.models:
            self.models[key].update_baseline_volume(month, new_mean)

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
        key = (species_code, category_code)
        rolling_splits = build_rolling_splits(inputs.latest_date)

        def fit_fn(train_start, train_end):
            stacked = fit_stacked_model(
                inputs.port_monthly, inputs.anchor_series, inputs.neighbor_signal, train_start, train_end
            )
            distance_only = fit_distance_only_model(
                inputs.port_monthly, inputs.anchor_series, inputs.neighbor_signal, train_start, train_end
            )
            trial_volume = self._trial_baseline_volume(inputs.port_monthly, train_start, train_end)
            return {"stacked": stacked, "distance_only": distance_only, "trial_volume": trial_volume}

        def score_fn(params, test_start, test_end):
            blended = self._blend_neighbor_candidate_predictions(
                inputs.port_monthly, inputs.anchor_series, inputs.neighbor_signal, params
            )
            mask = (inputs.port_monthly["date"] >= test_start) & (inputs.port_monthly["date"] < test_end)
            return score_mape(blended[mask], inputs.port_monthly["price"][mask])

        def sanity_check_fn(params):
            """Reject a candidate whose blended prediction goes non-positive anywhere in
            the real history. Checks the blended value, not each sub-model independently,
            since only one of Stacked/DistanceOnly is ever selected per tick -- a bad value
            in the unselected one would never be reached."""
            blended = self._blend_neighbor_candidate_predictions(
                inputs.port_monthly, inputs.anchor_series, inputs.neighbor_signal, params
            )
            valid = blended.notna()
            blended_min = float(blended[valid].min()) if valid.any() else None
            return blended_min is None or blended_min > 0

        ranked = rank_windows_by_score(
            consts.WINDOW_CANDIDATE_YEARS, rolling_splits, fit_fn, score_fn, sanity_check_fn
        )

        if not ranked:
            self.logger.warning(
                f"LocalMarket {self.market_code}: no stable neighbor-informed window found for {key}, "
                f"falling back to the base model only"
            )
            return None

        # Rolling-origin success doesn't guarantee the final re-fit succeeds -- the
        # final window (never scored directly) can land differently, e.g. inside a
        # real landings gap. Try each ranked candidate in order; a fittable, sane,
        # lower-scored window beats falling back to the base model only.
        for label, years in ranked:
            train_start = inputs.latest_date - relativedelta(years=years) if years is not None else None
            try:
                final_params = fit_fn(train_start, inputs.latest_date)
            except (ValueError, np.linalg.LinAlgError):
                self.logger.warning(
                    f"LocalMarket {self.market_code}: selected neighbor-informed window {label} for {key} "
                    f"scored acceptably in rolling-origin validation but failed to re-fit on the "
                    f"final window ending at {inputs.latest_date}; trying the next-best-scoring window"
                )
                continue
            if not sanity_check_fn(final_params):
                self.logger.warning(
                    f"LocalMarket {self.market_code}: selected neighbor-informed window {label} for {key} "
                    f"scored acceptably in rolling-origin validation but its final re-fit would "
                    f"predict a non-positive price somewhere in the real history available; "
                    f"trying the next-best-scoring window"
                )
                continue
            self.neighbor_models[key] = NeighborInformedModel(
                stacked=StackedModel(final_params["stacked"]),
                distance_only=DistanceOnlyModel(final_params["distance_only"]),
                volume_baseline=self.volume_baseline.get(key),
            )
            self.neighbor_window_labels[key] = label
            return label

        self.logger.warning(
            f"LocalMarket {self.market_code}: every candidate neighbor-informed window for {key} scored "
            f"acceptably in rolling-origin validation but failed to re-fit or failed the sanity "
            f"check on the final window ending at {inputs.latest_date}; falling back to the base model only"
        )
        return None

    def _trial_baseline_volume(self, port_monthly, train_start, train_end):
        port = with_month_column(port_monthly)
        train = port[port["date"] < train_end]
        if train_start is not None:
            train = train[train["date"] >= train_start]
        return train.groupby("month")["landings_kg"].mean()

    def _blend_neighbor_candidate_predictions(self, port_monthly, anchor_series, neighbor_signal, params) -> pd.Series:
        """Dynamic-switch-blended prediction used only during fit_neighbor_models'
        rolling-origin scoring."""
        stacked_pred = predict_stacked_model(port_monthly, anchor_series, neighbor_signal, params["stacked"])
        distance_pred = predict_distance_only_model(
            port_monthly, anchor_series, neighbor_signal, params["distance_only"]
        )

        port = with_month_column(port_monthly)
        baseline_volume = params["trial_volume"]
        volume_ratio = port["landings_kg"].fillna(0.0) / port["month"].map(baseline_volume)

        use_distance_only = should_use_distance_only(volume_ratio)
        blended = stacked_pred.where(~use_distance_only, distance_pred)
        blended.index = port_monthly.index
        return blended

    def predict_from_neighbor_model(
        self,
        species_code: str,
        category_code: str,
        date: pd.Timestamp,
        volume: float,
        anchor_value: float,
        neighbor_value: float,
    ) -> float | None:
        """Runtime dynamic switch, now owned by NeighborInformedModel itself --
        this method just looks up the right instance and delegates."""
        key = (species_code, category_code)
        model = self.neighbor_models.get(key)
        if model is None:
            return None
        return model.predict(date, volume, anchor_value, neighbor_value)
