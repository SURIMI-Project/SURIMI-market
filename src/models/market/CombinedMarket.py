"""
One instance per 2-letter market code. Holds the frozen seasonal-baseline
+ volume-response model, keyed by (species_code, category_code). Stays
passive -- never holds or queries LocalMarket references, only receives
plain numbers.
"""
import logging

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from . import consts
from .CatchAccumulator import CatchAccumulator
from .PriceModel import PriceModel
from .TrendCorrection import TrendCorrection
from .LevelOnlyModel import LevelOnlyModel
from .LevelVolumeModel import LevelVolumeModel
from .level_volume_model import (
    fit_level_volume_model,
    predict_level_volume,
    fit_trend_rate_model,
    score_mape,
    compute_volume_baseline_stats,
    apply_volume_baseline_update,
    populate_price_time_series,
)
from .trend_confidence import compute_trend_confidence
from .window_selection import build_rolling_splits, select_best_window


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

        # (species_code, category_code) -> {month: (running_sum, running_count)}
        self.volume_baseline_stats: dict[tuple[str, str], dict[int, tuple[float, int]]] = {}

        # (species_code, category_code, date) -> real recorded price
        self.price_time_series: dict[tuple[str, str, pd.Timestamp], float] = {}

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
                    model = LevelVolumeModel.fit(
                        monthly, train_start, latest_date, self._fit_trend(monthly, latest_date)
                    )
                except (ValueError, np.linalg.LinAlgError):
                    # Rolling-origin validation scored this window acceptably, but the
                    # final window (train_start..latest_date) can still fail to fit if
                    # sparse real data makes the volume-anomaly series degenerate (zero
                    # variance) -- fall through to the same fallback used when no window
                    # was selected at all.
                    self.logger.warning(
                        f"CombinedMarket {self.market_code}: selected window {label} for {key} "
                        f"scored acceptably in rolling-origin validation but failed to re-fit on "
                        f"the final window ending at {latest_date}; falling back to level-only fit"
                    )
                    model = None

                if model is not None:
                    self.models[key] = model
                    self.window_labels[key] = label
                    self._seed_volume_baseline_stats(key, monthly, train_start, latest_date)
                    selected[key] = label
                    continue

            # Fallback: try a level-only fit.
            else:
                self.logger.warning(
                    f"CombinedMarket {self.market_code}: no stable level-volume window found "
                    f"for {key}, falling back to level-only fit"
                )
            try:
                model = LevelOnlyModel.fit(monthly, None, latest_date, self._fit_trend(monthly, latest_date))
                self.models[key] = model
                self.window_labels[key] = "level-only-fallback"
                selected[key] = "level-only-fallback"
            except ValueError:
                self.logger.warning(
                    f"CombinedMarket {self.market_code}: no usable price history at all "
                    f"for {key}, excluding from output"
                )
                selected[key] = None

        return selected

    def _fit_trend(self, monthly: pd.DataFrame, train_end: pd.Timestamp) -> TrendCorrection:
        trend_rate_params = fit_trend_rate_model(monthly, train_end)
        return TrendCorrection(
            confidence=compute_trend_confidence(monthly),
            rate=trend_rate_params["rate"],
            years_trained=trend_rate_params["years_trained"],
            train_end=train_end,
        )

    def _seed_volume_baseline_stats(self, key, monthly, train_start, train_end):
        self.volume_baseline_stats[key] = compute_volume_baseline_stats(monthly, train_start, train_end)

    def update_volume_baseline(self, species_code: str, category_code: str, date: pd.Timestamp, volume: float):
        """Incremental, never-decaying running mean."""
        key = (species_code, category_code)
        result = apply_volume_baseline_update(self.volume_baseline_stats, key, date, volume)
        if result is None:
            return
        month, new_mean = result
        if key in self.models:
            self.models[key].update_baseline_volume(month, new_mean)

    def load_price_time_series(self, prices_df: pd.DataFrame):
        """Load real recorded prices for this market_code."""
        populate_price_time_series(self.price_time_series, prices_df, self.market_code)

    def get_price(
        self, species_code: str, category_code: str, date: pd.Timestamp, volume: float
    ) -> float | None:
        """Real-price-first: check price_time_series, else predict."""
        key = (species_code, category_code)
        real_price = self.price_time_series.get((species_code, category_code, date))
        if pd.notna(real_price):
            return real_price

        if key not in self.models:
            return None

        return self.models[key].predict(date, volume)

    def get_live_anchor_series(
        self, species_code: str, category_code: str, own_monthly: pd.DataFrame, target_dates
    ) -> pd.Series | None:
        """
        Alternate anchor scheme: this market's own frozen coefficients
        applied to its own real historical volume at each date, varying
        month-to-month (unlike get_baseline_price_table's fixed 12-month
        seasonal cycle). Analysis-only -- never used by the live
        production path.

        own_monthly is this CombinedMarket's own (date, price,
        landings_kg) series, retained by the caller since CombinedMarket
        doesn't keep it after fit() returns. Returns None if there's no
        frozen model for the key; otherwise reindexed to target_dates,
        with NaN outside own_monthly's own range.
        """
        key = (species_code, category_code)
        if key not in self.models:
            return None

        predicted = self.models[key].predict_series(own_monthly).copy()
        predicted.index = own_monthly["date"]
        return predicted.reindex(pd.DatetimeIndex(target_dates))

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
