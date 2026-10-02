"""
One instance per 5-letter LOCODE (spec section 3). Holds frozen Step 1
coefficients (this task) and Step 2 Stacked/DistanceOnly coefficients
(Task 9), plus the per-port catch accumulator reused unchanged from
Elsa's LocalMarket.
"""
from collections import defaultdict
from dateutil.relativedelta import relativedelta
import logging
import threading

import pandas as pd

from . import consts
from .level_volume_model import (
    fit_level_volume_model,
    predict_level_volume,
    fit_level_only_model,
    predict_level_only,
    score_mape,
)
from .stacked_model import (
    fit_stacked_model,
    predict_stacked_model,
    fit_distance_only_model,
    predict_distance_only_model,
)
from .window_selection import build_rolling_splits, select_best_window


class LocalMarket:
    def __init__(self, market_code: str, name: str = None, lon: float = None, lat: float = None):
        self.market_code = market_code
        self.market_name = name
        self.lon = lon
        self.lat = lat

        self.logger = logging.getLogger(consts.LOCAL_MARKET_LOGGER_NAME)

        # Step 1 state (this task)
        self.step1_models: dict[tuple[str, str], dict] = {}
        self.step1_window_labels: dict[tuple[str, str], str] = {}
        self._step1_historical_series: dict[tuple[str, str], pd.Series] = {}

        # Step 2 state (added in Task 9)
        self.step2_stacked_models: dict[tuple[str, str], dict] = {}
        self.step2_distance_only_models: dict[tuple[str, str], dict] = {}
        self.step2_window_labels: dict[tuple[str, str], str] = {}

        self.volume_baseline_stats: dict[tuple[str, str], dict[int, tuple[float, int]]] = {}
        self.price_time_series: dict[tuple[str, str, pd.Timestamp], float] = {}

        self._monthly_catch: dict[tuple[str, str], float] = defaultdict(float)
        self.price_history = []  # [(tick, {"prices": {...}, "catches": {...}})]
        self.lock = threading.Lock()

    # --- Catch accumulation (reused unchanged from Elsa's LocalMarket) ---

    def add_monthly_catch(self, species_code, category_code, quantity):
        with self.lock:
            self._monthly_catch[(species_code, category_code)] += quantity

    def clear_monthly_catch(self):
        with self.lock:
            self._monthly_catch.clear()

    def get_monthly_catch_snapshot(self):
        with self.lock:
            return dict(self._monthly_catch)

    # --- Step 1 fitting ---

    def fit_step1(
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
                params = fit_level_volume_model(monthly, train_start, latest_date)
                self.step1_models[key] = params
                self.step1_window_labels[key] = label
                self._step1_historical_series[key] = self._date_indexed(
                    predict_level_volume(monthly, params), monthly
                )
                self._seed_volume_baseline_stats(key, monthly, train_start, latest_date)
                selected[key] = label
                continue

            self.logger.warning(
                f"LocalMarket {self.market_code}: no stable Step-1 window found for {key}, "
                f"falling back to level-only fit"
            )
            try:
                params = fit_level_only_model(monthly, None, latest_date)
                self.step1_models[key] = params
                self.step1_window_labels[key] = "level-only-fallback"
                self._step1_historical_series[key] = self._date_indexed(
                    predict_level_only(monthly, params), monthly
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
        monthly = monthly.copy()
        monthly["month"] = monthly["date"].dt.month
        train = monthly[monthly["date"] < train_end]
        if train_start is not None:
            train = train[train["date"] >= train_start]

        stats: dict[int, tuple[float, int]] = {}
        for month, group in train.groupby("month"):
            valid = group["landings_kg"].dropna()
            if len(valid) > 0:
                stats[month] = (float(valid.sum()), int(len(valid)))
        self.volume_baseline_stats[key] = stats

    @staticmethod
    def _date_indexed(series: pd.Series, monthly: pd.DataFrame) -> pd.Series:
        """Re-index a freshly-predicted series (which inherits `monthly`'s plain
        positional RangeIndex -- `monthly` comes from build_complete_monthly_series,
        whose last step is a reset_index()) onto `monthly["date"]` instead. Only
        the copy stored in _step1_historical_series needs this: that series is the
        one OTHER ports' neighbor signals (Task 10-12) align onto real calendar
        dates via distance_weighted_neighbor_signal/build_static_anchor_series's
        `.map(date)` lookups, which silently produce all-NaN against a RangeIndex."""
        series = series.copy()
        series.index = monthly["date"]
        return series

    def update_volume_baseline(self, species_code: str, category_code: str, date: pd.Timestamp, volume: float):
        key = (species_code, category_code)
        if key not in self.volume_baseline_stats or pd.isna(volume):
            return

        month = date.month
        prev_sum, prev_count = self.volume_baseline_stats[key].get(month, (0.0, 0))
        new_sum, new_count = prev_sum + volume, prev_count + 1
        self.volume_baseline_stats[key][month] = (new_sum, new_count)

        if key in self.step1_models and self.step1_models[key]["baseline_volume"] is not None:
            self.step1_models[key]["baseline_volume"][month] = new_sum / new_count

    def predict_step1(
        self, species_code: str, category_code: str, date: pd.Timestamp, volume: float
    ) -> float | None:
        """Evaluate the frozen Step-1 model for a given tick (plan header decision 6: always
        computed, regardless of whether this port itself has real data that tick)."""
        key = (species_code, category_code)
        if key not in self.step1_models:
            return None

        params = self.step1_models[key]
        monthly = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
        if params.get("baseline_volume") is None:
            predicted = predict_level_only(monthly, params)
        else:
            predicted = predict_level_volume(monthly, params)
        value = predicted.iloc[0]
        return float(value) if pd.notna(value) else None

    def get_step1_historical_series(self, species_code: str, category_code: str) -> pd.Series | None:
        """Full historical predicted series -- used by OTHER ports' neighbor signals (Task 10-12)."""
        return self._step1_historical_series.get((species_code, category_code))

    # --- Real-price-first (spec section 8) ---

    def load_price_time_series(self, prices_df: pd.DataFrame):
        rows = prices_df[(prices_df["market_code"] == self.market_code) & (prices_df["size"].isna())]
        for row in rows.itertuples(index=False):
            self.price_time_series[(row.species_code, row.category_code, row.date)] = row.price

    def get_real_price(self, species_code: str, category_code: str, date: pd.Timestamp) -> float | None:
        return self.price_time_series.get((species_code, category_code, date))

    def record_history(self, tick: int, prices: dict, catches: dict):
        self.price_history.append((tick, {"prices": dict(prices), "catches": dict(catches)}))

    # --- Step 2 fitting (spec section 3/5; plan header decisions 3, 8, 9) ---

    def fit_step2(
        self,
        species_code: str,
        category_code: str,
        port_monthly: pd.DataFrame,
        anchor_series: pd.Series,
        neighbor_signal: pd.Series,
        latest_date: pd.Timestamp,
    ) -> str | None:
        key = (species_code, category_code)
        rolling_splits = build_rolling_splits(latest_date)

        def fit_fn(train_start, train_end):
            stacked = fit_stacked_model(port_monthly, anchor_series, neighbor_signal, train_start, train_end)
            distance_only = fit_distance_only_model(
                port_monthly, anchor_series, neighbor_signal, train_start, train_end
            )
            trial_volume = self._trial_baseline_volume(port_monthly, train_start, train_end)
            return {"stacked": stacked, "distance_only": distance_only, "trial_volume": trial_volume}

        def score_fn(params, test_start, test_end):
            blended = self._blend_step2_predictions(port_monthly, anchor_series, neighbor_signal, params)
            mask = (port_monthly["date"] >= test_start) & (port_monthly["date"] < test_end)
            return score_mape(blended[mask], port_monthly["price"][mask])

        result = select_best_window(consts.WINDOW_CANDIDATE_YEARS, rolling_splits, fit_fn, score_fn)

        if result is None:
            self.logger.warning(
                f"LocalMarket {self.market_code}: no stable Step-2 window found for {key}, "
                f"falling back to Step-1-only"
            )
            return None

        label, years = result
        train_start = latest_date - relativedelta(years=years) if years is not None else None
        final_params = fit_fn(train_start, latest_date)
        self.step2_stacked_models[key] = final_params["stacked"]
        self.step2_distance_only_models[key] = final_params["distance_only"]
        self.step2_window_labels[key] = label
        return label

    def _trial_baseline_volume(self, port_monthly, train_start, train_end):
        port = port_monthly.copy()
        port["month"] = port["date"].dt.month
        train = port[port["date"] < train_end]
        if train_start is not None:
            train = train[train["date"] >= train_start]
        return train.groupby("month")["landings_kg"].mean()

    def _blend_step2_predictions(self, port_monthly, anchor_series, neighbor_signal, params) -> pd.Series:
        """Dynamic-switch-blended prediction used ONLY during fit_step2's rolling-origin
        scoring (spec section 7's "score each with the same rolling-origin validation
        methodology" applied to the compound Stacked/DistanceOnly fitting unit)."""
        stacked_pred = predict_stacked_model(port_monthly, anchor_series, neighbor_signal, params["stacked"])
        distance_pred = predict_distance_only_model(
            port_monthly, anchor_series, neighbor_signal, params["distance_only"]
        )

        port = port_monthly.copy()
        port["month"] = port["date"].dt.month
        baseline_volume = params["trial_volume"]
        volume_ratio = port["landings_kg"].fillna(0.0) / port["month"].map(baseline_volume)

        use_distance_only = (
            volume_ratio.isna()
            | (volume_ratio < consts.LOW_VOLUME_RATIO_THRESHOLD)
            | (volume_ratio > consts.HIGH_VOLUME_RATIO_THRESHOLD)
        )
        blended = stacked_pred.where(~use_distance_only, distance_pred)
        blended.index = port_monthly.index
        return blended

    def predict_step2(
        self,
        species_code: str,
        category_code: str,
        date: pd.Timestamp,
        volume: float,
        anchor_value: float,
        neighbor_value: float,
    ) -> float | None:
        """Runtime dynamic switch (spec section 5 step 4). VolumeRatio reads from
        self.volume_baseline_stats -- the same incrementally-updated structure Step 1
        seeds and update_volume_baseline keeps current (plan header decision 9), not a
        separate Step-2-only baseline."""
        key = (species_code, category_code)
        if key not in self.step2_stacked_models or key not in self.step2_distance_only_models:
            return None

        stats = self.volume_baseline_stats.get(key, {})
        month_stats = stats.get(date.month)
        if month_stats is None or month_stats[1] == 0 or month_stats[0] <= 0:
            volume_ratio = None
        else:
            baseline = month_stats[0] / month_stats[1]
            volume_ratio = (volume / baseline) if baseline > 0 else None

        use_distance_only = (
            pd.isna(volume_ratio)
            or volume_ratio < consts.LOW_VOLUME_RATIO_THRESHOLD
            or volume_ratio > consts.HIGH_VOLUME_RATIO_THRESHOLD
        )

        single_row = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
        anchor_series = pd.Series([anchor_value], index=[date])
        neighbor_signal = pd.Series([neighbor_value], index=[date])

        if use_distance_only:
            predicted = predict_distance_only_model(
                single_row, anchor_series, neighbor_signal, self.step2_distance_only_models[key]
            )
        else:
            predicted = predict_stacked_model(
                single_row, anchor_series, neighbor_signal, self.step2_stacked_models[key]
            )

        value = predicted.iloc[0]
        return float(value) if pd.notna(value) else None
