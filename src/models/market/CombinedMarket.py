"""
One instance per 2-letter market code (spec section 3). Holds the frozen
Model 2 (seasonal baseline + volume-response), keyed by
(species_code, category_code). Never holds or queries LocalMarket
references -- stays passive, only ever receives plain numbers (spec
section 4).
"""
from collections import defaultdict
import logging
import threading

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from . import consts
from .level_volume_model import (
    fit_level_volume_model,
    predict_level_volume,
    fit_trend_rate_model,
    predict_trend_correction,
    fit_level_only_model,
    predict_level_only,
    score_mape,
)
from .trend_confidence import compute_trend_confidence
from .window_selection import build_rolling_splits, select_best_window


class CombinedMarket:
    def __init__(self, market_code: str):
        self.market_code = market_code
        self.logger = logging.getLogger(consts.COMBINED_MARKET_LOGGER_NAME)

        # Direct catch accumulator for sales routed straight to this 2-letter
        # prefix (spec section 4's lenient catch-all) -- same shape as
        # LocalMarket's own accumulator.
        self._direct_catch: dict[tuple[str, str], float] = defaultdict(float)
        self._lock = threading.Lock()

        # (species_code, category_code) -> frozen params dict (baseline_price,
        # baseline_volume, alpha, beta) or level-only params (alpha=beta=0).
        self.models: dict[tuple[str, str], dict] = {}
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
        Fit and freeze Model 2 for every (species_code, category_code),
        selecting the best training-window length independently per key
        (spec section 7). Falls back to a level-only fit (spec section 9
        point 2) if no window produces a stable level-volume model; excludes
        the key entirely (spec section 9 point 4) if even that fails.

        Window selection and the frozen baseline (baseline_price/alpha/beta)
        are completely unconditional -- identical for every key, with no
        trend-related branching anywhere in this path (spec
        2026-10-08-combined-market-trend-term-v2-design.md section 3, Step
        1 / Global Constraints). Trend confidence and rate (Steps 2-3) are
        computed separately from the complete monthly series and attached
        to the frozen params afterward via _attach_trend_params -- for
        every key that gets a model at all, main path or fallback alike --
        so get_price() never needs to branch on whether a key "has" a
        trend; the correction it applies is just ~1.0 (a no-op) for a
        non-trending key.
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
                    params = fit_level_volume_model(monthly, train_start, latest_date)
                except (ValueError, np.linalg.LinAlgError):
                    # The window length scored acceptably across historical rolling-origin
                    # splits (each with its own, earlier train_end), but real volume data is
                    # sparse enough that the literal final window (train_start..latest_date)
                    # can still fail to fit -- e.g. every non-null landings_kg reading in this
                    # exact window happens to be the sole representative of its calendar
                    # month, making the volume-anomaly series degenerate (zero variance).
                    # Cross-validation generalizing to historical splits doesn't guarantee the
                    # production window itself is fittable, so fall through to the same
                    # lower-tier fallback used when no window was selected at all.
                    self.logger.warning(
                        f"CombinedMarket {self.market_code}: selected window {label} for {key} "
                        f"scored acceptably in rolling-origin validation but failed to re-fit on "
                        f"the final window ending at {latest_date}; falling back to level-only fit"
                    )
                    params = None

                if params is not None:
                    self._attach_trend_params(params, monthly, latest_date)
                    self.models[key] = params
                    self.window_labels[key] = label
                    self._seed_volume_baseline_stats(key, monthly, train_start, latest_date)
                    selected[key] = label
                    continue

            # Fallback cascade (spec section 9 point 2): try level-only.
            else:
                self.logger.warning(
                    f"CombinedMarket {self.market_code}: no stable level-volume window found "
                    f"for {key}, falling back to level-only fit"
                )
            try:
                params = fit_level_only_model(monthly, None, latest_date)
                self._attach_trend_params(params, monthly, latest_date)
                self.models[key] = params
                self.window_labels[key] = "level-only-fallback"
                selected[key] = "level-only-fallback"
            except ValueError:
                self.logger.warning(
                    f"CombinedMarket {self.market_code}: no usable price history at all "
                    f"for {key}, excluding from output"
                )
                selected[key] = None

        return selected

    def _attach_trend_params(self, params: dict, monthly: pd.DataFrame, train_end: pd.Timestamp) -> None:
        """
        spec 2026-10-08-combined-market-trend-term-v2-design.md section 3,
        Steps 2-4: continuous confidence + block-based rate, computed from
        the complete real historical record and attached unconditionally
        -- same procedure for every key, main level-volume path and the
        level-only fallback alike, so get_price() can apply the correction
        without ever checking which path produced a given key's params.
        """
        trend_rate_params = fit_trend_rate_model(monthly, train_end)
        params["trend_confidence"] = compute_trend_confidence(monthly)
        params["trend_rate"] = trend_rate_params["rate"]
        params["trend_years_trained"] = trend_rate_params["years_trained"]
        params["trend_train_end"] = train_end

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

    def update_volume_baseline(self, species_code: str, category_code: str, date: pd.Timestamp, volume: float):
        """Incremental, never-decaying running mean (spec section 6, plan header decision 9)."""
        key = (species_code, category_code)
        if key not in self.volume_baseline_stats or pd.isna(volume):
            return

        month = date.month
        prev_sum, prev_count = self.volume_baseline_stats[key].get(month, (0.0, 0))
        new_sum, new_count = prev_sum + volume, prev_count + 1
        self.volume_baseline_stats[key][month] = (new_sum, new_count)

        if key in self.models:
            if self.models[key]["baseline_volume"] is not None:
                self.models[key]["baseline_volume"][month] = new_sum / new_count

    def load_price_time_series(self, prices_df: pd.DataFrame):
        """
        Load real recorded prices for this market_code (spec section 8).
        Filters to blended rows only (size is NaN) -- spec section 13:
        "The core model filters specifically to Size = NaN rows for its
        own fitting." Done here, not assumed pre-filtered by the caller,
        so this holds regardless of what the caller passes in.
        """
        rows = prices_df[(prices_df["market_code"] == self.market_code) & (prices_df["size"].isna())]
        for row in rows.itertuples(index=False):
            self.price_time_series[(row.species_code, row.category_code, row.date)] = row.price

    def get_price(
        self, species_code: str, category_code: str, date: pd.Timestamp, volume: float
    ) -> float | None:
        """
        Real-price-first (spec section 8): check price_time_series, else
        predict. The trend correction (spec v2 section 3, Step 4) is
        applied unconditionally -- it is a pure multiplier of ~1.0 for a
        key with near-zero trend confidence, so there is no branch here on
        whether a key "has" a trend, unlike v1's "gamma" in params check.
        """
        key = (species_code, category_code)
        real_price = self.price_time_series.get((species_code, category_code, date))
        if pd.notna(real_price):
            return real_price

        if key not in self.models:
            return None

        params = self.models[key]
        monthly = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
        if params.get("baseline_volume") is None:
            predicted = predict_level_only(monthly, params)
        else:
            predicted = predict_level_volume(monthly, params)
        predicted = predicted * predict_trend_correction(monthly, params)
        value = predicted.iloc[0]
        return float(value) if pd.notna(value) else None

    def get_live_anchor_series(
        self, species_code: str, category_code: str, own_monthly: pd.DataFrame, target_dates
    ) -> pd.Series | None:
        """
        The alternate Step-2 anchor scheme (ANCHOR_SCHEME_LIVE, spec
        section 5's open item / plan header decision 1) -- this market's
        own frozen coefficients applied to its OWN real historical volume
        at each date, varying month-to-month, unlike
        get_baseline_price_table's fixed 12-month seasonal cycle.
        Analysis-only: never used by the live production path, which
        always calls get_baseline_price_table instead.

        own_monthly is this CombinedMarket's own (date, price, landings_kg)
        series (the same shape MarketSystem.fit_all already builds via
        _build_monthly_by_key for this market_code/key, retained by the
        caller since CombinedMarket itself doesn't keep it after fit()
        returns). Returns None if this market has no frozen model for the
        key at all; otherwise reindexed (by date) to target_dates, with
        NaN for any date outside own_monthly's own range.
        """
        key = (species_code, category_code)
        if key not in self.models:
            return None

        params = self.models[key]
        if params.get("baseline_volume") is None:
            predicted = predict_level_only(own_monthly, params)
        else:
            predicted = predict_level_volume(own_monthly, params)
        predicted = predicted.copy()
        predicted.index = own_monthly["date"]
        return predicted.reindex(pd.DatetimeIndex(target_dates))

    def get_baseline_price_table(self, species_code: str, category_code: str) -> pd.Series | None:
        """
        Frozen per-month seasonal baseline -- used as LocalMarket Step 2's
        static anchor. Always reindexed to the full 1..12 calendar-month
        range: the underlying baseline_price (especially on the level-only
        fallback path, which groups only over whatever months happen to
        appear in a possibly-short training window) may not cover every
        month. Genuinely-missing months come back as NaN rather than being
        silently dropped or fabricated -- downstream consumers (Task 9) are
        expected to treat a NaN anchor as "no data for this month".
        """
        key = (species_code, category_code)
        if key not in self.models:
            return None
        return self.models[key]["baseline_price"].reindex(range(1, 13))

    def record_history(self, tick: int, prices: dict, volumes: dict):
        self.price_history.append((tick, {"prices": dict(prices), "volumes": dict(volumes)}))

    def add_direct_catch(self, species_code: str, category_code: str, quantity: float):
        with self._lock:
            self._direct_catch[(species_code, category_code)] += quantity

    def clear_direct_catch(self):
        with self._lock:
            self._direct_catch.clear()

    def get_direct_catch_snapshot(self) -> dict:
        with self._lock:
            return dict(self._direct_catch)
