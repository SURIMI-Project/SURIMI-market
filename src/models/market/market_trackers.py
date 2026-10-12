"""
Three small, always-together per-key running-state trackers that
CombinedMarket and LocalMarket both hold identically: `CatchAccumulator`
(catch quantities routed between ticks), `RealPriceLookup` (real recorded
prices, keyed by species/category/date), and `VolumeBaselineTracker`
(running-mean landings volume per calendar month, used for the
volume-anomaly term). Kept in one file -- deliberately diverging from the
usual one-class-per-file convention, same justification as
price_models.py -- because these three are never read or changed
independently of one another: they are CombinedMarket/LocalMarket's
shared per-key state layer as one concept, not three unrelated utilities
that happen to be small.
"""
from collections import defaultdict
import threading

import pandas as pd

from .level_volume_model import compute_volume_baseline_stats, populate_price_time_series


class CatchAccumulator:
    """Thread-safe accumulator for catch quantities routed to a market
    between ticks -- the same pattern CombinedMarket and LocalMarket each
    need for their own catch bookkeeping."""

    def __init__(self):
        self._totals: dict[tuple[str, str], float] = defaultdict(float)
        self._lock = threading.Lock()

    def add(self, species_code: str, category_code: str, quantity: float):
        with self._lock:
            self._totals[(species_code, category_code)] += quantity

    def clear(self):
        with self._lock:
            self._totals.clear()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._totals)

    def take(self) -> dict:
        """Atomically snapshot and clear in one lock acquisition -- unlike
        calling .snapshot() then .clear() separately, this leaves no window
        where a concurrent .add() could be captured by neither the snapshot
        nor the next tick's."""
        with self._lock:
            result = dict(self._totals)
            self._totals.clear()
            return result


class RealPriceLookup:
    """Real recorded prices for one market_code, keyed by (species_code,
    category_code, date). Wraps level_volume_model.populate_price_time_series,
    which stays the free function doing the actual CSV-row filtering; this
    class just gives the per-market lookup table a home instead of a raw
    dict CombinedMarket and LocalMarket each built/read identically."""

    def __init__(self):
        self._prices: dict[tuple[str, str, pd.Timestamp], float] = {}

    def load(self, prices_df: pd.DataFrame, market_code: str) -> None:
        populate_price_time_series(self._prices, prices_df, market_code)

    def get(self, species_code: str, category_code: str, date: pd.Timestamp) -> float | None:
        return self._prices.get((species_code, category_code, date))


class VolumeBaselineTracker:
    """Per-(species_code, category_code)-key tracker of observed landings
    volume, by calendar month -- used to compute how far a tick's volume
    is from what's normal for that month ("volume anomaly"/"volume
    ratio"). .seed() delegates to level_volume_model.compute_volume_baseline_stats,
    the free function doing that initial-fit math; .update()'s incremental
    running-mean logic is implemented directly here instead -- there's no
    free-function equivalent it's wrapping. This class just gives the
    per-key running state a home instead of a raw dict CombinedMarket and
    LocalMarket each poked at directly and identically."""

    def __init__(self):
        self._stats: dict[int, tuple[float, int]] = {}

    def seed(self, monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp) -> None:
        self._stats = compute_volume_baseline_stats(monthly, train_start, train_end)

    def update(self, date: pd.Timestamp, volume: float) -> tuple[int, float] | None:
        """Incremental, never-decaying running-mean update. Returns
        (month, new_mean), or None if volume is NaN -- the caller patches
        new_mean into its own frozen model's baseline_volume[month]."""
        if pd.isna(volume):
            return None
        month = date.month
        prev_sum, prev_count = self._stats.get(month, (0.0, 0))
        new_sum, new_count = prev_sum + volume, prev_count + 1
        self._stats[month] = (new_sum, new_count)
        return month, new_sum / new_count

    def ratio(self, date: pd.Timestamp, volume: float) -> float:
        """volume / this calendar month's running-mean baseline. NaN if
        there's no usable baseline for this month yet (never seeded,
        zero observations, or a non-positive mean)."""
        month_stats = self._stats.get(date.month)
        if month_stats is None or month_stats[1] == 0 or month_stats[0] <= 0:
            return float("nan")
        baseline = month_stats[0] / month_stats[1]
        return (volume / baseline) if baseline > 0 else float("nan")

    def snapshot(self) -> dict[int, tuple[float, int]]:
        """Read-only copy of the raw per-month (running_sum, running_count)
        pairs -- for callers (tests, diagnostics) that need to observe the
        bookkeeping directly rather than through .ratio()."""
        return dict(self._stats)


def seed_tracker(
    volume_baseline: dict[tuple[str, str], "VolumeBaselineTracker"],
    key: tuple[str, str],
    monthly: pd.DataFrame,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
) -> None:
    """Shared by CombinedMarket and LocalMarket (identical logic in both,
    previously each class's own private _seed_volume_baseline_stats):
    construct a fresh VolumeBaselineTracker for key, seed it from monthly's
    training range, and store it in volume_baseline."""
    tracker = VolumeBaselineTracker()
    tracker.seed(monthly, train_start, train_end)
    volume_baseline[key] = tracker


def update_tracker_and_model(
    volume_baseline: dict[tuple[str, str], "VolumeBaselineTracker"],
    models: dict[tuple[str, str], object],
    key: tuple[str, str],
    date: pd.Timestamp,
    volume: float,
) -> None:
    """Shared by CombinedMarket.update_volume_baseline and
    LocalMarket.update_volume_baseline (identical logic in both):
    incremental, never-decaying running-mean update, patched into the
    frozen model's own baseline_volume[month] if key has both a tracker
    and a fitted model."""
    tracker = volume_baseline.get(key)
    if tracker is None:
        return
    result = tracker.update(date, volume)
    if result is None:
        return
    month, new_mean = result
    if key in models:
        models[key].update_baseline_volume(month, new_mean)
