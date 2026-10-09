"""
Per-(species_code, category_code)-key tracker of observed landings volume,
by calendar month -- used to compute how far a tick's volume is from what's
normal for that month ("volume anomaly"/"volume ratio"). Wraps
level_volume_model.compute_volume_baseline_stats/apply_volume_baseline_update,
which stay the free functions doing the actual math; this class just gives
the per-key running state a home instead of a raw dict CombinedMarket and
LocalMarket each poked at directly and identically.
"""
import pandas as pd

from .level_volume_model import compute_volume_baseline_stats


class VolumeBaselineTracker:
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
