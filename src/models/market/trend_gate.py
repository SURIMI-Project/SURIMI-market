"""
Direct-evidence gate (spec 2026-10-07-combined-market-trend-term-design.md,
section 3) for whether a (species_code, category_code) key's CombinedMarket
fit should include a trend term. Uses ONLY the complete real historical
record -- no train/test split, no dependency on level_volume_model.py's own
fit. Per-calendar-month OLS (year -> price), requiring unanimous sign
agreement across all 12 months AND a strong majority individually
significant in that same direction.
"""
import numpy as np
import pandas as pd

from . import consts


def passes_trend_gate(monthly: pd.DataFrame) -> bool:
    """
    Determine whether a (species_code, category_code) key's CombinedMarket
    fit should include a trend term using per-calendar-month OLS regression
    on the complete real historical record.

    Returns True iff all 12 calendar months exhibit unanimous sign agreement
    in their year-over-year price slopes, AND at least TREND_GATE_MIN_SIGNIFICANT_MONTHS
    are individually statistically significant (|t| > TREND_GATE_T_THRESHOLD) in
    that same direction. Returns False for any month with fewer than 3 years
    of data, or if all years in any month are identical (no slope estimable).

    Args:
        monthly: DataFrame with columns date, price, landings_kg (price/landings_kg
                 may be NaN for gaps). Only date and price are read.

    Returns:
        bool: True if gate passes; False otherwise.
    """
    monthly = monthly.dropna(subset=["price"]).copy()
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    slopes: dict[int, float] = {}
    tstats: dict[int, float] = {}
    for month in range(1, 13):
        sub = monthly[monthly["month"] == month]
        years = sub["year"].to_numpy(dtype=float)
        prices = sub["price"].to_numpy(dtype=float)
        n = len(years)
        if n < 3:
            return False  # not enough evidence to assess every month

        x_bar, y_bar = years.mean(), prices.mean()
        sxx = ((years - x_bar) ** 2).sum()
        if sxx == 0:
            return False  # every observation of this month falls in the same year

        slope = ((years - x_bar) * (prices - y_bar)).sum() / sxx
        intercept = y_bar - slope * x_bar
        residuals = prices - (intercept + slope * years)
        s2 = (residuals ** 2).sum() / (n - 2)
        se_slope = np.sqrt(s2 / sxx)
        # Edge case: if slope != 0 and se_slope == 0 (perfect fit with zero
        # residual variance), t_stat would be infinity. We conservatively treat
        # it as non-significant (t=0), though this is extremely unlikely with
        # real noisy price data.
        t_stat = slope / se_slope if se_slope > 0 else 0.0

        slopes[month] = slope
        tstats[month] = t_stat

    signs = {np.sign(s) for s in slopes.values()}
    if len(signs) != 1:
        return False  # not unanimous

    majority_sign = next(iter(signs))
    if majority_sign == 0:
        return False  # exactly-flat consensus is not a trend

    n_significant = sum(
        1 for month, t in tstats.items()
        if abs(t) > consts.TREND_GATE_T_THRESHOLD and np.sign(slopes[month]) == majority_sign
    )
    return n_significant >= consts.TREND_GATE_MIN_SIGNIFICANT_MONTHS
