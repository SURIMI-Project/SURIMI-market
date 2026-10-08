"""
Continuous trend confidence (spec
2026-10-08-combined-market-trend-term-v2-design.md, section 3, Step 2) for
a (species_code, category_code) key's CombinedMarket fit. Uses ONLY the
complete real historical record -- no train/test split, no dependency on
level_volume_model.py's own fit. Per-calendar-month OLS (year -> price),
same slope/standard-error/t-statistic math as v1's trend_gate.py, but
returned as a continuous fraction in [0, 1] instead of a boolean -- there
is no sign-agreement requirement and no significant-months threshold. A
species with no real trend naturally measures near 0; a species with a
clear, sustained trend measures near 1. Never a hard exclusion.
"""
import numpy as np
import pandas as pd

from . import consts


def compute_trend_confidence(monthly: pd.DataFrame) -> float:
    """
    Fraction (0.0-1.0) of the 12 calendar months whose own year-over-year
    OLS slope is individually significant (|t| > TREND_CONFIDENCE_T_THRESHOLD)
    in the majority sign direction across all 12 months. A month with
    fewer than 3 years of observations, or whose observations all fall in
    a single year (zero year-variance, so no slope is estimable), simply
    does not count toward the numerator -- it is treated as "not
    significant" for that month rather than raising or forcing the whole
    key to a fixed value, since this is a continuous measure, not a gate.
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
            continue  # not enough evidence to assess this month

        x_bar, y_bar = years.mean(), prices.mean()
        sxx = ((years - x_bar) ** 2).sum()
        if sxx == 0:
            continue  # every observation of this month falls in the same year

        slope = ((years - x_bar) * (prices - y_bar)).sum() / sxx
        intercept = y_bar - slope * x_bar
        residuals = prices - (intercept + slope * years)
        s2 = (residuals ** 2).sum() / (n - 2)
        se_slope = np.sqrt(s2 / sxx)
        t_stat = slope / se_slope if se_slope > 0 else 0.0

        slopes[month] = slope
        tstats[month] = t_stat

    if not slopes:
        return 0.0

    n_positive = sum(1 for s in slopes.values() if s > 0)
    n_negative = sum(1 for s in slopes.values() if s < 0)
    if n_positive == 0 and n_negative == 0:
        return 0.0
    majority_sign = 1 if n_positive >= n_negative else -1

    n_significant = sum(
        1 for month, t in tstats.items()
        if abs(t) > consts.TREND_CONFIDENCE_T_THRESHOLD and np.sign(slopes[month]) == majority_sign
    )
    return n_significant / 12
