"""
Continuous trend confidence for a (species_code, category_code) key's
CombinedMarket fit. Uses only the full real historical record -- no
train/test split, no dependency on level_volume_model's own fit.
Per-calendar-month OLS (year -> price) yields a fraction in [0, 1]
instead of a boolean gate: no sign-agreement requirement, no
significant-months threshold, never a hard exclusion. A species with no
real trend measures near 0; a clear, sustained trend measures near 1.
"""
import numpy as np
import pandas as pd

from . import consts


def compute_trend_confidence(monthly: pd.DataFrame, train_end: pd.Timestamp) -> float:
    """
    Fraction (0.0-1.0) of the 12 calendar months whose year-over-year OLS
    slope is individually significant in the majority sign direction.
    Significance is tested against a critical value scaled to each
    month's own degrees-of-freedom (via consts.trend_confidence_critical_value),
    not a fixed |t| threshold -- a month with only 3 years of data needs
    a much higher |t| to be genuinely significant than a month with 15
    years does, and a flat threshold either over-credits thin months or
    under-credits well-sampled ones. A month with fewer than 3 years of
    data, or zero year-variance (no slope estimable), simply doesn't
    count toward the numerator rather than raising or forcing a result.

    Filters to date < train_end (previously this function used the full,
    unfiltered monthly series regardless of train_end, unlike
    fit_trend_rate_model, which did filter -- a real data-leakage
    inconsistency between the two trend sub-procedures, fixed here).
    """
    monthly = monthly.dropna(subset=["price"]).copy()
    monthly = monthly[monthly["date"] < train_end]
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    slopes: dict[int, float] = {}
    tstats: dict[int, float] = {}
    dfs: dict[int, int] = {}
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
        df = n - 2
        s2 = (residuals ** 2).sum() / df
        se_slope = np.sqrt(s2 / sxx)
        t_stat = slope / se_slope if se_slope > 0 else 0.0

        slopes[month] = slope
        tstats[month] = t_stat
        dfs[month] = df

    if not slopes:
        return 0.0

    n_positive = sum(1 for s in slopes.values() if s > 0)
    n_negative = sum(1 for s in slopes.values() if s < 0)
    if n_positive == 0 and n_negative == 0:
        return 0.0
    majority_sign = 1 if n_positive >= n_negative else -1

    n_significant = sum(
        1 for month, t in tstats.items()
        if abs(t) > consts.trend_confidence_critical_value(dfs[month]) and np.sign(slopes[month]) == majority_sign
    )
    return n_significant / 12
