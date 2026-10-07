import numpy as np
import pandas as pd

from src.models.market.trend_gate import passes_trend_gate


def _make_monthly(n_years, slope, noise_std, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_years * 12, freq="MS")
    years = dates.year.to_numpy(dtype=float)
    seasonal = np.array([2.0 + 0.1 * d.month for d in dates])
    trend = slope * (years - years.min())
    noise = rng.normal(0, noise_std, len(dates))
    price = seasonal + trend + noise
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": [1000.0] * len(dates)})


def test_passes_for_a_clear_consistent_trend():
    monthly = _make_monthly(n_years=10, slope=0.1, noise_std=0.05, seed=0)
    assert passes_trend_gate(monthly) is True


def test_fails_for_flat_noise_with_no_real_trend():
    monthly = _make_monthly(n_years=10, slope=0.0, noise_std=0.1, seed=0)
    assert passes_trend_gate(monthly) is False


def test_fails_when_sign_agrees_but_not_significant():
    # Shaped after the real HKE@FR near-miss found during investigation:
    # all 12 months drift the same direction, but the slope is small
    # relative to its own noise, so only a few months are individually
    # significant -- sign agreement alone must not be enough to pass.
    monthly = _make_monthly(n_years=10, slope=0.03, noise_std=0.12, seed=6)
    assert passes_trend_gate(monthly) is False


def test_fails_with_insufficient_history():
    monthly = _make_monthly(n_years=2, slope=0.1, noise_std=0.05, seed=0)
    assert passes_trend_gate(monthly) is False


def test_fails_when_a_month_has_zero_variance_in_years():
    # Degenerate: January has multiple observations (n >= 3) but all from
    # the same calendar year, so sxx == 0 -- no year-over-year slope is
    # estimable for that month at all. Other months have normal 3-year
    # data with clear positive trend to ensure the gate reaches January's
    # sxx == 0 check (not failing earlier on insufficient history).
    dates = []
    prices = []

    # January: 3 observations, all from 2015 (sxx will be 0)
    dates.extend([
        pd.Timestamp(2015, 1, 1),
        pd.Timestamp(2015, 1, 15),
        pd.Timestamp(2015, 1, 28),
    ])
    prices.extend([2.0, 2.05, 2.02])

    # Other months: one observation per year (2015, 2016, 2017)
    # with consistent positive trend to ensure they all pass their checks
    for month in range(2, 13):
        for year in [2015, 2016, 2017]:
            dates.append(pd.Timestamp(year=year, month=month, day=15))
            prices.append(2.0 + (year - 2015) * 0.15)  # Clear positive trend

    monthly = pd.DataFrame({
        "date": dates,
        "price": prices,
        "landings_kg": [1000.0] * len(dates),
    })
    assert passes_trend_gate(monthly) is False
