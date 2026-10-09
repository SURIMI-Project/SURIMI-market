import numpy as np
import pandas as pd
import pytest

from src.models.market.trend_confidence import compute_trend_confidence


def _make_monthly(n_years, slope, noise_std, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_years * 12, freq="MS")
    years = dates.year.to_numpy(dtype=float)
    seasonal = np.array([2.0 + 0.1 * d.month for d in dates])
    trend = slope * (years - years.min())
    noise = rng.normal(0, noise_std, len(dates))
    price = seasonal + trend + noise
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": [1000.0] * len(dates)})


def test_confidence_is_1_for_a_clear_consistent_trend():
    monthly = _make_monthly(n_years=10, slope=0.1, noise_std=0.05, seed=0)
    assert compute_trend_confidence(monthly) == pytest.approx(1.0)


def test_confidence_is_low_for_flat_noise_with_no_real_trend():
    monthly = _make_monthly(n_years=10, slope=0.0, noise_std=0.1, seed=0)
    assert compute_trend_confidence(monthly) == pytest.approx(1 / 12)


def test_confidence_is_partial_when_sign_agrees_but_not_all_months_significant():
    # Mirrors a real near-miss case (HKE@FR): all 12 months drift the same
    # direction, but the slope is small relative to noise, so only some
    # months are individually significant -- confidence must land
    # strictly between 0 and 1, not at an extreme.
    monthly = _make_monthly(n_years=10, slope=0.03, noise_std=0.12, seed=6)
    assert compute_trend_confidence(monthly) == pytest.approx(0.25)


def test_confidence_is_zero_with_insufficient_history():
    monthly = _make_monthly(n_years=2, slope=0.1, noise_std=0.05, seed=0)
    assert compute_trend_confidence(monthly) == 0.0


def test_confidence_handles_a_month_with_zero_variance_in_years_without_crashing():
    # Degenerate case: January has n >= 3 observations but all in the
    # same year, so sxx == 0 -- no slope is estimable; it must not count
    # toward the numerator, not raise. Other months have 3 years
    # (2015-2017) with an exact linear trend and zero noise.
    dates = []
    prices = []

    dates.extend([
        pd.Timestamp(2015, 1, 1),
        pd.Timestamp(2015, 1, 15),
        pd.Timestamp(2015, 1, 28),
    ])
    prices.extend([2.0, 2.05, 2.02])

    for month in range(2, 13):
        for year in [2015, 2016, 2017]:
            dates.append(pd.Timestamp(year=year, month=month, day=15))
            prices.append(2.0 + (year - 2015) * 0.15)

    monthly = pd.DataFrame({
        "date": dates,
        "price": prices,
        "landings_kg": [1000.0] * len(dates),
    })
    # January is skipped (sxx == 0); the other 11 months have a perfectly
    # linear trend, which in floating point still leaves a tiny nonzero
    # residual, giving an enormous (not undefined) t-stat -- so all 11
    # evaluated months count as significant.
    assert compute_trend_confidence(monthly) == pytest.approx(11 / 12)
