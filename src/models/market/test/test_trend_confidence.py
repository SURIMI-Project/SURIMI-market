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


def _train_end_after(monthly: pd.DataFrame) -> pd.Timestamp:
    """One year past the synthetic data's last date -- keeps every row in
    scope for tests that aren't specifically exercising the train_end
    filtering boundary."""
    return monthly["date"].max() + pd.DateOffset(years=1)


def test_confidence_is_1_for_a_clear_consistent_trend():
    monthly = _make_monthly(n_years=10, slope=0.1, noise_std=0.05, seed=0)
    assert compute_trend_confidence(monthly, _train_end_after(monthly)) == pytest.approx(1.0)


def test_confidence_is_low_for_flat_noise_with_no_real_trend():
    monthly = _make_monthly(n_years=10, slope=0.0, noise_std=0.1, seed=0)
    assert compute_trend_confidence(monthly, _train_end_after(monthly)) == pytest.approx(1 / 12)


def test_confidence_is_partial_when_sign_agrees_but_not_all_months_significant():
    # Mirrors a real near-miss case (HKE@FR): all 12 months drift the same
    # direction, but the slope is small relative to noise, so only some
    # months are individually significant -- confidence must land
    # strictly between 0 and 1, not at an extreme.
    #
    # Was 3/12 (0.25) under the OLD fixed |t| > 2.0 threshold; under the
    # df-scaled critical value fix (Stage 6b), df=8 here needs |t| > 2.306,
    # not 2.0, so one fewer month now clears the bar -- 2/12. This is the
    # fix working as intended, not a regression: this fixture's point is
    # "strictly between 0 and 1", which still holds.
    monthly = _make_monthly(n_years=10, slope=0.03, noise_std=0.12, seed=6)
    confidence = compute_trend_confidence(monthly, _train_end_after(monthly))
    assert confidence == pytest.approx(2 / 12)
    assert 0.0 < confidence < 1.0


def test_confidence_is_zero_with_insufficient_history():
    monthly = _make_monthly(n_years=2, slope=0.1, noise_std=0.05, seed=0)
    assert compute_trend_confidence(monthly, _train_end_after(monthly)) == 0.0


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
    train_end = monthly["date"].max() + pd.DateOffset(years=1)
    assert compute_trend_confidence(monthly, train_end) == pytest.approx(11 / 12)


def test_df_scaling_rejects_a_thin_month_the_old_fixed_threshold_would_accept():
    """
    Stage 6b's primary fix: a month backed by only 3-4 years of data
    (df=1-2) needs a much higher |t| to be genuinely significant than the
    old fixed |t| > 2.0 threshold demanded -- the true 95% critical value
    at df=2 is ~4.303, at df=1 it's ~12.706 (consts.trend_confidence_critical_value).
    Construct a single calendar month (January) across 4 years whose
    year-over-year OLS |t| clears the OLD flat 2.0 bar but not the
    properly df-scaled one (df=2 here), and confirm it does NOT count
    toward confidence's numerator -- the old logic would have counted it.
    """
    years = np.array([2020.0, 2021.0, 2022.0, 2023.0])
    # Small, noisy upward drift: enough for |t| to clear 2.0 but land
    # well short of the df=2 critical value of ~4.303.
    prices = np.array([2.00, 2.08, 2.05, 2.19])

    x_bar, y_bar = years.mean(), prices.mean()
    sxx = ((years - x_bar) ** 2).sum()
    slope = ((years - x_bar) * (prices - y_bar)).sum() / sxx
    intercept = y_bar - slope * x_bar
    residuals = prices - (intercept + slope * years)
    df = len(years) - 2
    s2 = (residuals ** 2).sum() / df
    se_slope = np.sqrt(s2 / sxx)
    t_stat = slope / se_slope
    # Sanity-check the construction before relying on it below.
    assert 2.0 < abs(t_stat) < 4.303, (
        f"test fixture must clear the OLD flat threshold (2.0) but not the "
        f"properly df-scaled one (~4.303 at df=2); got |t|={abs(t_stat)}"
    )

    dates = [pd.Timestamp(year=int(y), month=1, day=15) for y in years]
    monthly = pd.DataFrame({
        "date": dates,
        "price": prices,
        "landings_kg": [1000.0] * len(dates),
    })
    train_end = monthly["date"].max() + pd.DateOffset(years=1)

    # Only January has any data at all, so if it's (wrongly) counted
    # significant, confidence is 1/12; if correctly rejected for being
    # under-powered, confidence is 0.0 -- no other month contributes.
    assert compute_trend_confidence(monthly, train_end) == pytest.approx(0.0)


def test_filters_to_before_train_end_only():
    """
    Data-leakage fix: a genuine trend placed entirely AFTER train_end
    must not be visible to the confidence calculation at all -- the old
    compute_trend_confidence(monthly) never filtered by train_end,
    unlike fit_trend_rate_model, which did.
    """
    train_end = pd.Timestamp("2020-01-01")
    flat_history = _make_monthly(n_years=5, slope=0.0, noise_std=0.1, seed=1)
    flat_history = flat_history[flat_history["date"] < train_end]

    future_trend = _make_monthly(n_years=10, slope=0.3, noise_std=0.05, seed=0)
    # Re-date the "future" trend rows so they land at/after train_end
    # relative to the flat history above (compute_trend_confidence filters
    # with date < train_end, so train_end itself is already excluded).
    offset = train_end - future_trend["date"].min()
    future_trend = future_trend.copy()
    future_trend["date"] = future_trend["date"] + offset

    combined = pd.concat([flat_history, future_trend], ignore_index=True).sort_values("date")

    confidence_with_future = compute_trend_confidence(combined, train_end)
    confidence_flat_only = compute_trend_confidence(flat_history, train_end)

    # Only the pre-train_end (flat) portion should ever be visible --
    # identical result whether or not the future trend rows are present.
    assert confidence_with_future == pytest.approx(confidence_flat_only)
    # And it should actually be the LOW, flat-series confidence, not
    # inflated by the strong trend sitting after train_end.
    assert confidence_with_future < 0.3


def _make_cyclical_monthly(n_years=15, period_years=7.0, amplitude=0.3, noise_std=0.02, seed=0):
    """
    Multi-year sine-like cyclical price pattern with NO genuine linear
    drift -- the adversarial case ARCHITECTURE.md section 6's SAA@FR
    lesson documents: a cyclical, not genuinely trending, price history
    must not measure a confidently high trend.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_years * 12, freq="MS")
    t_years = (dates - dates[0]).days / 365.25
    cycle = amplitude * np.sin(2 * np.pi * t_years / period_years)
    seasonal = 0.05 * dates.month
    noise = rng.normal(0, noise_std, len(dates))
    price = 2.0 + seasonal + cycle + noise
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": [1000.0] * len(dates)})


def test_does_not_mistake_a_cyclical_series_for_a_trend():
    """
    The SAA@FR adversarial case (previously written against the reverted
    single-pooled-OLS fit_trend_model as an xfail -- see .stage6-report.md
    -- now targets compute_trend_confidence instead, and is expected to
    PASS: the restored per-calendar-month structure compares each
    calendar month across years independently, so a multi-year cycle
    (whose within-year-month residuals are far less autocorrelated than
    raw consecutive monthly residuals) doesn't inflate significance the
    way pooling ~180 raw monthly points into one OLS did.
    """
    monthly = _make_cyclical_monthly(n_years=15, period_years=7.0, amplitude=0.3, noise_std=0.02, seed=0)
    train_end = monthly["date"].max() + pd.DateOffset(months=1)

    confidence = compute_trend_confidence(monthly, train_end)

    assert confidence < 0.3, (
        f"expected a cyclical, non-trending series to measure low confidence, "
        f"got {confidence}"
    )
