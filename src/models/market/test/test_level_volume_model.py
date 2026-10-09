import numpy as np
import pandas as pd
import pytest

from src.models.market.level_volume_model import (
    fit_level_volume_model,
    predict_level_volume,
    score_mape,
    build_complete_monthly_series,
    fit_level_only_model,
    predict_level_only,
    fit_trend_rate_model,
    predict_trend_correction,
    with_month_column,
)


def _make_monthly(n_years=5, base_price=2.0, base_volume=1000.0, vol_response=-0.001, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    volume = base_volume + rng.normal(0, 50, len(dates))
    # Price responds negatively to volume anomaly, plus small noise.
    monthly_mean_vol = base_volume
    price = base_price + vol_response * (volume - monthly_mean_vol) + rng.normal(0, 0.01, len(dates))
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})


def test_fit_and_predict_recovers_known_relationship():
    monthly = _make_monthly()
    params = fit_level_volume_model(monthly, train_start=None, train_end=pd.Timestamp("2023-01-01"))

    assert params["beta"] < 0  # higher volume -> lower price, as constructed
    assert set(params["baseline_price"].index) == set(range(1, 13))

    predicted = predict_level_volume(monthly, params)
    assert len(predicted) == len(monthly)
    # Predictions should track real prices reasonably well on training data.
    train_mask = monthly["date"] < pd.Timestamp("2023-01-01")
    mape = score_mape(predicted[train_mask], monthly["price"][train_mask])
    assert mape < 5.0  # well-fit synthetic data, should be very low error


def test_fit_respects_train_start_window():
    monthly = _make_monthly()
    full = fit_level_volume_model(monthly, train_start=None, train_end=pd.Timestamp("2023-01-01"))
    windowed = fit_level_volume_model(
        monthly, train_start=pd.Timestamp("2021-01-01"), train_end=pd.Timestamp("2023-01-01")
    )
    # Different training windows should generally produce different coefficients.
    assert full["alpha"] != windowed["alpha"] or full["beta"] != windowed["beta"]


def test_fit_raises_on_insufficient_data():
    monthly = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"]),
        "price": [2.0],
        "landings_kg": [1000.0],
    })
    with pytest.raises(ValueError):
        fit_level_volume_model(monthly, train_start=None, train_end=pd.Timestamp("2021-01-01"))


def test_fit_raises_when_volume_has_zero_variance():
    dates = pd.date_range("2018-01-01", periods=24, freq="MS")
    monthly = pd.DataFrame({
        "date": dates,
        "price": np.linspace(1.8, 2.2, 24),
        "landings_kg": [1000.0] * 24,  # zero variance
    })
    with pytest.raises(ValueError):
        fit_level_volume_model(monthly, train_start=None, train_end=pd.Timestamp("2020-01-01"))


def test_score_mape_returns_none_when_no_valid_rows():
    predicted = pd.Series([np.nan, np.nan])
    actual = pd.Series([np.nan, np.nan])
    assert score_mape(predicted, actual) is None


def test_score_mape_computes_mean_absolute_percentage_error():
    predicted = pd.Series([1.1, 2.2])
    actual = pd.Series([1.0, 2.0])
    mape = score_mape(predicted, actual)
    assert mape == pytest.approx(10.0, rel=1e-6)


def test_build_complete_monthly_series_fills_gaps_with_nan():
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-03-01"]),
        "price": [2.0, 2.5],
        "landings_kg": [1000.0, 1200.0],
    })
    monthly = build_complete_monthly_series(rows)

    assert len(monthly) == 3  # Jan, Feb (gap), Mar
    assert monthly.iloc[1]["date"] == pd.Timestamp("2020-02-01")
    assert pd.isna(monthly.iloc[1]["price"])
    assert pd.isna(monthly.iloc[1]["landings_kg"])


def test_build_complete_monthly_series_raises_on_empty_input():
    with pytest.raises(ValueError):
        build_complete_monthly_series(pd.DataFrame(columns=["date", "price", "landings_kg"]))


def test_fit_level_only_model_ignores_volume_entirely():
    # Zero volume variance would make fit_level_volume_model raise, but
    # the level-only fallback only needs price.
    dates = pd.date_range("2018-01-01", periods=24, freq="MS")
    monthly = pd.DataFrame({
        "date": dates,
        "price": [2.0, 2.1] * 12,
        "landings_kg": [1000.0] * 24,
    })
    params = fit_level_only_model(monthly, train_start=None, train_end=pd.Timestamp("2020-01-01"))
    assert params["alpha"] == 0.0
    assert params["beta"] == 0.0

    predicted = predict_level_only(monthly, params)
    assert predicted.notna().all()


def test_fit_level_only_model_raises_on_zero_price_data():
    monthly = pd.DataFrame({
        "date": pd.date_range("2018-01-01", periods=3, freq="MS"),
        "price": [np.nan, np.nan, np.nan],
        "landings_kg": [1000.0, 1000.0, 1000.0],
    })
    with pytest.raises(ValueError):
        fit_level_only_model(monthly, train_start=None, train_end=pd.Timestamp("2018-04-01"))


def _make_trending_monthly(n_years=10, slope=0.1, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_years * 12, freq="MS")
    years = dates.year.to_numpy(dtype=float)
    price = 2.0 + slope * (years - years.min()) + rng.normal(0, 0.01, len(dates))
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": [1000.0] * len(dates)})


def test_fit_trend_rate_model_measures_a_positive_rate_from_rising_blocks():
    monthly = _make_trending_monthly(n_years=10, slope=0.1, seed=0)
    train_end = pd.Timestamp("2025-01-01")

    params = fit_trend_rate_model(monthly, train_end)

    assert params["rate"] == pytest.approx(0.0397, abs=0.001)
    assert params["years_trained"] == pytest.approx(9.0, abs=0.01)


def test_fit_trend_rate_model_returns_zero_rate_with_fewer_than_two_blocks():
    # 5 years of history with 3-year blocks gives one full block (years
    # 0-3) plus a 2-year remainder, too short for a second block -- a
    # line needs >=2 points, so this yields rate 0.0 rather than raising.
    monthly = _make_trending_monthly(n_years=5, slope=0.1, seed=0)
    train_end = pd.Timestamp("2020-01-01")

    params = fit_trend_rate_model(monthly, train_end)

    assert params["rate"] == 0.0
    # years_trained reflects the span actually used to measure the rate,
    # not the full available history -- with fewer than 2 usable blocks,
    # no rate was measured, so it's 0.0 here, not the 5-year span of
    # available-but-unused history.
    assert params["years_trained"] == 0.0


def test_fit_trend_rate_model_handles_empty_history():
    monthly = pd.DataFrame({"date": pd.to_datetime([]), "price": [], "landings_kg": []})
    params = fit_trend_rate_model(monthly, pd.Timestamp("2020-01-01"))
    assert params == {"rate": 0.0, "years_trained": 0.0}


def test_predict_trend_correction_is_identity_at_and_before_train_end():
    monthly = _make_trending_monthly(n_years=10, slope=0.1, seed=0)
    train_end = pd.Timestamp("2025-01-01")
    rate_params = fit_trend_rate_model(monthly, train_end)
    params = {
        "trend_rate": rate_params["rate"],
        "trend_years_trained": rate_params["years_trained"],
        "trend_confidence": 1.0,
        "trend_train_end": train_end,
    }

    at_or_before = pd.DataFrame({
        "date": [train_end, train_end - pd.DateOffset(years=3)],
        "price": [None, None],
        "landings_kg": [1000.0, 1000.0],
    })
    correction = predict_trend_correction(at_or_before, params)
    assert correction.iloc[0] == pytest.approx(1.0)
    assert correction.iloc[1] == pytest.approx(1.0)


def test_predict_trend_correction_grows_forward_and_clamps_at_years_trained():
    monthly = _make_trending_monthly(n_years=10, slope=0.1, seed=0)
    train_end = pd.Timestamp("2025-01-01")
    rate_params = fit_trend_rate_model(monthly, train_end)
    params = {
        "trend_rate": rate_params["rate"],
        "trend_years_trained": rate_params["years_trained"],
        "trend_confidence": 1.0,
        "trend_train_end": train_end,
    }

    one_year_out = pd.DataFrame(
        {"date": [train_end + pd.DateOffset(years=1)], "price": [None], "landings_kg": [1000.0]}
    )
    far_future = pd.DataFrame(
        {"date": [train_end + pd.DateOffset(years=20)], "price": [None], "landings_kg": [1000.0]}
    )
    at_years_trained_bound = pd.DataFrame(
        {
            # Round the day offset up to a whole day: trend_years is computed
            # via Timedelta.dt.days, which truncates to whole days, so a
            # fractional-day offset here would land just short of the clamp
            # bound and no longer compare bit-for-bit equal to the
            # far-future (already-clamped) case below.
            "date": [
                train_end
                + pd.Timedelta(days=int(np.ceil(rate_params["years_trained"] * 365.25)))
            ],
            "price": [None],
            "landings_kg": [1000.0],
        }
    )

    corr_1yr = predict_trend_correction(one_year_out, params).iloc[0]
    corr_far = predict_trend_correction(far_future, params).iloc[0]
    corr_bound = predict_trend_correction(at_years_trained_bound, params).iloc[0]

    assert corr_1yr > 1.0  # positive rate -> correction grows forward
    assert corr_far == pytest.approx(corr_bound, rel=1e-6)  # clamped, doesn't keep extrapolating


def test_predict_trend_correction_is_a_no_op_at_zero_confidence():
    monthly = _make_trending_monthly(n_years=10, slope=0.1, seed=0)
    train_end = pd.Timestamp("2025-01-01")
    rate_params = fit_trend_rate_model(monthly, train_end)
    params = {
        "trend_rate": rate_params["rate"],
        "trend_years_trained": rate_params["years_trained"],
        "trend_confidence": 0.0,
        "trend_train_end": train_end,
    }

    far_future = pd.DataFrame(
        {"date": [train_end + pd.DateOffset(years=5)], "price": [None], "landings_kg": [1000.0]}
    )
    assert predict_trend_correction(far_future, params).iloc[0] == pytest.approx(1.0)


def test_with_month_column_derives_month_without_mutating_the_original():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-15", "2024-12-01"]), "price": [1.0, 2.0]})

    result = with_month_column(df)

    assert result["month"].tolist() == [1, 12]
    assert "month" not in df.columns  # original untouched, since with_month_column copies
