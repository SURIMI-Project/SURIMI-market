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
    # the level-only fallback (spec section 9 point 2) only needs price.
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
