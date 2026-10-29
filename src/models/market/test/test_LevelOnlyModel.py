import numpy as np
import pandas as pd
import pytest

from src.models.market.price_models import LevelOnlyModel, PriceModel, TrendCorrection
from src.models.market.level_volume_model import fit_level_only_model, predict_level_only


def _make_monthly(n_years=4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    volume = 1000.0 + rng.normal(0, 50, len(dates))
    price = 2.0 + rng.normal(0, 0.01, len(dates))
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})


def test_fit_and_predict_match_the_free_functions_exactly():
    monthly = _make_monthly()
    train_end = pd.Timestamp("2021-06-01")
    trend = TrendCorrection.identity(train_end)

    model = LevelOnlyModel.fit(monthly, None, train_end, trend)
    expected_params = fit_level_only_model(monthly, None, train_end)
    expected_predicted = predict_level_only(monthly, expected_params)

    actual_predicted = model.predict_series(monthly)

    np.testing.assert_allclose(actual_predicted.to_numpy(), expected_predicted.to_numpy())


def test_predict_matches_predict_series_for_a_single_row():
    monthly = _make_monthly()
    train_end = pd.Timestamp("2021-06-01")
    trend = TrendCorrection.identity(train_end)

    model = LevelOnlyModel.fit(monthly, None, train_end, trend)

    date = pd.Timestamp("2021-07-01")
    volume = 1000.0
    scalar_result = model.predict(date, volume)

    single_row = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
    series_result = model.predict_series(single_row).iloc[0]

    assert scalar_result == pytest.approx(series_result)


def test_update_baseline_volume_default_noop_does_not_raise():
    monthly = _make_monthly()
    train_end = pd.Timestamp("2021-06-01")
    trend = TrendCorrection.identity(train_end)

    model = LevelOnlyModel.fit(monthly, None, train_end, trend)
    assert isinstance(model, PriceModel)
    model.update_baseline_volume(1, 12345.0)  # must not raise, must not set anything meaningful
    assert model.baseline_volume is None
