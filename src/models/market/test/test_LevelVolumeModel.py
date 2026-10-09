import numpy as np
import pandas as pd
import pytest

from src.models.market.LevelVolumeModel import LevelVolumeModel
from src.models.market.PriceModel import PriceModel
from src.models.market.TrendCorrection import TrendCorrection
from src.models.market.level_volume_model import fit_level_volume_model, predict_level_volume


def _make_monthly(n_years=5, base_price=2.0, base_volume=1000.0, vol_response=-0.001, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    volume = base_volume + rng.normal(0, 50, len(dates))
    price = base_price + vol_response * (volume - base_volume) + rng.normal(0, 0.01, len(dates))
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})


def test_fit_and_predict_match_the_free_functions_exactly():
    monthly = _make_monthly()
    train_end = pd.Timestamp("2023-01-01")
    trend = TrendCorrection.identity(train_end)

    model = LevelVolumeModel.fit(monthly, None, train_end, trend)
    expected_params = fit_level_volume_model(monthly, None, train_end)
    expected_predicted = predict_level_volume(monthly, expected_params)

    actual_predicted = model.predict_series(monthly)

    np.testing.assert_allclose(actual_predicted.to_numpy(), expected_predicted.to_numpy())


def test_predict_matches_predict_series_for_a_single_row():
    monthly = _make_monthly()
    train_end = pd.Timestamp("2023-01-01")
    trend = TrendCorrection.identity(train_end)

    model = LevelVolumeModel.fit(monthly, None, train_end, trend)

    date = pd.Timestamp("2023-02-01")
    volume = 1000.0
    scalar_result = model.predict(date, volume)

    single_row = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
    series_result = model.predict_series(single_row).iloc[0]

    assert scalar_result == pytest.approx(series_result)


def test_update_baseline_volume_mutates_in_place():
    monthly = _make_monthly()
    train_end = pd.Timestamp("2023-01-01")
    trend = TrendCorrection.identity(train_end)

    model = LevelVolumeModel.fit(monthly, None, train_end, trend)
    assert isinstance(model, PriceModel)

    before = model.baseline_volume[1]
    model.update_baseline_volume(1, before + 5000.0)
    after = model.baseline_volume[1]

    assert after == pytest.approx(before + 5000.0)
    assert after != before
