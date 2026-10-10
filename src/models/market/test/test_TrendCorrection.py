import numpy as np
import pandas as pd
import pytest

from src.models.market.price_models import TrendCorrection
from src.models.market.level_volume_model import predict_trend_correction


def _arbitrary_trend():
    return TrendCorrection(confidence=0.8, rate=0.05, years_trained=6.0, train_end=pd.Timestamp("2024-01-01"))


def test_factor_series_matches_predict_trend_correction_exactly():
    trend = _arbitrary_trend()
    dates = pd.to_datetime(["2023-06-01", "2024-01-01", "2025-01-01", "2035-01-01"])
    monthly = pd.DataFrame({"date": dates})
    params = {
        "trend_confidence": trend.confidence,
        "trend_rate": trend.rate,
        "trend_years_trained": trend.years_trained,
        "trend_train_end": trend.train_end,
    }

    expected = predict_trend_correction(monthly, params)
    actual = trend.factor_series(monthly["date"])

    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())


def test_factor_matches_factor_series_for_a_single_date():
    trend = _arbitrary_trend()
    date = pd.Timestamp("2025-06-01")
    monthly = pd.DataFrame({"date": [date]})

    scalar_result = trend.factor(date)
    series_result = trend.factor_series(monthly["date"]).iloc[0]

    assert scalar_result == pytest.approx(series_result)


def test_identity_is_always_a_noop():
    trend = TrendCorrection.identity(train_end=pd.Timestamp("2024-01-01"))
    assert trend.factor(pd.Timestamp("2024-01-01")) == pytest.approx(1.0)
    assert trend.factor(pd.Timestamp("2030-01-01")) == pytest.approx(1.0)
    assert trend.factor(pd.Timestamp("2015-01-01")) == pytest.approx(1.0)
