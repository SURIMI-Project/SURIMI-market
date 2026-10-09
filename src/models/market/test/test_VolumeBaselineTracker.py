import pandas as pd
import pytest

from src.models.market.VolumeBaselineTracker import VolumeBaselineTracker


def test_seed_then_ratio_reflects_training_data():
    monthly = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=24, freq="MS"),
        "price": [2.0] * 24,
        "landings_kg": [1000.0] * 24,
    })
    tracker = VolumeBaselineTracker()
    tracker.seed(monthly, train_start=None, train_end=pd.Timestamp("2022-01-01"))

    assert tracker.ratio(pd.Timestamp("2022-06-01"), volume=1000.0) == pytest.approx(1.0)
    assert tracker.ratio(pd.Timestamp("2022-06-01"), volume=500.0) == pytest.approx(0.5)


def test_ratio_is_nan_before_any_seed_or_update():
    tracker = VolumeBaselineTracker()
    assert pd.isna(tracker.ratio(pd.Timestamp("2022-01-01"), volume=1000.0))


def test_update_is_incremental_and_never_decaying():
    tracker = VolumeBaselineTracker()
    result1 = tracker.update(pd.Timestamp("2022-01-01"), volume=100.0)
    assert result1 == (1, 100.0)
    result2 = tracker.update(pd.Timestamp("2022-01-15"), volume=300.0)
    assert result2 == (1, 200.0)  # running mean of 100 and 300
    assert tracker.ratio(pd.Timestamp("2022-01-20"), volume=200.0) == pytest.approx(1.0)


def test_update_returns_none_for_nan_volume():
    tracker = VolumeBaselineTracker()
    tracker.update(pd.Timestamp("2022-01-01"), volume=100.0)
    assert tracker.update(pd.Timestamp("2022-01-15"), volume=float("nan")) is None
    # the NaN update must not have corrupted the existing baseline
    assert tracker.ratio(pd.Timestamp("2022-01-20"), volume=100.0) == pytest.approx(1.0)


def test_ratio_is_nan_for_a_month_with_no_observations():
    tracker = VolumeBaselineTracker()
    tracker.update(pd.Timestamp("2022-01-01"), volume=100.0)
    assert pd.isna(tracker.ratio(pd.Timestamp("2022-07-01"), volume=100.0))  # July never seeded/updated
