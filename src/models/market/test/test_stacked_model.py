import numpy as np
import pandas as pd
import pytest

from src.models.market.stacked_model import (
    compute_volume_shares,
    compute_tier,
    compute_high_tier_ports,
    distance_weighted_neighbor_signal,
)


def test_compute_volume_shares_sums_to_100():
    shares = compute_volume_shares({"A": 60.0, "B": 30.0, "C": 10.0})
    assert shares["A"] == pytest.approx(60.0)
    assert sum(shares.values()) == pytest.approx(100.0)


def test_compute_volume_shares_handles_zero_total():
    shares = compute_volume_shares({"A": 0.0, "B": 0.0})
    assert shares == {"A": 0.0, "B": 0.0}


def test_compute_tier_boundaries_match_right_inclusive_lower_bin():
    # Matches pd.cut(bins=[-0.01, 0.2, 2, 100]) semantics exactly:
    # exactly 0.2 -> Low, exactly 2.0 -> Mid, only > 2.0 -> High.
    assert compute_tier(0.2) == "Low"
    assert compute_tier(0.20001) == "Mid"
    assert compute_tier(2.0) == "Mid"
    assert compute_tier(2.00001) == "High"
    assert compute_tier(0.0) == "Low"


def test_compute_high_tier_ports_selects_only_high_tier():
    high_tier = compute_high_tier_ports({"A": 90.0, "B": 1.5, "C": 1.0})
    assert "A" in high_tier
    assert "B" not in high_tier
    assert "C" not in high_tier


def test_distance_weighted_neighbor_signal_inverse_weights_closer_ports_more():
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    predicted_series_by_port = {
        "NEAR": pd.Series([1.0, 1.0, 1.0], index=dates),
        "FAR": pd.Series([10.0, 10.0, 10.0], index=dates),
    }
    distance_lookup = {
        ("TARGET", "NEAR"): 1000.0,
        ("NEAR", "TARGET"): 1000.0,
        ("TARGET", "FAR"): 100000.0,
        ("FAR", "TARGET"): 100000.0,
    }

    signal = distance_weighted_neighbor_signal(
        "TARGET", {"NEAR", "FAR"}, predicted_series_by_port, distance_lookup
    )

    # NEAR is 100x closer, so its weight should dominate: result much closer to 1.0 than 10.0.
    assert signal.iloc[0] < 2.0


def test_distance_weighted_neighbor_signal_returns_none_with_no_valid_sources():
    signal = distance_weighted_neighbor_signal("TARGET", {"ONLY_SELF"}, {}, {})
    assert signal is None


def test_distance_weighted_neighbor_signal_excludes_target_from_its_own_sources():
    dates = pd.date_range("2020-01-01", periods=2, freq="MS")
    predicted_series_by_port = {"TARGET": pd.Series([5.0, 5.0], index=dates)}
    signal = distance_weighted_neighbor_signal("TARGET", {"TARGET"}, predicted_series_by_port, {})
    assert signal is None


def test_distance_weighted_neighbor_signal_handles_missing_dates_gracefully():
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    predicted_series_by_port = {
        "A": pd.Series([1.0, np.nan, 3.0], index=dates),
    }
    distance_lookup = {("TARGET", "A"): 1000.0, ("A", "TARGET"): 1000.0}

    signal = distance_weighted_neighbor_signal("TARGET", {"A"}, predicted_series_by_port, distance_lookup)

    assert signal.iloc[0] == pytest.approx(1.0)
    assert pd.isna(signal.iloc[1])  # no valid source for this date
    assert signal.iloc[2] == pytest.approx(3.0)


from src.models.market.stacked_model import (
    build_static_anchor_series,
    fit_stacked_model,
    predict_stacked_model,
    fit_distance_only_model,
    predict_distance_only_model,
)


def _synthetic_port_and_neighbor(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    baseline_price = pd.Series({m: 2.0 for m in range(1, 13)})

    neighbor_signal = pd.Series(2.0 + rng.normal(0, 0.3, len(dates)), index=dates)
    volume = 500.0 + rng.normal(0, 40, len(dates))
    # Port price responds to its own volume anomaly AND tracks the neighbor signal.
    price = (
        2.0
        - 0.001 * (volume - 500.0)
        + 0.5 * (neighbor_signal.to_numpy() - 2.0)
        + rng.normal(0, 0.02, len(dates))
    )
    port_monthly = pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})
    anchor_series = build_static_anchor_series(dates, baseline_price)
    return port_monthly, anchor_series, neighbor_signal


def test_build_static_anchor_series_broadcasts_month_table_across_dates():
    dates = pd.date_range("2020-01-01", periods=3, freq="MS")
    table = pd.Series({1: 1.0, 2: 2.0, 3: 3.0})
    anchor = build_static_anchor_series(dates, table)
    assert list(anchor.values) == [1.0, 2.0, 3.0]


def test_fit_and_predict_stacked_model_recovers_neighbor_sensitivity():
    port_monthly, anchor_series, neighbor_signal = _synthetic_port_and_neighbor()
    train_end = pd.Timestamp("2023-01-01")

    params = fit_stacked_model(port_monthly, anchor_series, neighbor_signal, None, train_end)

    assert params["gamma"] == pytest.approx(0.5, abs=0.2)  # constructed sensitivity to neighbor

    predicted = predict_stacked_model(port_monthly, anchor_series, neighbor_signal, params)
    train_mask = port_monthly["date"] < train_end
    from src.models.market.level_volume_model import score_mape
    mape = score_mape(predicted[train_mask], port_monthly["price"][train_mask])
    assert mape < 5.0


def test_fit_stacked_model_raises_on_insufficient_rows():
    port_monthly = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=2, freq="MS"),
        "price": [2.0, 2.1],
        "landings_kg": [500.0, 510.0],
    })
    anchor_series = pd.Series([2.0, 2.0], index=port_monthly["date"])
    neighbor_signal = pd.Series([2.0, 2.0], index=port_monthly["date"])

    with pytest.raises(ValueError):
        fit_stacked_model(port_monthly, anchor_series, neighbor_signal, None, pd.Timestamp("2020-03-01"))


def test_fit_and_predict_distance_only_model_tracks_neighbor_signal():
    port_monthly, anchor_series, neighbor_signal = _synthetic_port_and_neighbor()
    train_end = pd.Timestamp("2023-01-01")

    params = fit_distance_only_model(port_monthly, anchor_series, neighbor_signal, None, train_end)
    assert params["gamma"] != 0.0

    predicted = predict_distance_only_model(port_monthly, anchor_series, neighbor_signal, params)
    assert predicted.notna().sum() > 0


def test_fit_distance_only_model_raises_on_zero_variance_neighbor():
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    port_monthly = pd.DataFrame({"date": dates, "price": np.linspace(2.0, 2.5, 12), "landings_kg": [500.0] * 12})
    anchor_series = pd.Series([2.0] * 12, index=dates)
    neighbor_signal = pd.Series([3.0] * 12, index=dates)  # zero variance

    with pytest.raises(ValueError):
        fit_distance_only_model(port_monthly, anchor_series, neighbor_signal, None, pd.Timestamp("2021-01-01"))
