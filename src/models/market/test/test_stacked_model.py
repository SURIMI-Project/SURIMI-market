import numpy as np
import pandas as pd
import pytest

from src.models.market import consts
from src.models.market.StackedModel import (
    distance_weighted_neighbor_signal,
    compute_inverse_distance_weights,
)


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


def test_compute_inverse_distance_weights_keeps_only_the_k_nearest():
    # 7 candidates, all available -- only the TOP_K_NEAREST_DONORS (5) closest
    # to TARGET should survive, regardless of them all otherwise qualifying.
    source_ports = {"P1", "P2", "P3", "P4", "P5", "P6", "P7"}
    available_ports = {port: 1.0 for port in source_ports}
    distance_lookup = {("TARGET", port): float(i) + 1 for i, port in enumerate(
        ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
    )}

    weights = compute_inverse_distance_weights("TARGET", source_ports, available_ports, distance_lookup)

    assert len(weights) == consts.TOP_K_NEAREST_DONORS
    assert set(weights) == {"P1", "P2", "P3", "P4", "P5"}
    assert "P6" not in weights
    assert "P7" not in weights


def test_compute_inverse_distance_weights_nearest_k_chosen_before_availability_filter():
    # Nearest 5 by distance are P1..P5; P6 is 6th-nearest. P1 (the single
    # nearest) is unavailable. The tested behavior narrows to the nearest 5
    # FIRST, then drops unavailable ones from that set -- it must NOT
    # backfill with P6 to keep 5 candidates. If the implementation instead
    # filtered by availability before ranking by distance, P6 would appear
    # in the result -- this test would then fail.
    source_ports = {"P1", "P2", "P3", "P4", "P5", "P6"}
    distance_lookup = {("TARGET", port): float(i) + 1 for i, port in enumerate(
        ["P1", "P2", "P3", "P4", "P5", "P6"]
    )}
    available_ports = {"P2": 1.0, "P3": 1.0, "P4": 1.0, "P5": 1.0, "P6": 1.0}  # P1 missing

    weights = compute_inverse_distance_weights("TARGET", source_ports, available_ports, distance_lookup)

    assert set(weights) == {"P2", "P3", "P4", "P5"}
    assert "P6" not in weights  # must not be backfilled in place of unavailable P1
    assert "P1" not in weights


def test_compute_inverse_distance_weights_handles_fewer_than_k_candidates():
    source_ports = {"P1", "P2", "P3"}
    available_ports = {port: 1.0 for port in source_ports}
    distance_lookup = {("TARGET", "P1"): 10.0, ("TARGET", "P2"): 20.0, ("TARGET", "P3"): 30.0}

    weights = compute_inverse_distance_weights("TARGET", source_ports, available_ports, distance_lookup)

    assert set(weights) == {"P1", "P2", "P3"}
    assert weights["P1"] == pytest.approx(1.0 / 10.0)


from src.models.market.StackedModel import (
    build_static_anchor_series,
    fit_stacked_model,
    predict_stacked_model,
)


def _synthetic_port_and_neighbor(seed=0, beta_true=-0.3, gamma_true=0.5):
    """Generated directly in the new model's own log-space shape --
    log(price) - log(anchor) = beta*vol_rel_anom + gamma*neighbor_anom +
    noise -- so recovering beta_true/gamma_true is an exact check of
    fit_stacked_model's formula, not just an approximate one (as it would
    be if price were generated additively in linear space, the old
    model's shape)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    baseline_price = pd.Series({m: 2.0 for m in range(1, 13)})
    anchor_series = build_static_anchor_series(dates, baseline_price)

    neighbor_signal = pd.Series(2.0 + rng.normal(0, 0.3, len(dates)), index=dates)
    volume = 500.0 + rng.normal(0, 40, len(dates))

    vol_rel_anom = (volume - 500.0) / 500.0
    neighbor_anom = neighbor_signal.to_numpy() - 2.0
    noise = rng.normal(0, 0.02, len(dates))

    price = anchor_series.to_numpy() * np.exp(beta_true * vol_rel_anom + gamma_true * neighbor_anom + noise)
    port_monthly = pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})
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

    assert params["gamma"] == pytest.approx(0.5, abs=0.1)  # constructed sensitivity to neighbor
    assert params["beta"] == pytest.approx(-0.3, abs=0.1)  # constructed sensitivity to own volume
    assert "vol_anom_min" in params and "vol_anom_max" in params
    assert params["vol_anom_min"] < 0 < params["vol_anom_max"]

    predicted = predict_stacked_model(port_monthly, anchor_series, neighbor_signal, params)
    train_mask = port_monthly["date"] < train_end
    from src.models.market.level_volume_model import score_mape
    mape = score_mape(predicted[train_mask], port_monthly["price"][train_mask])
    assert mape < 5.0


def test_predict_stacked_model_returns_nan_when_port_baseline_volume_is_zero():
    """Same guard as level_volume_model.predict_level_volume: a calendar
    month with a legitimate zero port_baseline_volume (e.g. a recorded
    closed season every year in training) must not silently produce a
    confident-looking vol_anom_max prediction. Dividing by a zero
    baseline is undefined -- NaN, not +-inf clipped into looking real."""
    params = {
        "delta": 0.0,
        "beta": -0.3,
        "gamma": 0.5,
        "port_baseline_volume": pd.Series({1: 0.0}),
        "neighbor_baseline": pd.Series({1: 2.0}),
        "vol_anom_min": -0.5,
        "vol_anom_max": 0.5,
    }
    dates = pd.to_datetime(["2024-01-01"])
    port_monthly = pd.DataFrame({"date": dates, "price": [np.nan], "landings_kg": [100.0]})
    anchor_series = pd.Series([2.0], index=dates)
    neighbor_signal = pd.Series([2.0], index=dates)

    predicted = predict_stacked_model(port_monthly, anchor_series, neighbor_signal, params)

    assert pd.isna(predicted.iloc[0])


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


def test_predict_stacked_model_never_predicts_a_non_positive_price():
    """Log-space guarantees a positive prediction by construction (exp(...)
    is always > 0) -- unlike the old linear-space formula, which needed a
    separate sanity-check/retry safety net to catch this. Confirms it holds
    even for an extreme, far-outside-training volume that would have driven
    a linear-space additive formula negative."""
    port_monthly, anchor_series, neighbor_signal = _synthetic_port_and_neighbor()
    train_end = pd.Timestamp("2023-01-01")
    params = fit_stacked_model(port_monthly, anchor_series, neighbor_signal, None, train_end)

    extreme_dates = pd.date_range("2023-01-01", periods=3, freq="MS")
    extreme_monthly = pd.DataFrame({
        "date": extreme_dates,
        "price": [np.nan] * 3,
        "landings_kg": [1e8, 0.0, -1e6],  # wildly outside the observed training range
    })
    extreme_anchor = build_static_anchor_series(extreme_dates, pd.Series({m: 2.0 for m in range(1, 13)}))
    extreme_neighbor = pd.Series([2.0, 2.0, 2.0], index=extreme_dates)

    predicted = predict_stacked_model(extreme_monthly, extreme_anchor, extreme_neighbor, params)
    assert predicted.notna().all()
    assert (predicted > 0).all()
