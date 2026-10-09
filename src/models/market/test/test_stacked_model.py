import numpy as np
import pandas as pd
import pytest

from src.models.market import consts
from src.models.market.stacked_model import (
    should_use_distance_only,
    distance_weighted_neighbor_signal,
    compute_inverse_distance_weights,
)


def test_should_use_distance_only_true_when_ratio_missing_or_extreme():
    assert should_use_distance_only(float("nan")) == True
    assert should_use_distance_only(0.05) == True  # below LOW_VOLUME_RATIO_THRESHOLD (0.10)
    assert should_use_distance_only(3.0) == True  # above HIGH_VOLUME_RATIO_THRESHOLD (2.0)
    assert should_use_distance_only(1.0) == False  # normal range


def test_should_use_distance_only_works_on_a_vectorized_series():
    import pandas as pd
    ratios = pd.Series([float("nan"), 0.05, 1.0, 3.0])
    result = should_use_distance_only(ratios)
    assert result.tolist() == [True, True, False, True]


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
