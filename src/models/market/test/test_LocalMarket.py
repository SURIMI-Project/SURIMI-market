import logging

import numpy as np
import pandas as pd
import pytest

from src.models.market.LocalMarket import LocalMarket, NeighborFitInputs


def _synthetic_monthly(n_years=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    volume = 500.0 + rng.normal(0, 30, len(dates))
    price = 2.5 - 0.002 * (volume - 500.0) + rng.normal(0, 0.01, len(dates))
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})


def test_local_market_holds_lon_lat_once():
    port = LocalMarket("ESALC", name="Alicante", lon=-0.5, lat=38.3)
    assert port.lon == -0.5
    assert port.lat == 38.3


def test_fit_base_models_selects_window_and_stores_historical_series():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}

    selected = port.fit_base_models(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    assert selected[("ANE", "Fresh - Whole")] is not None
    series = port.get_base_predicted_series("ANE", "Fresh - Whole")
    assert series is not None
    assert len(series) == len(monthly_by_key[("ANE", "Fresh - Whole")])


def test_predict_from_base_model_uses_frozen_coefficients():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    port.fit_base_models(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    price = port.predict_from_base_model("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=500.0)
    assert price == pytest.approx(2.5, abs=0.5)


def test_predict_from_base_model_returns_none_for_unknown_species():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    assert port.predict_from_base_model("ZZZ", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=500.0) is None


def test_fit_base_models_falls_back_and_excludes_same_as_combined_market(caplog):
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    monthly = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2, freq="MS"),
        "price": [np.nan, np.nan],
        "landings_kg": [np.nan, np.nan],
    })

    with caplog.at_level(logging.WARNING):
        selected = port.fit_base_models({("GHOST", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2023-03-01"))

    assert selected[("GHOST", "Fresh - Whole")] is None


def test_catch_accumulator_reused_from_elsa_pattern():
    port = LocalMarket("ESALC")
    port.add_monthly_catch("ANE", "Fresh - Whole", 100.0)
    port.add_monthly_catch("ANE", "Fresh - Whole", 50.0)

    snapshot = port.get_monthly_catch_snapshot()
    assert snapshot[("ANE", "Fresh - Whole")] == 150.0

    port.clear_monthly_catch()
    assert port.get_monthly_catch_snapshot() == {}


def test_load_price_time_series_filters_to_this_port_and_blended_rows():
    port = LocalMarket("ESALC")
    port.load_price_time_series(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
        "market_code": ["ESALC", "ESALC", "ESBCN"],
        "species_code": ["ANE", "ANE", "ANE"],
        "category_code": ["Fresh - Whole"] * 3,
        "price": [2.0, 99.0, 5.0],
        "size": [None, "1", None],
    }))

    assert port.get_real_price("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01")) == 2.0


def test_record_history_appends_a_snapshot():
    port = LocalMarket("ESALC")
    port.record_history(tick=0, prices={("ANE", "Fresh - Whole"): 2.0}, catches={("ANE", "Fresh - Whole"): 150.0})
    assert len(port.price_history) == 1


# --- Regression tests for two recurring bug classes in fit+fallback-cascade code ---
#
# 1. A frozen per-month seasonal-baseline table must be indexed by all 12 months,
#    even on the level-only fallback path.
# 2. Any "is there a valid value" check must guard with pd.notna(), not `is not None`
#    (a stored NaN is not None, but it is not usable).
#
# Neither bug currently has a live path to exploit in this file, but these tests
# lock in the NaN-safe behavior so a refactor can't silently reintroduce either.


def test_predict_from_base_model_returns_none_for_month_absent_from_thin_fallback_training():
    """Bug class 1 (per-month table coverage): LocalMarket never exposes the raw
    baseline_price/baseline_volume Series to a caller, only consumes them via
    pandas' NaN-safe `.map()` inside predict_level_only/predict_level_volume.
    Proves NaN-safety holds on the thin (2-month) fallback path: the frozen
    baseline_price table only has entries for months 1 and 2, so predicting for
    the absent March must return None (via pd.notna), not raise KeyError."""
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    monthly = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2, freq="MS"),  # Jan, Feb only
        "price": [2.0, 2.1],
        "landings_kg": [500.0, 510.0],
    })

    selected = port.fit_base_models({("ANE", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2023-03-01"))
    assert selected[("ANE", "Fresh - Whole")] == "level-only-fallback"

    baseline_price = port.models[("ANE", "Fresh - Whole")].baseline_price
    assert set(baseline_price.index) == {1, 2}  # confirms the thin/sparse shape, same as CombinedMarket's finding

    price_for_known_month = port.predict_from_base_model(
        "ANE", "Fresh - Whole", pd.Timestamp("2023-02-01"), volume=500.0
    )
    assert price_for_known_month == pytest.approx(2.1, abs=0.5)

    price_for_absent_month = port.predict_from_base_model(
        "ANE", "Fresh - Whole", pd.Timestamp("2023-03-01"), volume=500.0
    )
    assert price_for_absent_month is None


def test_real_price_first_pattern_falls_through_on_nan_when_caller_guards_with_notna():
    """Bug class 2 (NaN guard): LocalMarket splits real-price-first into
    get_real_price (raw lookup) and predict_from_base_model (frozen-model
    prediction), deferring the combining to MarketSystem.step(). Locks in the
    contract that makes that safe: get_real_price returns a stored NaN as-is
    (not swallowed to None), so the caller must guard with pd.notna(), not
    `is not None`."""
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    port.fit_base_models(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    # Simulate a loaded blended row with a missing/NaN price for this date (as
    # load_price_time_series would store verbatim for such a row).
    port.load_price_time_series(pd.DataFrame({
        "date": pd.to_datetime(["2024-02-01"]),
        "market_code": ["ESALC"],
        "species_code": ["ANE"],
        "category_code": ["Fresh - Whole"],
        "price": [float("nan")],
        "size": [None],
    }))

    real_price = port.get_real_price("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"))
    assert real_price is not None  # a stored value was found (not a missing-key None)
    assert pd.isna(real_price)  # ...but it is not usable

    # The correct real-price-first caller pattern: guard with pd.notna(), not `is not None`.
    price = real_price if pd.notna(real_price) else port.predict_from_base_model(
        "ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=500.0
    )
    assert price == pytest.approx(2.5, abs=0.5)


def _synthetic_port_neighbor_anchor(seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    baseline_price = pd.Series({m: 2.0 for m in range(1, 13)})

    from src.models.market.StackedModel import build_static_anchor_series
    anchor_series = build_static_anchor_series(dates, baseline_price)
    neighbor_signal = pd.Series(2.0 + rng.normal(0, 0.3, len(dates)), index=dates)

    volume = 500.0 + rng.normal(0, 40, len(dates))
    price = (
        2.0 - 0.001 * (volume - 500.0) + 0.5 * (neighbor_signal.to_numpy() - 2.0)
        + rng.normal(0, 0.02, len(dates))
    )
    port_monthly = pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})
    return port_monthly, anchor_series, neighbor_signal


def test_fit_neighbor_models_reuses_the_base_models_window_label():
    """No independent rolling-origin search anymore -- fit_neighbor_models
    must reuse exactly the window label fit_base_models already chose for
    this key, not re-derive its own."""
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    base_selected = port.fit_base_models(
        {("ANE", "Fresh - Whole"): _synthetic_monthly()}, latest_date=pd.Timestamp("2024-01-01")
    )
    base_label = base_selected[("ANE", "Fresh - Whole")]
    assert base_label is not None and base_label != "level-only-fallback"

    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()
    label = port.fit_neighbor_models(
        "ANE", "Fresh - Whole",
        NeighborFitInputs(
            port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
            latest_date=pd.Timestamp("2024-01-01"),
        ),
    )

    assert label == base_label
    assert port.neighbor_window_labels[("ANE", "Fresh - Whole")] == base_label
    assert ("ANE", "Fresh - Whole") in port.neighbor_models
    model = port.neighbor_models[("ANE", "Fresh - Whole")]
    # Direct attributes now -- the old .stacked./.distance_only indirection
    # is gone, there's only ever the one StackedModel.
    assert isinstance(model.delta, float)
    assert isinstance(model.beta, float)
    assert isinstance(model.gamma, float)


def test_update_volume_baseline_also_updates_the_neighbor_models_own_baseline():
    """update_volume_baseline must keep StackedModel's own
    port_baseline_volume in sync with the base model's baseline_volume --
    both represent the same real-world quantity (this port's own observed
    monthly-average landings), and StackedModel's own vol_anom term is
    built to behave exactly like the base model's for this reason. If only
    the base model is patched, the neighbor model's volume anomaly keeps
    comparing against a stale, frozen-at-fit-time baseline forever."""
    key = ("ANE", "Fresh - Whole")
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    port.fit_base_models({key: _synthetic_monthly()}, latest_date=pd.Timestamp("2024-01-01"))

    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()
    port.fit_neighbor_models(
        key[0], key[1],
        NeighborFitInputs(
            port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
            latest_date=pd.Timestamp("2024-01-01"),
        ),
    )

    base_model = port.models[key]
    neighbor_model = port.neighbor_models[key]
    update_date = pd.Timestamp("2024-02-01")
    month = update_date.month
    base_before = base_model.baseline_volume[month]
    neighbor_before = neighbor_model.port_baseline_volume[month]

    new_volume = base_before + 10_000.0  # a large, unmistakable jump
    port.update_volume_baseline(key[0], key[1], update_date, new_volume)

    base_after = base_model.baseline_volume[month]
    neighbor_after = neighbor_model.port_baseline_volume[month]

    assert base_after != base_before, "sanity check: the base model's own baseline should have moved"
    assert neighbor_after == base_after, (
        "the neighbor model's own baseline must track the base model's exactly -- "
        f"base moved to {base_after}, neighbor stayed at {neighbor_after} (was {neighbor_before})"
    )


def test_fit_neighbor_models_returns_none_when_base_model_used_level_only_fallback(caplog):
    """No well-defined window to reuse when the base model itself fell
    back to the no-volume-term fit -- must fall back to the base model
    only, not attempt a neighbor fit at all."""
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    thin_monthly = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2, freq="MS"),
        "price": [2.0, 2.1],
        "landings_kg": [500.0, 510.0],
    })
    selected = port.fit_base_models({("ANE", "Fresh - Whole"): thin_monthly}, latest_date=pd.Timestamp("2023-03-01"))
    assert selected[("ANE", "Fresh - Whole")] == "level-only-fallback"

    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()
    with caplog.at_level(logging.WARNING):
        label = port.fit_neighbor_models(
            "ANE", "Fresh - Whole",
            NeighborFitInputs(
                port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
                latest_date=pd.Timestamp("2023-03-01"),
            ),
        )

    assert label is None
    assert ("ANE", "Fresh - Whole") not in port.neighbor_models


def test_fit_neighbor_models_returns_none_when_base_model_was_never_fit_for_this_key(caplog):
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()

    with caplog.at_level(logging.WARNING):
        label = port.fit_neighbor_models(
            "GHOST", "Fresh - Whole",
            NeighborFitInputs(
                port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
                latest_date=pd.Timestamp("2024-01-01"),
            ),
        )

    assert label is None
    assert ("GHOST", "Fresh - Whole") not in port.neighbor_models


def test_fit_neighbor_models_falls_back_to_base_model_only_when_the_single_refit_fails(monkeypatch):
    """Unlike the old design (which retried the next-best-scoring candidate
    window if the top one's final re-fit failed), there is only ever one
    candidate now -- the base model's own reused window -- so a re-fit
    failure here must fall straight back to the base model only, with no
    retry loop."""
    import src.models.market.StackedModel as stacked_model_module

    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    port.fit_base_models({("ANE", "Fresh - Whole"): _synthetic_monthly()}, latest_date=pd.Timestamp("2024-01-01"))
    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()

    def always_fails(*args, **kwargs):
        raise ValueError("simulated: this port's real landings gap falls inside the reused window")

    monkeypatch.setattr(stacked_model_module, "fit_stacked_model", always_fails)

    label = port.fit_neighbor_models(
        "ANE", "Fresh - Whole",
        NeighborFitInputs(
            port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
            latest_date=pd.Timestamp("2024-01-01"),
        ),
    )

    assert label is None
    assert ("ANE", "Fresh - Whole") not in port.neighbor_models


def test_predict_from_neighbor_model_returns_none_when_no_neighbor_model():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    price = port.predict_from_neighbor_model("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=500.0,
                                              anchor_value=2.0, neighbor_value=2.0)
    assert price is None


def test_predict_from_neighbor_model_stays_positive_for_extreme_volume_that_would_have_broken_the_old_linear_model():
    """The old neighbor-informed model was additive in raw price (anchor +
    delta + beta*vol_anom + gamma*neighbor_anom) with vol_anom computed as
    an UNCLIPPED, ABSOLUTE volume anomaly -- an extreme enough volume would
    drive that formula below zero, which is why the old fit_neighbor_models
    needed a whole sanity-check/retry safety net. The new log-space,
    clipped-relative-anomaly formula can't do that by construction: confirm
    a wildly extreme volume (orders of magnitude past anything seen in
    training, ~500kg) still produces a finite, positive price with no
    special-casing anywhere in this path."""
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    port.fit_base_models({("ANE", "Fresh - Whole"): _synthetic_monthly()}, latest_date=pd.Timestamp("2024-01-01"))
    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()
    port.fit_neighbor_models(
        "ANE", "Fresh - Whole",
        NeighborFitInputs(
            port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
            latest_date=pd.Timestamp("2024-01-01"),
        ),
    )
    assert ("ANE", "Fresh - Whole") in port.neighbor_models

    extreme_high_price = port.predict_from_neighbor_model(
        "ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=1_000_000.0,
        anchor_value=2.0, neighbor_value=2.0,
    )
    assert extreme_high_price is not None
    assert extreme_high_price > 0

    zero_volume_price = port.predict_from_neighbor_model(
        "ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=0.0,
        anchor_value=2.0, neighbor_value=2.0,
    )
    assert zero_volume_price is not None
    assert zero_volume_price > 0
