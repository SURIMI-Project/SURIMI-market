import logging

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from src.models.market.LocalMarket import LocalMarket, Step2FitInputs


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


def test_fit_step1_selects_window_and_stores_historical_series():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}

    selected = port.fit_step1(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    assert selected[("ANE", "Fresh - Whole")] is not None
    series = port.get_step1_historical_series("ANE", "Fresh - Whole")
    assert series is not None
    assert len(series) == len(monthly_by_key[("ANE", "Fresh - Whole")])


def test_predict_step1_uses_frozen_coefficients():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    port.fit_step1(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    price = port.predict_step1("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=500.0)
    assert price == pytest.approx(2.5, abs=0.5)


def test_predict_step1_returns_none_for_unknown_species():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    assert port.predict_step1("ZZZ", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=500.0) is None


def test_fit_step1_falls_back_and_excludes_same_as_combined_market(caplog):
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    monthly = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2, freq="MS"),
        "price": [np.nan, np.nan],
        "landings_kg": [np.nan, np.nan],
    })

    with caplog.at_level(logging.WARNING):
        selected = port.fit_step1({("GHOST", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2023-03-01"))

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


def test_predict_step1_returns_none_for_month_absent_from_thin_fallback_training():
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

    selected = port.fit_step1({("ANE", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2023-03-01"))
    assert selected[("ANE", "Fresh - Whole")] == "level-only-fallback"

    baseline_price = port.step1_models[("ANE", "Fresh - Whole")]["baseline_price"]
    assert set(baseline_price.index) == {1, 2}  # confirms the thin/sparse shape, same as CombinedMarket's finding

    price_for_known_month = port.predict_step1("ANE", "Fresh - Whole", pd.Timestamp("2023-02-01"), volume=500.0)
    assert price_for_known_month == pytest.approx(2.1, abs=0.5)

    price_for_absent_month = port.predict_step1("ANE", "Fresh - Whole", pd.Timestamp("2023-03-01"), volume=500.0)
    assert price_for_absent_month is None


def test_real_price_first_pattern_falls_through_on_nan_when_caller_guards_with_notna():
    """Bug class 2 (NaN guard): LocalMarket splits real-price-first into
    get_real_price (raw lookup) and predict_step1 (frozen-model prediction),
    deferring the combining to MarketSystem.step(). Locks in the contract that
    makes that safe: get_real_price returns a stored NaN as-is (not swallowed to
    None), so the caller must guard with pd.notna(), not `is not None`."""
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    port.fit_step1(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    # Simulate a loaded blended row with a missing/NaN price for this date (as
    # load_price_time_series would store verbatim for such a row).
    port.price_time_series[("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"))] = float("nan")

    real_price = port.get_real_price("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"))
    assert real_price is not None  # a stored value was found (not a missing-key None)
    assert pd.isna(real_price)  # ...but it is not usable

    # The correct real-price-first caller pattern: guard with pd.notna(), not `is not None`.
    price = real_price if pd.notna(real_price) else port.predict_step1(
        "ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=500.0
    )
    assert price == pytest.approx(2.5, abs=0.5)


def _synthetic_port_neighbor_anchor(seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    baseline_price = pd.Series({m: 2.0 for m in range(1, 13)})

    from src.models.market.stacked_model import build_static_anchor_series
    anchor_series = build_static_anchor_series(dates, baseline_price)
    neighbor_signal = pd.Series(2.0 + rng.normal(0, 0.3, len(dates)), index=dates)

    volume = 500.0 + rng.normal(0, 40, len(dates))
    price = (
        2.0 - 0.001 * (volume - 500.0) + 0.5 * (neighbor_signal.to_numpy() - 2.0)
        + rng.normal(0, 0.02, len(dates))
    )
    port_monthly = pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})
    return port_monthly, anchor_series, neighbor_signal


def test_fit_step2_selects_a_window_and_freezes_both_submodels():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    port.fit_step1({("ANE", "Fresh - Whole"): _synthetic_monthly()}, latest_date=pd.Timestamp("2024-01-01"))

    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()
    label = port.fit_step2(
        "ANE", "Fresh - Whole",
        Step2FitInputs(
            port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
            latest_date=pd.Timestamp("2024-01-01"),
        ),
    )

    assert label is not None
    assert ("ANE", "Fresh - Whole") in port.step2_stacked_models
    assert ("ANE", "Fresh - Whole") in port.step2_distance_only_models


def test_predict_step2_picks_stacked_when_volume_ratio_in_normal_range():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    port.fit_step1({("ANE", "Fresh - Whole"): _synthetic_monthly()}, latest_date=pd.Timestamp("2024-01-01"))
    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()
    port.fit_step2(
        "ANE", "Fresh - Whole",
        Step2FitInputs(
            port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
            latest_date=pd.Timestamp("2024-01-01"),
        ),
    )

    # Seed the baseline so VolumeRatio = 1.0 (normal) for this month.
    port.volume_baseline_stats[("ANE", "Fresh - Whole")] = {1: (500.0, 1)}

    price = port.predict_step2("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=500.0,
                                anchor_value=2.0, neighbor_value=2.0)
    assert price is not None


def test_predict_step2_falls_back_to_distance_only_on_extreme_volume_ratio():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    port.fit_step1({("ANE", "Fresh - Whole"): _synthetic_monthly()}, latest_date=pd.Timestamp("2024-01-01"))
    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()
    port.fit_step2(
        "ANE", "Fresh - Whole",
        Step2FitInputs(
            port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
            latest_date=pd.Timestamp("2024-01-01"),
        ),
    )
    port.volume_baseline_stats[("ANE", "Fresh - Whole")] = {1: (500.0, 1)}

    # Volume is 0 -> VolumeRatio = 0, below LOW_VOLUME_RATIO_THRESHOLD -> must use DistanceOnly.
    stacked_price = port.predict_step2("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=500.0,
                                        anchor_value=2.0, neighbor_value=2.0)
    distance_only_price = port.predict_step2("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=0.0,
                                              anchor_value=2.0, neighbor_value=2.0)
    assert distance_only_price is not None
    assert distance_only_price != stacked_price


def test_predict_step2_falls_back_to_distance_only_on_nan_volume():
    """Regression test for the pd.isna() fix in predict_step2's use_distance_only check:
    a NaN volume must route to DistanceOnly (matching the volume=0.0 extreme-ratio case),
    not silently fall through to the Stacked branch (which would NaN-propagate to None)."""
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    port.fit_step1({("ANE", "Fresh - Whole"): _synthetic_monthly()}, latest_date=pd.Timestamp("2024-01-01"))
    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()
    port.fit_step2(
        "ANE", "Fresh - Whole",
        Step2FitInputs(
            port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
            latest_date=pd.Timestamp("2024-01-01"),
        ),
    )
    port.volume_baseline_stats[("ANE", "Fresh - Whole")] = {1: (500.0, 1)}

    distance_only_price = port.predict_step2("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=0.0,
                                              anchor_value=2.0, neighbor_value=2.0)
    nan_price = port.predict_step2("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=float("nan"),
                                    anchor_value=2.0, neighbor_value=2.0)

    assert nan_price is not None
    assert nan_price == distance_only_price


def test_predict_step2_returns_none_when_no_step2_coefficients():
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    price = port.predict_step2("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=500.0,
                                anchor_value=2.0, neighbor_value=2.0)
    assert price is None


def test_fit_step2_falls_back_to_next_best_window_when_the_best_ones_final_refit_fails(monkeypatch):
    """Rolling-origin success doesn't guarantee the literal final window is
    fittable -- a real example (MUT@ESLLC) had its best-scoring window's final
    window fall mostly inside a real landings gap. fit_step2 must try the
    next-best-scoring candidate instead of giving up straight to Step-1-only."""
    import src.models.market.LocalMarket as local_market_module

    latest_date = pd.Timestamp("2024-01-01")
    port_monthly, anchor_series, neighbor_signal = _synthetic_port_neighbor_anchor()

    # Discover which window this fixture naturally selects as best, with no
    # monkeypatching in effect yet.
    baseline_port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    baseline_port.fit_step1({("ANE", "Fresh - Whole"): _synthetic_monthly()}, latest_date=latest_date)
    natural_label = baseline_port.fit_step2(
        "ANE", "Fresh - Whole",
        Step2FitInputs(
            port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
            latest_date=latest_date,
        ),
    )
    assert natural_label is not None  # fixture must have a stable window to make this test meaningful

    real_fit_stacked_model = local_market_module.fit_stacked_model

    def flaky_fit_stacked_model(port_monthly, anchor_series, neighbor_signal, train_start, train_end):
        # Fail only the natural best candidate's final re-fit (train_end ==
        # latest_date marks the final re-fit, not a rolling-origin scoring call);
        # every other call must keep succeeding normally.
        from src.models.market import consts
        years = consts.WINDOW_CANDIDATE_YEARS[natural_label]
        natural_train_start = latest_date - relativedelta(years=years) if years is not None else None
        if train_end == latest_date and train_start == natural_train_start:
            raise ValueError("simulated: this port's real landings gap falls inside exactly this window")
        return real_fit_stacked_model(port_monthly, anchor_series, neighbor_signal, train_start, train_end)

    monkeypatch.setattr(local_market_module, "fit_stacked_model", flaky_fit_stacked_model)

    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    port.fit_step1({("ANE", "Fresh - Whole"): _synthetic_monthly()}, latest_date=latest_date)
    label = port.fit_step2(
        "ANE", "Fresh - Whole",
        Step2FitInputs(
            port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
            latest_date=latest_date,
        ),
    )

    assert label is not None, "a fallback window should have been used instead of giving up to Step-1-only"
    assert label != natural_label
    assert ("ANE", "Fresh - Whole") in port.step2_stacked_models
    assert ("ANE", "Fresh - Whole") in port.step2_distance_only_models


def test_fit_step2_returns_none_when_no_window_is_stable(caplog):
    port = LocalMarket("ESALC", lon=-0.5, lat=38.3)
    dates = pd.date_range("2023-01-01", periods=2, freq="MS")
    port_monthly = pd.DataFrame({"date": dates, "price": [np.nan, np.nan], "landings_kg": [np.nan, np.nan]})
    anchor_series = pd.Series([2.0, 2.0], index=dates)
    neighbor_signal = pd.Series([2.0, 2.0], index=dates)

    with caplog.at_level(logging.WARNING):
        label = port.fit_step2(
            "GHOST", "Fresh - Whole",
            Step2FitInputs(
                port_monthly=port_monthly, anchor_series=anchor_series, neighbor_signal=neighbor_signal,
                latest_date=pd.Timestamp("2023-03-01"),
            ),
        )

    assert label is None
    assert ("GHOST", "Fresh - Whole") not in port.step2_stacked_models
