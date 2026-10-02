import logging

import numpy as np
import pandas as pd
import pytest

from src.models.market.LocalMarket import LocalMarket


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


# --- Regression tests added per review guidance carried over from Task 5 (CombinedMarket) ---
#
# Task 5's first review round caught two recurring bug classes in the fit+fallback-cascade
# pattern that Task 6 (LocalMarket) shares structurally with Task 5 (CombinedMarket):
#   1. A frozen per-month seasonal-baseline table must be reliably indexed by all 12 months,
#      even on the level-only fallback path.
#   2. Any "is there a real/valid value" check must guard with pd.notna(), not just `is not None`
#      (a stored NaN is not None, but it is not usable).
#
# For LocalMarket (this task), neither bug class has a live code path to exploit *within this
# file* -- see task-6-report.md for the full analysis -- but the tests below lock in the
# NaN-safe behavior that makes that true, so a future refactor cannot silently reintroduce
# either bug class.


def test_predict_step1_returns_none_for_month_absent_from_thin_fallback_training():
    """Bug class 1 (per-month table coverage), adapted to what LocalMarket actually does with
    its per-month table: LocalMarket never exposes the raw baseline_price/baseline_volume
    Series to a caller (unlike CombinedMarket.get_baseline_price_table, which the Task 5 review
    found needed an explicit .reindex(range(1, 13)) fix) -- it only ever consumes them via
    pandas' NaN-safe `.map()` inside predict_level_only/predict_level_volume. This test proves
    that NaN-safety holds on the thin (2-month) fallback path: February was never present in the
    fallback's Jan-only... (here: a 2-month Jan/Feb window, so we probe March, which is genuinely
    absent) training window, so the frozen level-only baseline_price table only has entries for
    months 1 and 2. Predicting for a March date must come back as None (via pd.notna), not raise
    KeyError and not fabricate a value.
    """
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
    """Bug class 2 (NaN guard), adapted to LocalMarket's actual split design: unlike
    CombinedMarket.get_price (one method doing real-price-first internally), LocalMarket splits
    this into get_real_price (raw lookup) and predict_step1 (frozen-model prediction) -- per the
    plan's own section 8 note, the combining is deferred to MarketSystem.step() (Task 13), not
    done inside LocalMarket. This test documents and locks in the contract that makes that safe:
    get_real_price returns a stored NaN as-is (not swallowed to None), and the correct
    real-price-first caller pattern (pd.notna(), not `is not None`) is what must be used on top
    of it -- mirroring CombinedMarket.get_price's fix at the call site this method feeds.
    """
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
