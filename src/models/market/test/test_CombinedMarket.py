import logging

import numpy as np
import pandas as pd
import pytest

from src.models.market.CombinedMarket import CombinedMarket


def _synthetic_monthly(n_years=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    volume = 1000.0 + rng.normal(0, 50, len(dates))
    price = 2.0 - 0.001 * (volume - 1000.0) + rng.normal(0, 0.01, len(dates))
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})


def test_fit_selects_and_freezes_a_window_per_key():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}

    selected = market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    assert selected[("ANE", "Fresh - Whole")] is not None
    assert ("ANE", "Fresh - Whole") in market.models


def test_get_price_uses_real_data_before_frozen_model(caplog):
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    market.load_price_time_series(pd.DataFrame({
        "date": pd.to_datetime(["2024-02-01"]),
        "market_code": ["ES"],
        "species_code": ["ANE"],
        "category_code": ["Fresh - Whole"],
        "price": [9.99],
        "size": [None],
    }))

    price = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=1000.0)
    assert price == 9.99


def test_get_price_falls_back_to_frozen_model_when_no_real_price():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    price = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2024-03-01"), volume=1000.0)
    assert price is not None
    assert price == pytest.approx(2.0, abs=0.5)


def test_load_price_time_series_excludes_per_size_rows():
    market = CombinedMarket("ES")
    market.load_price_time_series(pd.DataFrame({
        "date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
        "market_code": ["ES", "ES"],
        "species_code": ["ANE", "ANE"],
        "category_code": ["Fresh - Whole", "Fresh - Whole"],
        "price": [2.0, 99.0],
        "size": [None, "1"],
    }))

    assert market.price_time_series[("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"))] == 2.0


def test_get_price_returns_none_for_unknown_species():
    market = CombinedMarket("ES")
    price = market.get_price("ZZZ", "Fresh - Whole", pd.Timestamp("2024-03-01"), volume=1000.0)
    assert price is None


def test_get_baseline_price_table_returns_frozen_static_table():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    table = market.get_baseline_price_table("ANE", "Fresh - Whole")
    assert table is not None
    assert set(table.index) == set(range(1, 13))


def test_fit_falls_back_to_level_only_then_excludes_on_total_failure(caplog):
    market = CombinedMarket("ES")
    # Only 2 months of data: not enough for level-volume (needs variance +
    # >=2 points) but also not even enough to try fitting meaningfully --
    # construct a case where level-volume fails every window but
    # level-only (price-only) still succeeds.
    dates = pd.date_range("2023-01-01", periods=2, freq="MS")
    monthly = pd.DataFrame({"date": dates, "price": [2.0, 2.1], "landings_kg": [1000.0, 1000.0]})

    with caplog.at_level(logging.WARNING):
        selected = market.fit({("THIN", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2023-03-01"))

    assert ("THIN", "Fresh - Whole") in market.models
    assert any("level-only" in r.message or "fallback" in r.message for r in caplog.records)


def test_fit_excludes_species_with_zero_usable_data(caplog):
    market = CombinedMarket("ES")
    monthly = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2, freq="MS"),
        "price": [np.nan, np.nan],
        "landings_kg": [np.nan, np.nan],
    })

    with caplog.at_level(logging.WARNING):
        selected = market.fit({("GHOST", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2023-03-01"))

    assert selected[("GHOST", "Fresh - Whole")] is None
    assert ("GHOST", "Fresh - Whole") not in market.models


def test_update_volume_baseline_is_incremental_running_mean():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    before = market.models[("ANE", "Fresh - Whole")]["baseline_volume"][1]  # January baseline
    market.update_volume_baseline("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=5000.0)
    after = market.models[("ANE", "Fresh - Whole")]["baseline_volume"][1]

    assert after != before
    assert after > before  # one large new observation should pull the mean up


def test_record_history_appends_a_snapshot():
    market = CombinedMarket("ES")
    market.record_history(tick=0, prices={("ANE", "Fresh - Whole"): 2.0}, volumes={("ANE", "Fresh - Whole"): 1000.0})
    assert len(market.price_history) == 1
    assert market.price_history[0][0] == 0


def test_direct_catch_accumulator_for_sales_routed_straight_to_the_prefix():
    market = CombinedMarket("FR")
    market.add_direct_catch("ANE", "Fresh - Whole", 100.0)
    market.add_direct_catch("ANE", "Fresh - Whole", 25.0)

    assert market.get_direct_catch_snapshot()[("ANE", "Fresh - Whole")] == 125.0

    market.clear_direct_catch()
    assert market.get_direct_catch_snapshot() == {}
