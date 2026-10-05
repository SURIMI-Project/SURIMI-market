import datetime

import pandas as pd
from grpc import StatusCode

from src.models.market import consts
from src.models.market.MarketSystem import MarketSystem, build_prefix_to_locals


def _markets_df():
    return pd.DataFrame({
        "market_code": ["ES", "FR", "ESALC", "ESBCN"],
        "market_name": ["Spain", "France", "Alicante", "Barcelona"],
        "lon": [None, None, -0.5, 2.2],
        "lat": [None, None, 38.3, 41.4],
    })


def test_build_prefix_to_locals_groups_by_prefix():
    result = build_prefix_to_locals(_markets_df())
    assert set(result["ES"]) == {"ESALC", "ESBCN"}
    assert result["FR"] == []


def test_build_prefix_to_locals_infers_prefix_with_no_explicit_combined_row():
    markets_df = pd.DataFrame({
        "market_code": ["ESALC"], "market_name": ["Alicante"], "lon": [-0.5], "lat": [38.3],
    })
    result = build_prefix_to_locals(markets_df)
    assert set(result["ES"]) == {"ESALC"}


def test_setup_topology_instantiates_local_and_combined_markets():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    err = system.setup_topology(_markets_df())

    assert err is None
    assert set(system.local_markets.keys()) == {"ESALC", "ESBCN"}
    assert set(system.combined_markets.keys()) == {"ES", "FR"}
    assert system.local_markets["ESALC"].lon == -0.5


def test_setup_topology_computes_distance_lookup():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df())
    assert ("ESALC", "ESBCN") in system.distance_lookup


def test_setup_topology_errors_on_missing_local_coordinates():
    markets_df = pd.DataFrame({
        "market_code": ["ES", "ESALC", "ESBCN"],
        "market_name": ["Spain", "Alicante", "Barcelona"],
        "lon": [None, -0.5, None],
        "lat": [None, 38.3, None],
    })
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    err = system.setup_topology(markets_df)

    assert err is not None
    assert err.status_code == StatusCode.FAILED_PRECONDITION
    assert "ESBCN" in err.message


def test_setup_topology_does_not_error_on_combined_markets_missing_coordinates():
    # ES/FR having NA lon/lat is expected (spec section 4) -- only LOCAL
    # markets (5-letter codes) are checked.
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    err = system.setup_topology(_markets_df())
    assert err is None


from grpc import StatusCode

from surimi.v1 import sales_pb2, species_pb2


def _market_sales(market_code, species_code="ANE", category_code="Fresh - Whole", quantity=10.0):
    sale = sales_pb2.Sale(
        species=species_pb2.Species(species_code=species_code),
        category_code=category_code,
        quantity=quantity,
        value=20.0,
    )
    return [sales_pb2.MarketSales(market_code=market_code, currency="EUR", sales=[sale])]


def _system_with_topology():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df())
    return system


def _utc_epoch(dt: datetime.datetime) -> int:
    """Encode a (possibly naive, semantically-UTC per Tick.reset()'s utcnow())
    datetime as a genuine UTC epoch int -- the same contract update_sales's
    start_date_time is held to in production (always a real UTC epoch, per
    gRPC convention). A bare dt.timestamp() on a naive datetime assumes the
    HOST's local timezone instead, which only round-trips correctly on a
    UTC-host and silently breaks on any other (the bug this helper exists to
    not reintroduce in test fixtures)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return int(dt.timestamp())


def test_update_sales_routes_exact_local_match_to_local_market():
    system = _system_with_topology()
    err = system.update_sales(_market_sales("ESALC"), _utc_epoch(system.tick.get_datetime()))

    assert err is None
    assert system.local_markets["ESALC"].get_monthly_catch_snapshot()[("ANE", "Fresh - Whole")] == 10.0


def test_update_sales_routes_unknown_prefix_match_to_combined_market_with_warning(caplog):
    import logging
    system = _system_with_topology()

    with caplog.at_level(logging.WARNING):
        err = system.update_sales(_market_sales("ESXXX"), _utc_epoch(system.tick.get_datetime()))

    assert err is None
    assert system.combined_markets["ES"].get_direct_catch_snapshot()[("ANE", "Fresh - Whole")] == 10.0
    assert any("ESXXX" in r.message for r in caplog.records)


def test_update_sales_routes_exact_two_letter_code_to_combined_market():
    system = _system_with_topology()
    err = system.update_sales(_market_sales("FR"), _utc_epoch(system.tick.get_datetime()))

    assert err is None
    assert system.combined_markets["FR"].get_direct_catch_snapshot()[("ANE", "Fresh - Whole")] == 10.0


def test_update_sales_rejects_code_matching_no_known_prefix():
    system = _system_with_topology()
    err = system.update_sales(_market_sales("ZZXXX"), _utc_epoch(system.tick.get_datetime()))

    assert err is not None
    assert err.status_code == StatusCode.FAILED_PRECONDITION


def test_update_sales_rejects_non_positive_quantity():
    system = _system_with_topology()
    err = system.update_sales(_market_sales("ESALC", quantity=-1.0), _utc_epoch(system.tick.get_datetime()))

    assert err is not None
    assert err.status_code == StatusCode.INVALID_ARGUMENT


def test_update_sales_defers_future_dated_sales():
    system = _system_with_topology()
    future_ts = _utc_epoch(system.tick.get_datetime() + datetime.timedelta(days=31))
    err = system.update_sales(_market_sales("ESALC"), future_ts)

    assert err is None
    assert system.local_markets["ESALC"].get_monthly_catch_snapshot() == {}
    assert len(system.pending_sales_summary) == 1


def test_update_sales_interprets_start_date_time_as_utc_regardless_of_host_timezone():
    system = _system_with_topology()  # current_time (naive) = 2024-01-01 00:00:00

    # Unambiguous UTC epoch for exactly 2024-01-01T00:00:00Z, regardless of host tz.
    utc_epoch = int(datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC).timestamp())

    err = system.update_sales(_market_sales("ESALC"), utc_epoch)

    assert err is None
    # Must accumulate immediately (same tick), not defer to pending_sales_summary --
    # a buggy local-timezone interpretation would push this into the future on
    # any non-UTC host and wrongly defer it instead.
    assert system.local_markets["ESALC"].get_monthly_catch_snapshot()[("ANE", "Fresh - Whole")] == 10.0
    assert system.pending_sales_summary == {}


def test_update_sales_routes_correctly_with_utc_aware_tick():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC))
    system.setup_topology(_markets_df())

    utc_epoch = int(datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC).timestamp())
    err = system.update_sales(_market_sales("ESALC"), utc_epoch)

    assert err is None
    assert system.local_markets["ESALC"].get_monthly_catch_snapshot()[("ANE", "Fresh - Whole")] == 10.0
    assert system.pending_sales_summary == {}


import numpy as np


def _synthetic_prices_df(n_years=6, seed=0):
    """Two nearby high-volume ports (ESALC, ESBCN) sharing a species so
    they qualify as each other's High-tier neighbor, plus a third,
    far-away, low-volume port (ESVLC) with no qualifying neighbor."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    rows = []

    for port, base_price, base_volume in [("ESALC", 2.0, 5000.0), ("ESBCN", 2.2, 5500.0)]:
        volume = base_volume + rng.normal(0, 200, len(dates))
        price = base_price - 0.0005 * (volume - base_volume) + rng.normal(0, 0.01, len(dates))
        for d, p, v in zip(dates, price, volume):
            rows.append({
                "date": d, "market_code": port, "species_code": "ANE", "category_code": "Fresh - Whole",
                "price": p, "landings_kg": v, "size": None,
            })

    # ESVLC: tiny volume share (<0.2%) -> Low tier, never anyone's neighbor.
    # NOTE: at these coordinates ESVLC is NOT geographically far from
    # ESALC/ESBCN (haversine distance ~134km/~302km -- closer than ESALC
    # and ESBCN are to EACH OTHER, ~415km) and distance_weighted_neighbor_signal
    # has no distance cutoff, so ESVLC DOES get a valid (if low-weighted)
    # neighbor signal from them here. The "genuinely no High-tier neighbor"
    # edge case needs a port with no OTHER local High-tier port in the
    # universe at all -- see the dedicated fixture below this function.
    volume = 10.0 + rng.normal(0, 1, len(dates))
    price = 2.1 + rng.normal(0, 0.01, len(dates))
    for d, p, v in zip(dates, price, volume):
        rows.append({
            "date": d, "market_code": "ESVLC", "species_code": "ANE", "category_code": "Fresh - Whole",
            "price": p, "landings_kg": v, "size": None,
        })

    combined_volume = rows  # combined ES row = sum of all three ports
    combined_rows = []
    import pandas as _pd
    df_local = _pd.DataFrame(rows)
    for d, group in df_local.groupby("date"):
        combined_rows.append({
            "date": d, "market_code": "ES", "species_code": "ANE", "category_code": "Fresh - Whole",
            "price": group["price"].mean(), "landings_kg": group["landings_kg"].sum(), "size": None,
        })

    return pd.concat([pd.DataFrame(rows), pd.DataFrame(combined_rows)], ignore_index=True)


def _markets_df_three_ports():
    return pd.DataFrame({
        "market_code": ["ES", "ESALC", "ESBCN", "ESVLC"],
        "market_name": ["Spain", "Alicante", "Barcelona", "Valencia"],
        "lon": [None, -0.5, 2.2, -0.35],
        "lat": [None, 38.3, 41.4, 39.5],
    })


def test_fit_all_fits_combined_markets_and_loads_real_prices():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    prices_df = _synthetic_prices_df()

    system.fit_all(prices_df, latest_date=pd.Timestamp("2024-01-01"))

    assert ("ANE", "Fresh - Whole") in system.combined_markets["ES"].models


def test_fit_all_fits_step1_for_every_local_market():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_synthetic_prices_df(), latest_date=pd.Timestamp("2024-01-01"))

    for code in ["ESALC", "ESBCN", "ESVLC"]:
        assert ("ANE", "Fresh - Whole") in system.local_markets[code].step1_models


def test_fit_all_fits_step2_for_ports_with_a_high_tier_neighbor():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_synthetic_prices_df(), latest_date=pd.Timestamp("2024-01-01"))

    # ESALC and ESBCN are each other's dominant High-tier neighbor.
    assert ("ANE", "Fresh - Whole") in system.local_markets["ESALC"].step2_stacked_models
    assert ("ANE", "Fresh - Whole") in system.local_markets["ESBCN"].step2_stacked_models


def _prices_df_high_tier_share_straddles_threshold(n_years=6, seed=2):
    """A (3% of real local volume) is genuinely High-tier among the real
    local ports (A/B/C), but would be WRONGLY classified Mid-tier (1.5%)
    if the combined-market 'ES' aggregate row -- which duplicates the
    local total, not a fourth real port -- were counted in the
    denominator. A direct regression test for the High-tier volume-share
    fix (previously counted ES/FR, silently starving real ports of
    neighbors)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    rows = []
    for port, base_volume in [("ESALC", 3.0), ("ESBCN", 3.0), ("ESVLC", 94.0)]:
        volume = base_volume + rng.normal(0, base_volume * 0.01, len(dates))
        price = 2.0 + rng.normal(0, 0.01, len(dates))
        for d, p, v in zip(dates, price, volume):
            rows.append({
                "date": d, "market_code": port, "species_code": "ANE", "category_code": "Fresh - Whole",
                "price": p, "landings_kg": v, "size": None,
            })

    df_local = pd.DataFrame(rows)
    combined_rows = [
        {
            "date": d, "market_code": "ES", "species_code": "ANE", "category_code": "Fresh - Whole",
            "price": group["price"].mean(), "landings_kg": group["landings_kg"].sum(), "size": None,
        }
        for d, group in df_local.groupby("date")
    ]
    return pd.concat([df_local, pd.DataFrame(combined_rows)], ignore_index=True)


def test_fit_all_high_tier_share_excludes_combined_market_aggregate_row():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_prices_df_high_tier_share_straddles_threshold(), latest_date=pd.Timestamp("2024-01-01"))

    key = ("ANE", "Fresh - Whole")
    high_tier_ports = system.high_tier_ports_by_key[key]

    # ESALC's real share among local ports is 3/100 = 3% (High-tier,
    # >2%). If the ES aggregate row were wrongly counted in the
    # denominator (3/200 = 1.5%), it would be Mid-tier instead -- this
    # assertion fails under the pre-fix computation.
    assert "ESALC" in high_tier_ports
    assert "ESBCN" in high_tier_ports
    # The combined-market row itself is never a LocalMarket and must
    # never appear in a High-tier *set* at all, regardless of share.
    assert "ES" not in high_tier_ports


def test_fit_all_defaults_to_static_anchor_scheme():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_synthetic_prices_df(), latest_date=pd.Timestamp("2024-01-01"))

    key = ("ANE", "Fresh - Whole")
    assert key in system.local_markets["ESALC"].step2_stacked_models  # unchanged from before anchor_scheme existed


def test_fit_all_live_anchor_scheme_produces_a_different_but_valid_step2_model():
    key = ("ANE", "Fresh - Whole")
    prices_df = _synthetic_prices_df()

    static_system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    static_system.setup_topology(_markets_df_three_ports())
    static_system.fit_all(prices_df, latest_date=pd.Timestamp("2024-01-01"))

    live_system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    live_system.setup_topology(_markets_df_three_ports())
    live_system.fit_all(prices_df, latest_date=pd.Timestamp("2024-01-01"), anchor_scheme=consts.ANCHOR_SCHEME_LIVE)

    assert key in static_system.local_markets["ESALC"].step2_stacked_models
    assert key in live_system.local_markets["ESALC"].step2_stacked_models
    static_gamma = static_system.local_markets["ESALC"].step2_stacked_models[key]["gamma"]
    live_gamma = live_system.local_markets["ESALC"].step2_stacked_models[key]["gamma"]
    assert static_gamma != live_gamma  # different anchor source must produce a different fit

    # Step 1 and CombinedMarket don't depend on anchor_scheme at all.
    assert (
        static_system.local_markets["ESALC"].step1_models[key]["alpha"]
        == live_system.local_markets["ESALC"].step1_models[key]["alpha"]
    )
    assert (
        static_system.combined_markets["ES"].models[key]["alpha"]
        == live_system.combined_markets["ES"].models[key]["alpha"]
    )


def test_fit_all_retains_step2_fit_inputs_for_later_reuse():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_synthetic_prices_df(), latest_date=pd.Timestamp("2024-01-01"))

    key = ("ESALC", ("ANE", "Fresh - Whole"))
    assert key in system.step2_fit_inputs
    assert "neighbor_signal" in system.step2_fit_inputs[key]
    assert "port_monthly" in system.step2_fit_inputs[key]


def test_fit_all_only_fits_contracted_species_not_every_species_in_prices_csv():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.species_codes = ["ANE"]  # contract restricts to ANE only
    system.category_codes_contract = ["Fresh - Whole"]

    prices_df = _synthetic_prices_df()
    extra_species = prices_df[prices_df["market_code"] == "ESALC"].copy()
    extra_species["species_code"] = "UNCONTRACTED"
    prices_df = pd.concat([prices_df, extra_species], ignore_index=True)

    system.fit_all(prices_df, latest_date=pd.Timestamp("2024-01-01"))

    assert ("UNCONTRACTED", "Fresh - Whole") not in system.local_markets["ESALC"].step1_models
    assert ("ANE", "Fresh - Whole") in system.local_markets["ESALC"].step1_models


def _synthetic_prices_df_no_sibling_local_port(n_years=6, seed=1):
    """ESVLC is the ONLY local market reporting this species/category -- the
    combined 'ES' row (mirroring ESVLC alone) clears the High-tier
    volume-share threshold too, but CombinedMarket rows are never LocalMarket
    instances with a Step-1 historical series, so they can never pass
    distance_weighted_neighbor_signal's `port in predicted_series_by_port`
    check. With no OTHER local port reporting this key at all, ESVLC
    genuinely has no High-tier *local* neighbor to build a signal from --
    unlike the shared `_synthetic_prices_df` fixture above, where ESALC and
    ESBCN are both present and (since distance_weighted_neighbor_signal has
    no distance cutoff) always qualify as valid candidate sources for ANY
    local port, however far away."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    volume = 500.0 + rng.normal(0, 20, len(dates))
    price = 2.1 + rng.normal(0, 0.01, len(dates))

    rows = []
    for d, p, v in zip(dates, price, volume):
        rows.append({
            "date": d, "market_code": "ESVLC", "species_code": "ANE", "category_code": "Fresh - Whole",
            "price": p, "landings_kg": v, "size": None,
        })

    combined_rows = [
        {
            "date": r["date"], "market_code": "ES", "species_code": "ANE", "category_code": "Fresh - Whole",
            "price": r["price"], "landings_kg": r["landings_kg"], "size": None,
        }
        for r in rows
    ]
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(combined_rows)], ignore_index=True)


def test_fit_all_skips_step2_gracefully_for_port_with_no_high_tier_neighbor(caplog):
    import logging
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())

    with caplog.at_level(logging.WARNING):
        system.fit_all(_synthetic_prices_df_no_sibling_local_port(), latest_date=pd.Timestamp("2024-01-01"))

    # ESVLC is the only LOCAL market reporting ANE/Fresh-Whole here -- the
    # combined "ES" row is High-tier by volume share but is not a LocalMarket,
    # so it never qualifies as a neighbor *source* (spec section 9 point 3).
    assert ("ANE", "Fresh - Whole") not in system.local_markets["ESVLC"].step2_stacked_models
    assert any("No High-tier neighbor" in r.message or "no High-tier neighbor" in r.message for r in caplog.records)


def _fitted_system():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_synthetic_prices_df(), latest_date=pd.Timestamp("2024-01-01"))
    return system


def test_step_uses_real_price_when_available_and_skips_prediction_reporting():
    system = _fitted_system()
    system.local_markets["ESALC"].price_time_series[("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"))] = 7.77

    system.step()

    tick, snapshot = system.local_markets["ESALC"].price_history[-1]
    assert snapshot["prices"][("ANE", "Fresh - Whole")] == 7.77


def test_step_computes_step1_for_every_port_even_with_real_data(monkeypatch):
    system = _fitted_system()
    system.local_markets["ESALC"].price_time_series[("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"))] = 7.77

    calls = []
    original = system.local_markets["ESALC"].predict_step1

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    system.local_markets["ESALC"].predict_step1 = spy
    system.step()

    assert len(calls) > 0  # Step 1 still ran even though ESALC had real data this tick


def test_step_advances_tick_and_clears_accumulators():
    system = _fitted_system()
    system.local_markets["ESALC"].add_monthly_catch("ANE", "Fresh - Whole", 999.0)

    assert system.tick.get_tick() == 0
    system.step()

    assert system.tick.get_tick() == 1
    assert system.local_markets["ESALC"].get_monthly_catch_snapshot() == {}


def test_step_updates_volume_baseline_from_real_observed_volume():
    system = _fitted_system()
    before = dict(system.local_markets["ESALC"].volume_baseline_stats[("ANE", "Fresh - Whole")])

    system.local_markets["ESALC"].add_monthly_catch("ANE", "Fresh - Whole", 50000.0)
    system.step()

    after = system.local_markets["ESALC"].volume_baseline_stats[("ANE", "Fresh - Whole")]
    assert after[1] != before.get(1)


def test_step_combined_market_skips_volume_baseline_update_on_a_zero_catch_tick():
    system = _fitted_system()
    key = ("ANE", "Fresh - Whole")
    before = dict(system.combined_markets["ES"].volume_baseline_stats[key])

    system.step()  # no add_monthly_catch calls anywhere this tick

    after = system.combined_markets["ES"].volume_baseline_stats[key]
    assert after.get(1) == before.get(1), (
        "a zero-catch tick must not update the running mean (would dilute it toward 0 "
        "and inflate every later VolumeRatio) -- LocalMarket already skips absent keys "
        "for the same reason"
    )

    system.local_markets["ESALC"].add_monthly_catch("ANE", "Fresh - Whole", 1000.0)
    system.step()

    after_nonzero = system.combined_markets["ES"].volume_baseline_stats[key]
    assert after_nonzero[2] != before.get(2), "a real, nonzero catch must still update the baseline"


def test_step_records_history_for_combined_markets_too():
    system = _fitted_system()
    system.step()

    assert len(system.combined_markets["ES"].price_history) == 1


def test_step_falls_back_to_step1_when_step2_anchor_or_neighbor_unavailable():
    # ESVLC never got Step 2 coefficients (no High-tier neighbor, Task 12) --
    # its reported price must still be a valid Step-1 prediction, not None.
    system = _fitted_system()
    system.step()

    tick, snapshot = system.local_markets["ESVLC"].price_history[-1]
    assert snapshot["prices"][("ANE", "Fresh - Whole")] is not None


def test_step_neighbor_signal_is_unaffected_by_sibling_real_price():
    # ESALC and ESBCN are each other's dominant High-tier neighbor in
    # _fitted_system()'s synthetic data (see
    # test_fit_all_fits_step2_for_ports_with_a_high_tier_neighbor), so ESBCN
    # genuinely exercises the Step 2 neighbor-signal path here, not a
    # fallback-to-Step-1-only path.
    #
    # Two runs of the same fitted system and tick, differing only in
    # whether ESALC has a real recorded price set for this tick.
    system_without_real = _fitted_system()
    system_without_real.step()
    _, snapshot_without = system_without_real.local_markets["ESBCN"].price_history[-1]
    price_without_real = snapshot_without["prices"][("ANE", "Fresh - Whole")]

    system_with_real = _fitted_system()
    system_with_real.local_markets["ESALC"].price_time_series[
        ("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"))
    ] = 999.99  # a real price wildly different from anything Step 1 would predict
    system_with_real.step()
    _, snapshot_with = system_with_real.local_markets["ESBCN"].price_history[-1]
    price_with_real = snapshot_with["prices"][("ANE", "Fresh - Whole")]

    # ESBCN's own Step 2 output must be identical either way -- its neighbor
    # signal comes from ESALC's Step-1 PREDICTION, which doesn't change based
    # on whether ESALC also happens to have a real price on record this tick.
    assert price_with_real == price_without_real


import tempfile
from pathlib import Path


def test_get_species_prices_message_includes_only_valid_prices():
    system = _fitted_system()
    system.step()

    response = system.get_species_prices_message()
    codes = {sp.species.species_code for sp in response.species_price_summary.species_prices}
    assert "ANE" in codes
    # Every message must carry a real, non-None price and the configured currency.
    for sp in response.species_price_summary.species_prices:
        assert sp.currency == "EUR"
        assert sp.price is not None


def test_get_species_prices_message_includes_combined_market_prices():
    # Parity with ElsaHandler, which treats every markets.csv row --
    # including 2-letter combined-market codes -- as an ordinary local
    # market and reports a price for it. Without this, a caller querying
    # by market_code="ES" gets an answer from Elsa and nothing from here.
    system = _fitted_system()
    system.step()

    response = system.get_species_prices_message()
    combined_prices = [
        sp for sp in response.species_price_summary.species_prices if sp.market_code == "ES"
    ]
    assert combined_prices, "expected at least one SpeciesPrice reported under the combined-market code 'ES'"
    for sp in combined_prices:
        assert sp.species.species_code == "ANE"
        assert sp.currency == "EUR"
        assert sp.price is not None


def test_get_species_prices_message_includes_combined_market_prices_before_the_first_step():
    system = _fitted_system()  # fit_all() has run, step() has NOT

    response = system.get_species_prices_message()
    combined_prices = [
        sp for sp in response.species_price_summary.species_prices if sp.market_code == "ES"
    ]
    assert combined_prices, "expected a live-computed combined-market price even before the first step"
    for sp in combined_prices:
        assert sp.price is not None

    # Still a read-only computation -- no tick/history side effects.
    assert system.tick.get_tick() == 0
    assert system.combined_markets["ES"].price_history == []


def test_get_species_prices_message_works_before_the_first_step():
    system = _fitted_system()  # fit_all() has run, step() has NOT

    response = system.get_species_prices_message()

    codes = {sp.species.species_code for sp in response.species_price_summary.species_prices}
    assert "ANE" in codes
    for sp in response.species_price_summary.species_prices:
        assert sp.price is not None

    # Calling it again must not have advanced the tick or recorded any
    # history -- the live fallback is a read-only computation.
    assert system.tick.get_tick() == 0
    assert system.local_markets["ESALC"].price_history == []


def test_get_species_prices_message_skips_markets_with_no_history_yet():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    # No fit_all() and no step() -- no price_history exists anywhere.
    response = system.get_species_prices_message()
    assert len(response.species_price_summary.species_prices) == 0


def test_export_results_writes_local_and_combined_csv_files():
    system = _fitted_system()
    system.step()

    with tempfile.TemporaryDirectory() as tmp_dir:
        system.export_results("sim-001", tmp_dir)
        sim_dir = Path(tmp_dir) / "sim-001"
        assert (sim_dir / "combined_prices.csv").exists()
        assert (sim_dir / "local_prices.csv").exists()

        local_contents = (sim_dir / "local_prices.csv").read_text()
        assert "ESALC" in local_contents
