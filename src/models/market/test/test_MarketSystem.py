import datetime

import pandas as pd
import pytest
from grpc import StatusCode

from src.models.market.MarketSystem import MarketSystem, build_prefix_to_locals
from src.models.market.test.fidelity_backtest_harness import _live_anchor_resolver_from


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


def test_setup_topology_errors_cleanly_when_lon_lat_columns_are_missing_entirely():
    """A markets.csv missing the lon/lat columns entirely (not just NaN
    values within them -- a malformed reference-data file) must return a
    clean MarketError like every other validation failure here, not let
    an uncaught pandas KeyError surface from build_distance_lookup."""
    markets_df = pd.DataFrame({
        "market_code": ["ES", "ESALC"],
        "market_name": ["Spain", "Alicante"],
        # no "lon"/"lat" columns at all
    })
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    err = system.setup_topology(markets_df)

    assert err is not None
    assert err.status_code == StatusCode.FAILED_PRECONDITION


def test_setup_topology_does_not_error_on_combined_markets_missing_coordinates():
    # ES/FR having NA lon/lat is expected -- only LOCAL markets (5-letter
    # codes) are checked.
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    err = system.setup_topology(_markets_df())
    assert err is None


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
    """Encode a (possibly naive, semantically-UTC) datetime as a genuine UTC
    epoch int, matching update_sales's start_date_time contract. A bare
    dt.timestamp() on a naive datetime uses the HOST's local timezone
    instead, which only round-trips on a UTC host -- the bug this helper
    exists to avoid reintroducing in test fixtures."""
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
    # NOTE: ESVLC is not geographically far from ESALC/ESBCN (~134km/~302km,
    # closer than they are to each other) and the neighbor signal has no
    # distance cutoff, so ESVLC still gets a valid, low-weighted signal here.
    # The "genuinely no High-tier neighbor" case needs a port with no other
    # local High-tier port at all -- see the fixture below.
    volume = 10.0 + rng.normal(0, 1, len(dates))
    price = 2.1 + rng.normal(0, 0.01, len(dates))
    for d, p, v in zip(dates, price, volume):
        rows.append({
            "date": d, "market_code": "ESVLC", "species_code": "ANE", "category_code": "Fresh - Whole",
            "price": p, "landings_kg": v, "size": None,
        })

    combined_volume = rows  # combined ES row = sum of all three ports
    combined_rows = []
    df_local = pd.DataFrame(rows)
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


def test_fit_all_fits_base_models_for_every_local_market():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_synthetic_prices_df(), latest_date=pd.Timestamp("2024-01-01"))

    for code in ["ESALC", "ESBCN", "ESVLC"]:
        assert ("ANE", "Fresh - Whole") in system.local_markets[code].models


def test_fit_all_fits_neighbor_models_for_ports_with_a_high_tier_neighbor():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_synthetic_prices_df(), latest_date=pd.Timestamp("2024-01-01"))

    # ESALC and ESBCN are each other's dominant High-tier neighbor.
    assert ("ANE", "Fresh - Whole") in system.local_markets["ESALC"].neighbor_models
    assert ("ANE", "Fresh - Whole") in system.local_markets["ESBCN"].neighbor_models


def _prices_df_with_combined_market_aggregate_row(n_years=6, seed=2):
    """Three real local ports (ESALC, ESBCN, ESVLC) with positive volume,
    plus a combined-market 'ES' aggregate row that duplicates the local
    total rather than being a fourth real port -- regression fixture for
    making sure that aggregate row never leaks into the donor-candidate
    set alongside the genuine local ports."""
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


def test_fit_all_donor_candidates_excludes_combined_market_aggregate_row():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_prices_df_with_combined_market_aggregate_row(), latest_date=pd.Timestamp("2024-01-01"))

    key = ("ANE", "Fresh - Whole")
    donor_candidate_ports = system.donor_candidates_by_key[key]

    # All three real local ports have positive volume for this key, so
    # all three are donor candidates (final eligibility -- the nearest-K
    # narrowing -- happens later, inside compute_inverse_distance_weights).
    assert "ESALC" in donor_candidate_ports
    assert "ESBCN" in donor_candidate_ports
    assert "ESVLC" in donor_candidate_ports
    # The combined-market row itself is never a LocalMarket and must
    # never appear in the donor-candidate set at all.
    assert "ES" not in donor_candidate_ports


def test_fit_all_defaults_to_static_anchor_scheme():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_synthetic_prices_df(), latest_date=pd.Timestamp("2024-01-01"))

    key = ("ANE", "Fresh - Whole")
    assert key in system.local_markets["ESALC"].neighbor_models  # unchanged from before anchor_scheme existed


def test_fit_all_live_anchor_scheme_produces_a_different_but_valid_neighbor_model():
    key = ("ANE", "Fresh - Whole")
    prices_df = _synthetic_prices_df()

    static_system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    static_system.setup_topology(_markets_df_three_ports())
    static_system.fit_all(prices_df, latest_date=pd.Timestamp("2024-01-01"))

    live_system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    live_system.setup_topology(_markets_df_three_ports())
    blended = prices_df[prices_df["size"].isna()]
    combined_monthly_by_key = {
        prefix: live_system._build_monthly_by_key(blended[blended["market_code"] == prefix])
        for prefix in live_system.combined_markets
    }
    resolver = _live_anchor_resolver_from(combined_monthly_by_key)
    live_system.fit_all(prices_df, latest_date=pd.Timestamp("2024-01-01"), anchor_resolver=resolver)

    assert key in static_system.local_markets["ESALC"].neighbor_models
    assert key in live_system.local_markets["ESALC"].neighbor_models
    static_gamma = static_system.local_markets["ESALC"].neighbor_models[key].gamma
    live_gamma = live_system.local_markets["ESALC"].neighbor_models[key].gamma
    assert static_gamma != live_gamma  # different anchor source must produce a different fit

    # The base model and CombinedMarket don't depend on anchor_scheme at all.
    assert (
        static_system.local_markets["ESALC"].models[key].alpha
        == live_system.local_markets["ESALC"].models[key].alpha
    )
    assert (
        static_system.combined_markets["ES"].models[key].alpha
        == live_system.combined_markets["ES"].models[key].alpha
    )


def test_fit_all_retains_neighbor_fit_inputs_for_later_reuse():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_synthetic_prices_df(), latest_date=pd.Timestamp("2024-01-01"))

    key = ("ESALC", ("ANE", "Fresh - Whole"))
    assert key in system.neighbor_fit_inputs
    assert "neighbor_signal" in system.neighbor_fit_inputs[key]
    assert "port_monthly" in system.neighbor_fit_inputs[key]


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

    assert ("UNCONTRACTED", "Fresh - Whole") not in system.local_markets["ESALC"].models
    assert ("ANE", "Fresh - Whole") in system.local_markets["ESALC"].models


def _synthetic_prices_df_no_sibling_local_port(n_years=6, seed=1):
    """ESVLC is the ONLY local market reporting this key. The combined 'ES'
    row clears the High-tier volume-share threshold too, but CombinedMarket
    rows are never LocalMarket instances, so they can't pass
    distance_weighted_neighbor_signal's local-port check -- leaving ESVLC
    genuinely without a High-tier local neighbor, unlike the shared
    `_synthetic_prices_df` fixture where ESALC/ESBCN both qualify."""
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


def test_fit_all_skips_neighbor_model_gracefully_for_port_with_no_high_tier_neighbor(caplog):
    import logging
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())

    with caplog.at_level(logging.WARNING):
        system.fit_all(_synthetic_prices_df_no_sibling_local_port(), latest_date=pd.Timestamp("2024-01-01"))

    # ESVLC is the only LOCAL market reporting ANE/Fresh-Whole here -- the
    # combined "ES" row is High-tier by volume share but is not a
    # LocalMarket, so it never qualifies as a neighbor source.
    assert ("ANE", "Fresh - Whole") not in system.local_markets["ESVLC"].neighbor_models
    assert any("no eligible neighbor port available" in r.message for r in caplog.records)


def _fitted_system():
    system = MarketSystem("EUR", datetime.datetime(2024, 1, 1))
    system.setup_topology(_markets_df_three_ports())
    system.fit_all(_synthetic_prices_df(), latest_date=pd.Timestamp("2024-01-01"))
    return system


def test_step_uses_real_price_when_available_and_skips_prediction_reporting():
    system = _fitted_system()
    system.local_markets["ESALC"].load_price_time_series(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "market_code": ["ESALC"],
        "species_code": ["ANE"],
        "category_code": ["Fresh - Whole"],
        "price": [7.77],
        "size": [None],
    }))

    system.step()

    tick, snapshot = system.local_markets["ESALC"].price_history[-1]
    assert snapshot["prices"][("ANE", "Fresh - Whole")] == 7.77


def test_step_computes_base_model_for_every_port_even_with_real_data(monkeypatch):
    system = _fitted_system()
    system.local_markets["ESALC"].load_price_time_series(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "market_code": ["ESALC"],
        "species_code": ["ANE"],
        "category_code": ["Fresh - Whole"],
        "price": [7.77],
        "size": [None],
    }))

    calls = []
    original = system.local_markets["ESALC"].predict_from_base_model

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    system.local_markets["ESALC"].predict_from_base_model = spy
    system.step()

    assert len(calls) > 0  # the base model still ran even though ESALC had real data this tick


def test_step_advances_tick_and_clears_accumulators():
    system = _fitted_system()
    system.local_markets["ESALC"].add_monthly_catch("ANE", "Fresh - Whole", 999.0)

    assert system.tick.get_tick() == 0
    system.step()

    assert system.tick.get_tick() == 1
    assert system.local_markets["ESALC"].get_monthly_catch_snapshot() == {}


def test_step_restores_taken_catch_if_the_rest_of_the_tick_fails(monkeypatch):
    """take_monthly_catch()/take_direct_catch() atomically clear the
    accumulators as the very first thing in step(), to close the
    concurrent-UpdateSales race (a sale landing between a separate
    snapshot and a later clear used to be silently dropped). But that
    means if anything after the take() raises, the already-cleared catch
    must not simply vanish -- a retried step() (the tick number never
    advanced, since increase_tick() is the last line of
    _advance_tick_state) would otherwise silently treat this tick as
    zero-catch forever. step() must restore what it took before
    re-raising."""
    system = _fitted_system()
    system.local_markets["ESALC"].add_monthly_catch("ANE", "Fresh - Whole", 999.0)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure partway through the tick")

    monkeypatch.setattr(system, "_advance_tick_state", boom)

    with pytest.raises(RuntimeError, match="simulated failure"):
        system.step()

    assert system.tick.get_tick() == 0  # the tick never actually advanced
    assert system.local_markets["ESALC"].get_monthly_catch_snapshot() == {
        ("ANE", "Fresh - Whole"): 999.0
    }


def test_step_updates_volume_baseline_from_real_observed_volume():
    system = _fitted_system()
    before = system.local_markets["ESALC"].volume_baseline[("ANE", "Fresh - Whole")].snapshot()

    system.local_markets["ESALC"].add_monthly_catch("ANE", "Fresh - Whole", 50000.0)
    system.step()

    after = system.local_markets["ESALC"].volume_baseline[("ANE", "Fresh - Whole")].snapshot()
    assert after[1] != before.get(1)


def test_step_combined_market_skips_volume_baseline_update_on_a_zero_catch_tick():
    system = _fitted_system()
    key = ("ANE", "Fresh - Whole")
    before = system.combined_markets["ES"].volume_baseline[key].snapshot()

    system.step()  # no add_monthly_catch calls anywhere this tick

    after = system.combined_markets["ES"].volume_baseline[key].snapshot()
    assert after.get(1) == before.get(1), (
        "a zero-catch tick must not update the running mean (would dilute it toward 0 "
        "and inflate every later VolumeRatio) -- LocalMarket already skips absent keys "
        "for the same reason"
    )

    system.local_markets["ESALC"].add_monthly_catch("ANE", "Fresh - Whole", 1000.0)
    system.step()

    after_nonzero = system.combined_markets["ES"].volume_baseline[key].snapshot()
    assert after_nonzero[2] != before.get(2), "a real, nonzero catch must still update the baseline"


def test_step_records_history_for_combined_markets_too():
    system = _fitted_system()
    system.step()

    assert len(system.combined_markets["ES"].price_history) == 1


def test_step_falls_back_to_base_model_when_neighbor_model_anchor_or_neighbor_unavailable():
    # ESVLC never got a neighbor-informed model (no High-tier neighbor) -- its
    # reported price must still be a valid base-model prediction, not None.
    system = _fitted_system()
    system.step()

    tick, snapshot = system.local_markets["ESVLC"].price_history[-1]
    assert snapshot["prices"][("ANE", "Fresh - Whole")] is not None


def test_step_neighbor_signal_is_unaffected_by_sibling_real_price():
    # ESALC and ESBCN are each other's dominant High-tier neighbor here, so
    # ESBCN genuinely exercises the neighbor-informed model's neighbor-signal
    # path, not a fallback-to-base-model-only path.
    #
    # Two runs of the same fitted system and tick, differing only in
    # whether ESALC has a real recorded price set for this tick.
    system_without_real = _fitted_system()
    system_without_real.step()
    _, snapshot_without = system_without_real.local_markets["ESBCN"].price_history[-1]
    price_without_real = snapshot_without["prices"][("ANE", "Fresh - Whole")]

    system_with_real = _fitted_system()
    system_with_real.local_markets["ESALC"].load_price_time_series(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "market_code": ["ESALC"],
        "species_code": ["ANE"],
        "category_code": ["Fresh - Whole"],
        "price": [999.99],  # a real price wildly different from anything the base model would predict
        "size": [None],
    }))
    system_with_real.step()
    _, snapshot_with = system_with_real.local_markets["ESBCN"].price_history[-1]
    price_with_real = snapshot_with["prices"][("ANE", "Fresh - Whole")]

    # ESBCN's own neighbor-informed-model output must be identical either way --
    # its neighbor signal comes from ESALC's base-model PREDICTION, which
    # doesn't change based on whether ESALC also happens to have a real price
    # on record this tick.
    assert price_with_real == price_without_real


def test_fit_all_donor_candidates_by_key_pinned_before_perf_refactor():
    """Pins donor_candidates_by_key, built in _fit_neighbor_models_for_all_ports
    by filtering+grouping the whole `blended` DataFrame once per key, against
    a fixed synthetic fixture. Exists so the planned O(keys^2) -> O(n)
    optimization (grouping once before the key loop instead of re-filtering
    per key) can be proven behavior-preserving -- this must not change."""
    system = _fitted_system()
    assert system.donor_candidates_by_key == {("ANE", "Fresh - Whole"): {"ESVLC", "ESBCN", "ESALC"}}


def test_step_neighbor_candidate_lookup_pins_exact_price_before_perf_refactor():
    """Pins the exact neighbor-informed price for ESBCN (whose neighbor
    signal comes from ESALC's base-model prediction, via the
    base_predictions scan in _compute_tick_prices) against a fixed,
    deterministic synthetic fixture. Exists specifically so the planned
    O((ports*keys)^2) -> O(ports*keys) optimization of that scan (indexing
    base_predictions by key instead of filtering the whole dict per
    (port, key) pair) can be proven byte-for-byte behavior-preserving --
    this value must not change by even one bit after that refactor."""
    system = _fitted_system()
    system.step()

    tick, snapshot = system.local_markets["ESBCN"].price_history[-1]
    assert snapshot["prices"][("ANE", "Fresh - Whole")] == 2.305057802311024


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
