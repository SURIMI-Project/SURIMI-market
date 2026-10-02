import datetime

import pandas as pd
import pytest
from grpc import StatusCode

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
