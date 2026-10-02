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
