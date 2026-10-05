"""
Market-model equivalent of test/market/test_update_sales.py's scenarios.

That file, despite living under test/market/, actually tests ElsaHandler:
it drives MarketService through the shared root test/conftest.py fixtures
(`market_service`/`contract`/`config_dir`), which never set MODEL_NAME, so
MarketService.InitialiseSimulation silently defaults to "elsa" -- confirmed
empirically, not just read from code (constructed the handler and printed
its type). Those fixtures' PORT_A/PORT_B-style codes are also structurally
incompatible with this plugin's 5-letter/2-letter LOCODE topology (neither
length matches build_prefix_to_locals' filters), so forcing MODEL_NAME=market
onto them would not even work -- MarketSystem.setup_topology would derive
an empty topology.

This file gives MarketHandler/MarketSystem its own LOCODE-compatible
fixtures and exercises the SAME accumulation and future-landings-deferral
scenarios for real, using real protobuf UpdateSalesRequest messages.
"""
import datetime

import pandas as pd
import pytest
from google.protobuf.json_format import ParseDict
from surimi.v1.update_sales_pb2 import UpdateSalesRequest

from src.models.market.MarketHandler import MarketHandler

START = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)


class _FakeMarket:
    def __init__(self, market_code):
        self.market_code = market_code


class _FakeSpecies:
    def __init__(self, species_code):
        self.species_code = species_code


class _FakePriceCategory:
    def __init__(self, category_code):
        self.category_code = category_code


class _FakeItems:
    def __init__(self, markets, species, price_categories):
        self.markets = markets
        self.species = species
        self.price_categories = price_categories


class _FakeSimulation:
    def __init__(self, items):
        self.items = items


def _write_market_config(tmp_path, scenario_id="update_sales_scn"):
    scenario_dir = tmp_path / scenario_id
    scenario_dir.mkdir()

    pd.DataFrame({
        "market_code": ["ESALC", "ESBCN", "ESCAR"],
        "market_name": ["Alicante", "Barcelona", "Cartagena"],
        "lon": [-0.481, 2.169, -0.986],
        "lat": [38.345, 41.385, 37.605],
    }).to_csv(scenario_dir / "markets.csv", index=False)

    pd.DataFrame({
        "species_code": ["ANE", "HKE"],
        "species_name": ["Anchovy", "Hake"],
    }).to_csv(scenario_dir / "species.csv", index=False)

    dates = pd.date_range("2016-01-01", periods=48, freq="MS")
    rows = []
    for port, base_price in [("ESALC", 2.0), ("ESBCN", 2.3), ("ESCAR", 2.1)]:
        for species, sp_price in [("ANE", base_price), ("HKE", base_price + 1.0)]:
            for i, d in enumerate(dates):
                rows.append({
                    "date": d, "market_code": port, "species_code": species,
                    "category_code": "Fresh - Whole",
                    "landings_kg": 500.0 + 10 * (i % 7),
                    "price": sp_price + 0.01 * (i % 5),
                    "currency": "EUR", "measurement_unit": "kg",
                })
    pd.DataFrame(rows).to_csv(scenario_dir / "prices.csv", index=False)
    return scenario_id


def _init_handler(tmp_path):
    scenario_id = _write_market_config(tmp_path)
    simulation = _FakeSimulation(_FakeItems(
        markets=[_FakeMarket("ESALC"), _FakeMarket("ESBCN"), _FakeMarket("ESCAR")],
        species=[_FakeSpecies("ANE"), _FakeSpecies("HKE")],
        price_categories=[_FakePriceCategory("Fresh - Whole")],
    ))
    handler = MarketHandler("sim-update-sales", scenario_id, START, 1)
    response, err = handler.init_simulation(simulation, str(tmp_path))
    assert err is None, f"init_simulation failed: {err}"
    return handler


def _update_request(start_date_time, market_sales):
    data = {
        "simulation_id": "sim-update-sales",
        "start_date_time": start_date_time,
        "end_date_time": "2100-01-01T00:00:00Z",
        "sales_summary": {"market_sales": market_sales},
    }
    return ParseDict(data, UpdateSalesRequest())


def test_accumulate_landings_for_the_current_month(tmp_path):
    handler = _init_handler(tmp_path)
    market_system = handler.market_system

    # Request 1: two ANE/Fresh-Whole sales at ESALC in the current month.
    request_1 = _update_request("2020-01-01T00:00:00Z", [
        {
            "market_code": "ESALC", "currency": "EUR",
            "sales": [
                {"species": {"species_code": "ANE"}, "quantity": 1, "category_code": "Fresh - Whole"},
                {"species": {"species_code": "ANE"}, "quantity": 1, "category_code": "Fresh - Whole"},
            ],
        },
    ])
    market_system.update_sales(request_1.sales_summary.market_sales, request_1.start_date_time.seconds)

    snapshot_1 = market_system.local_markets["ESALC"].get_monthly_catch_snapshot()
    assert snapshot_1[("ANE", "Fresh - Whole")] == 2, (
        "local market with updated landings for the current timestep should store this number of landings (2kg)"
    )
    assert not snapshot_1.get(("ANE", "Frozen - Whole")), (
        "port should not register landings for a category code it was never updated with"
    )

    # Request 2: another update to the same port -- must ADD, not overwrite,
    # and register a brand-new species entry correctly.
    request_2 = _update_request("2020-01-01T00:00:00Z", [
        {
            "market_code": "ESALC", "currency": "EUR",
            "sales": [
                {"species": {"species_code": "ANE"}, "quantity": 2.3, "category_code": "Fresh - Whole"},
                {"species": {"species_code": "HKE"}, "quantity": 1, "category_code": "Fresh - Whole"},
            ],
        },
    ])
    market_system.update_sales(request_2.sales_summary.market_sales, request_2.start_date_time.seconds)

    snapshot_2 = market_system.local_markets["ESALC"].get_monthly_catch_snapshot()
    assert snapshot_2[("ANE", "Fresh - Whole")] == pytest.approx(4.3), (
        "sending another update request to already-updated landings should add the new landings (2 + 2.3 = 4.3)"
    )
    assert snapshot_2[("HKE", "Fresh - Whole")] == 1, (
        "sending an update for a new (species, category) should register only the new landings (1kg)"
    )

    # Request 3: update two different ports in a single request.
    request_3 = _update_request("2020-01-01T00:00:00Z", [
        {
            "market_code": "ESALC", "currency": "EUR",
            "sales": [{"species": {"species_code": "ANE"}, "quantity": 1, "category_code": "Fresh - Whole"}],
        },
        {
            "market_code": "ESBCN", "currency": "EUR",
            "sales": [{"species": {"species_code": "ANE"}, "quantity": 1, "category_code": "Fresh - Whole"}],
        },
    ])
    market_system.update_sales(request_3.sales_summary.market_sales, request_3.start_date_time.seconds)

    assert market_system.local_markets["ESALC"].get_monthly_catch_snapshot()[("ANE", "Fresh - Whole")] == pytest.approx(5.3), (
        "updating an existing port's landings for the current month should keep adding (4.3 + 1 = 5.3)"
    )
    assert market_system.local_markets["ESBCN"].get_monthly_catch_snapshot()[("ANE", "Fresh - Whole")] == 1.0, (
        "updating a different, not-yet-touched port in the same request should set its own landings (1kg)"
    )


def test_update_future_landings_are_deferred_not_applied_immediately(tmp_path):
    handler = _init_handler(tmp_path)
    market_system = handler.market_system

    # One month ahead of the handler's own tick 0 (start_time) -- must be
    # deferred into pending_sales_summary, not applied to current landings.
    request_1 = _update_request("2020-02-01T00:00:00Z", [
        {
            "market_code": "ESALC", "currency": "EUR",
            "sales": [
                {"species": {"species_code": "ANE"}, "quantity": 1, "category_code": "Fresh - Whole"},
                {"species": {"species_code": "ANE"}, "quantity": 2, "category_code": "Fresh - Whole"},
            ],
        },
        {
            "market_code": "ESBCN", "currency": "EUR",
            "sales": [{"species": {"species_code": "ANE"}, "quantity": 1, "category_code": "Fresh - Whole"}],
        },
    ])
    market_system.update_sales(request_1.sales_summary.market_sales, request_1.start_date_time.seconds)

    current_landings = market_system.local_markets["ESALC"].get_monthly_catch_snapshot().get(
        ("ANE", "Fresh - Whole")
    )
    assert not current_landings, "updating future landings must not affect the current month's landings"

    pending = market_system.pending_sales_summary
    assert len(pending) == 1, "updating one future month should result in exactly one pending month"
    month = next(iter(pending))
    assert len(pending[month]) == 1, "one update_sales call should register exactly one pending batch"
    batch = pending[month][0]
    assert len(batch) == 2, "the one pending batch should carry both ports from the single request"

    # A second update for the SAME future month must append a new pending
    # batch to the same month, not merge into or replace the first one.
    request_2 = _update_request("2020-02-01T00:00:00Z", [
        {
            "market_code": "ESALC", "currency": "EUR",
            "sales": [{"species": {"species_code": "ANE"}, "quantity": 1, "category_code": "Fresh - Whole"}],
        },
        {
            "market_code": "ESBCN", "currency": "EUR",
            "sales": [{"species": {"species_code": "ANE"}, "quantity": 1, "category_code": "Fresh - Whole"}],
        },
        {
            "market_code": "ESCAR", "currency": "EUR",
            "sales": [{"species": {"species_code": "ANE"}, "quantity": 1, "category_code": "Fresh - Whole"}],
        },
    ])
    market_system.update_sales(request_2.sales_summary.market_sales, request_2.start_date_time.seconds)

    pending = market_system.pending_sales_summary
    assert len(pending) == 1, "a second update for the same future month must not create a new pending month"
    assert len(pending[month]) == 2, "the same month should now carry 2 pending batches (one per update_sales call)"
    assert len(pending[month][-1]) == 3, "the newest pending batch should carry all 3 ports from its own request"

    # A future update for a DIFFERENT month must add a second pending month.
    request_3 = _update_request("2020-03-01T00:00:00Z", [
        {
            "market_code": "ESALC", "currency": "EUR",
            "sales": [{"species": {"species_code": "ANE"}, "quantity": 1, "category_code": "Fresh - Whole"}],
        },
    ])
    market_system.update_sales(request_3.sales_summary.market_sales, request_3.start_date_time.seconds)

    pending = market_system.pending_sales_summary
    assert len(pending) == 2, "an update for a second distinct future month should add a second pending month"
