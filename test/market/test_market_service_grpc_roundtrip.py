"""
End-to-end gRPC-level test for the market model, driven entirely through
MarketService's actual servicer methods with real protobuf request
messages -- not MarketHandler's Python methods called directly (every
other test in this project, including test_MarketHandler.py and the
fidelity backtest harness, calls MarketHandler directly with a lightweight
duck-typed stand-in for the protobuf `Simulation` message, which has never
verified that the REAL protobuf message shape matches what those stand-ins
assume).

This exercises: MODEL_NAME-based routing to MarketHandler (server/model_router.py),
InitialiseSimulation, UpdateSales, SimulateStep (twice), GetSpeciesPrices, and
FinaliseSimulation, in the order a real controller would call them.
"""
import datetime
import shutil
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from google.protobuf.json_format import ParseDict
from google.protobuf.timestamp_pb2 import Timestamp
from surimi.v1 import (
    finalise_simulation_pb2,
    get_species_prices_pb2,
    initialise_simulation_pb2,
    sales_pb2,
    simulate_step_pb2,
    species_pb2,
    update_sales_pb2,
)

import definitions
from server.MarketService import MarketService

START = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)


class _DummyServicerContext:
    """Unlike test/conftest.py's DummyServicerContext (which swallows
    context.abort() silently), this one raises -- any abort in this test
    is a real failure we want to see, not a return-None-and-continue."""

    def abort(self, status_code, message):
        raise AssertionError(f"gRPC call aborted: {status_code}: {message}")


def _write_market_config(tmp_path, scenario_id):
    scenario_dir = tmp_path / scenario_id
    scenario_dir.mkdir()

    pd.DataFrame({
        "market_code": ["ESALC", "ESBCN"],
        "market_name": ["Alicante", "Barcelona"],
        "lon": [-0.481, 2.169],
        "lat": [38.345, 41.385],
    }).to_csv(scenario_dir / "markets.csv", index=False)

    pd.DataFrame({
        "species_code": ["ANE", "HKE"],
        "species_name": ["Anchovy", "Hake"],
    }).to_csv(scenario_dir / "species.csv", index=False)

    dates = pd.date_range("2016-01-01", periods=48, freq="MS")
    rows = []
    for port, base_price in [("ESALC", 2.0), ("ESBCN", 2.3)]:
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


def _build_contract(simulation_id, scenario_id):
    data = {
        "simulation_id": simulation_id,
        "scenario_name": scenario_id,
        "simulation": {
            "start_date_time": "2020-01-01T00:00:00Z",
            "items": {
                "markets": [{"market_code": "ESALC"}, {"market_code": "ESBCN"}],
                "currencies": [{"code": "EUR"}],
                "species": [{"species_code": "ANE"}, {"species_code": "HKE"}],
                "price_categories": [{"category_code": "Fresh - Whole"}],
            },
        },
    }
    request = initialise_simulation_pb2.InitialiseSimulationRequest()
    return ParseDict(data, request)


def _timestamp(dt: datetime.datetime) -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(dt)
    return ts


def test_market_service_full_grpc_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "market")
    monkeypatch.setenv("ROOT_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("AWS_BUCKET_NAME", raising=False)

    scenario_id = "grpc_e2e_scn"
    simulation_id = "grpc-e2e-sim"
    _write_market_config(tmp_path, scenario_id)
    contract = _build_contract(simulation_id, scenario_id)
    context = _DummyServicerContext()
    service = MarketService("1.0.0-test")

    # 1. InitialiseSimulation -- routes to MarketHandler via MODEL_NAME=market,
    # downloads nothing from S3 (AWS_BUCKET_NAME unset), reads the real CSVs.
    init_response = service.InitialiseSimulation(contract, context)
    assert init_response is not None
    assert init_response.simulation_id == simulation_id
    assert simulation_id in service.simulation_dictionary
    handler = service.simulation_dictionary[simulation_id]
    assert type(handler).__name__ == "MarketHandler"

    # Initialising the same simulation_id twice must be rejected.
    try:
        service.InitialiseSimulation(contract, context)
        raise AssertionError("expected a duplicate simulation_id to abort")
    except AssertionError as e:
        assert "already exists" in str(e)

    # 2. UpdateSales at tick 0, before stepping, so step() sees the catch.
    sale = sales_pb2.Sale(
        species=species_pb2.Species(species_code="ANE"),
        category_code="Fresh - Whole",
        quantity=1000.0,
        value=2000.0,
    )
    market_sales = sales_pb2.MarketSales(market_code="ESALC", currency="EUR", sales=[sale])
    update_request = update_sales_pb2.UpdateSalesRequest(
        simulation_id=simulation_id,
        start_date_time=_timestamp(START),
        sales_summary=sales_pb2.SalesSummary(market_sales=[market_sales]),
    )
    update_response = service.UpdateSales(update_request, context)
    assert update_response is not None
    assert update_response.simulation_id == simulation_id
    assert handler.market_system.local_markets["ESALC"].get_monthly_catch_snapshot()[
        ("ANE", "Fresh - Whole")
    ] == 1000.0

    # 3. SimulateStep for tick 0 -- current_date_time must match the handler's
    # own tick clock exactly (still at start_time; the tick hasn't advanced).
    step_response_0 = service.SimulateStep(
        simulate_step_pb2.SimulateStepRequest(simulation_id=simulation_id, current_date_time=_timestamp(START)),
        context,
    )
    assert step_response_0 is not None
    assert handler.market_system.tick.get_tick() == 1
    # The catch fed via UpdateSales must have been consumed, not carried over.
    assert handler.market_system.local_markets["ESALC"].get_monthly_catch_snapshot() == {}

    # 4. GetSpeciesPrices after tick 0 -- real protobuf response, real fields.
    prices_response = service.GetSpeciesPrices(
        get_species_prices_pb2.GetSpeciesPricesRequest(simulation_id=simulation_id, date_time=_timestamp(START)),
        context,
    )
    assert prices_response is not None
    assert prices_response.simulation_id == simulation_id
    codes = {sp.species.species_code for sp in prices_response.species_price_summary.species_prices}
    assert "ANE" in codes
    assert "HKE" in codes
    for sp in prices_response.species_price_summary.species_prices:
        assert sp.price is not None
        assert sp.currency == "EUR"

    # 5. SimulateStep for tick 1 -- no catches fed this time.
    tick_1_time = START + relativedelta(months=1)
    step_response_1 = service.SimulateStep(
        simulate_step_pb2.SimulateStepRequest(simulation_id=simulation_id, current_date_time=_timestamp(tick_1_time)),
        context,
    )
    assert step_response_1 is not None
    assert handler.market_system.tick.get_tick() == 2

    # Calling SimulateStep again with a stale (already-passed) tick time must
    # be rejected, not silently re-processed.
    try:
        service.SimulateStep(
            simulate_step_pb2.SimulateStepRequest(
                simulation_id=simulation_id, current_date_time=_timestamp(START)
            ),
            context,
        )
        raise AssertionError("expected a stale SimulateStep time to abort")
    except AssertionError as e:
        assert "does not" in str(e)

    # 6. FinaliseSimulation -- writes real output files, then removes the
    # simulation from the service's in-memory dictionary.
    scenario_output_dir = Path(definitions.ROOT_DIR, "simulations", scenario_id)
    output_dir = scenario_output_dir / simulation_id
    try:
        finalise_response = service.FinaliseSimulation(
            finalise_simulation_pb2.FinaliseSimulationRequest(simulation_id=simulation_id), context
        )
        assert finalise_response is not None
        assert finalise_response.simulation_id == simulation_id
        assert simulation_id not in service.simulation_dictionary
        assert (output_dir / "combined_prices.csv").exists()
        assert (output_dir / "local_prices.csv").exists()
    finally:
        shutil.rmtree(scenario_output_dir, ignore_errors=True)
