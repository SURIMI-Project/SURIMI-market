import datetime
from pathlib import Path

import pandas as pd
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from grpc import StatusCode
from surimi.v1 import sales_pb2, species_pb2, update_sales_pb2

from src.models.market.MarketHandler import MarketHandler
from src.models.market.test.fake_simulation import (
    FakeMarket as _FakeMarket,
    FakeSpecies as _FakeSpecies,
    FakePriceCategory as _FakePriceCategory,
    FakeItems as _FakeItems,
    FakeSimulation as _FakeSimulation,
)


def _write_minimal_config(tmp_path, scenario_id="scn"):
    scenario_dir = tmp_path / scenario_id
    scenario_dir.mkdir()

    pd.DataFrame({
        "market_code": ["ES", "ESALC"],
        "market_name": ["Spain", "Alicante"],
        "lon": [None, -0.5],
        "lat": [None, 38.3],
    }).to_csv(scenario_dir / "markets.csv", index=False)

    pd.DataFrame({"species_code": ["ANE"], "species_name": ["Anchovy"]}).to_csv(
        scenario_dir / "species.csv", index=False
    )

    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    pd.DataFrame({
        "date": dates, "market_code": "ESALC", "species_code": "ANE", "category_code": "Fresh - Whole",
        "landings_kg": 500.0, "price": 2.0, "currency": "EUR", "measurement_unit": "kg", "size": None,
    }).to_csv(scenario_dir / "prices.csv", index=False)

    return tmp_path


def _fake_simulation():
    return _FakeSimulation(_FakeItems(
        markets=[_FakeMarket("ESALC")],
        species=[_FakeSpecies("ANE")],
        price_categories=[_FakePriceCategory("Fresh - Whole")],
    ))


def test_init_simulation_succeeds_with_valid_config(tmp_path):
    root_config_dir = _write_minimal_config(tmp_path)
    handler = MarketHandler("sim-1", "scn", datetime.datetime(2024, 1, 1), 1)

    response, err = handler.init_simulation(_fake_simulation(), str(root_config_dir))

    assert err is None
    assert response.simulation_id == "sim-1"
    assert "ESALC" in handler.market_system.local_markets


def test_init_simulation_errors_on_missing_config_files(tmp_path):
    (tmp_path / "scn").mkdir()
    handler = MarketHandler("sim-1", "scn", datetime.datetime(2024, 1, 1), 1)

    response, err = handler.init_simulation(_fake_simulation(), str(tmp_path))

    assert response is None
    assert err.status_code == StatusCode.NOT_FOUND


def test_init_simulation_errors_on_missing_contract_port(tmp_path):
    root_config_dir = _write_minimal_config(tmp_path)
    handler = MarketHandler("sim-1", "scn", datetime.datetime(2024, 1, 1), 1)

    simulation = _FakeSimulation(_FakeItems(
        markets=[_FakeMarket("ESALC"), _FakeMarket("ESXXX")],  # ESXXX not in markets.csv
        species=[_FakeSpecies("ANE")],
        price_categories=[_FakePriceCategory("Fresh - Whole")],
    ))

    response, err = handler.init_simulation(simulation, str(root_config_dir))

    assert response is None
    assert err.status_code == StatusCode.NOT_FOUND


def test_init_simulation_errors_on_missing_contract_species(tmp_path):
    root_config_dir = _write_minimal_config(tmp_path)
    handler = MarketHandler("sim-1", "scn", datetime.datetime(2024, 1, 1), 1)

    simulation = _FakeSimulation(_FakeItems(
        markets=[_FakeMarket("ESALC")],
        species=[_FakeSpecies("ANE"), _FakeSpecies("ZZZ")],  # ZZZ not in species.csv
        price_categories=[_FakePriceCategory("Fresh - Whole")],
    ))

    response, err = handler.init_simulation(simulation, str(root_config_dir))

    assert response is None
    assert err.status_code == StatusCode.NOT_FOUND


def test_init_simulation_succeeds_without_optional_size_ratios_csv(tmp_path):
    root_config_dir = _write_minimal_config(tmp_path)
    assert not (root_config_dir / "scn" / "size_ratios.csv").exists()

    handler = MarketHandler("sim-1", "scn", datetime.datetime(2024, 1, 1), 1)
    response, err = handler.init_simulation(_fake_simulation(), str(root_config_dir))

    assert err is None


def test_init_simulation_returns_a_clean_error_for_a_malformed_prices_csv(tmp_path):
    """prices.csv existing but failing to parse (missing the date column,
    corrupted content, wrong encoding -- a realistic condition from a
    truncated download or a manual edit) must return a MarketError like
    every other validation failure in this method, not let the underlying
    pandas ParserError/ValueError propagate uncaught out of the gRPC call."""
    scenario_dir = tmp_path / "scn"
    scenario_dir.mkdir()
    pd.DataFrame({
        "market_code": ["ES", "ESALC"], "market_name": ["Spain", "Alicante"],
        "lon": [None, -0.5], "lat": [None, 38.3],
    }).to_csv(scenario_dir / "markets.csv", index=False)
    pd.DataFrame({"species_code": ["ANE"], "species_name": ["Anchovy"]}).to_csv(
        scenario_dir / "species.csv", index=False
    )
    # No "date" column at all -- prices.csv exists but can't be parsed the
    # way this method requires.
    pd.DataFrame({
        "market_code": ["ESALC"], "species_code": ["ANE"], "category_code": ["Fresh - Whole"],
        "landings_kg": [500.0], "price": [2.0], "currency": ["EUR"], "measurement_unit": ["kg"],
    }).to_csv(scenario_dir / "prices.csv", index=False)

    handler = MarketHandler("sim-1", "scn", datetime.datetime(2024, 1, 1), 1)
    response, err = handler.init_simulation(_fake_simulation(), str(tmp_path))

    assert response is None
    assert err is not None
    assert err.status_code == StatusCode.INVALID_ARGUMENT


def _init_handler(tmp_path):
    root_config_dir = _write_minimal_config(tmp_path)
    handler = MarketHandler("sim-1", "scn", datetime.datetime(2024, 1, 1), 1)
    handler.init_simulation(_fake_simulation(), str(root_config_dir))
    return handler


def test_update_sales_accumulates_catch_for_current_tick(tmp_path):
    handler = _init_handler(tmp_path)

    sale = sales_pb2.Sale(
        species=species_pb2.Species(species_code="ANE"),
        category_code="Fresh - Whole",
        quantity=100.0,
        value=200.0,
    )
    market_sales = sales_pb2.MarketSales(market_code="ESALC", currency="EUR", sales=[sale])

    ts = Timestamp()
    ts.FromDatetime(datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc))
    request = update_sales_pb2.UpdateSalesRequest(
        sales_summary=sales_pb2.SalesSummary(market_sales=[market_sales]),
        start_date_time=ts,
    )

    response, err = handler.update_sales(request)

    assert err is None
    assert response.simulation_id == "sim-1"
    assert handler.market_system.local_markets["ESALC"].get_monthly_catch_snapshot()[("ANE", "Fresh - Whole")] == 100.0


def test_simulate_step_advances_tick_when_time_matches(tmp_path):
    handler = _init_handler(tmp_path)

    response, err = handler.simulate_step(datetime.datetime(2024, 1, 1))

    assert err is None
    assert response.simulation_id == "sim-1"
    assert handler.market_system.tick.get_tick() == 1


def test_simulate_step_errors_on_time_mismatch(tmp_path):
    handler = _init_handler(tmp_path)

    response, err = handler.simulate_step(datetime.datetime(2099, 1, 1))

    assert response is None
    assert err is not None


def test_get_species_prices_returns_latest_message(tmp_path):
    handler = _init_handler(tmp_path)
    handler.simulate_step(datetime.datetime(2024, 1, 1))

    response = handler.get_species_prices()

    assert len(response.species_price_summary.species_prices) > 0


def test_finalise_simulation_writes_export_files(tmp_path):
    handler = _init_handler(tmp_path)
    handler.simulate_step(datetime.datetime(2024, 1, 1))

    response = handler.finalise_simulation(str(tmp_path / "simulations"))

    assert response.simulation_id == "sim-1"
    assert (tmp_path / "simulations" / "scn" / "sim-1" / "local_prices.csv").exists()
