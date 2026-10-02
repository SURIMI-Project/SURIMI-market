import datetime
from pathlib import Path

import pandas as pd
import pytest
from grpc import StatusCode

from src.models.market.MarketHandler import MarketHandler


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
