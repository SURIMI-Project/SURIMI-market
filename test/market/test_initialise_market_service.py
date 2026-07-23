import os
import pandas as pd
from pathlib import Path
from src.server.MarketService import MarketService
from src.models.elsa.ElsaHandler import load_ports_from_csv, \
                                  load_species_from_csv, \
                                  ElsaHandler
PORTS_CSV = "markets.csv"
SPECIES_CSV = "species.csv"
PRICES_CSV = "prices.csv"
ROOT_CONFIG_DIR = "ROOT_CONFIG_DIR"

def test_load_ports(contract, config_dir):
    # handler = SimulationHandler({}, "1234")
    scenario_name = contract.scenario_name
    ports_path = config_dir / scenario_name / PORTS_CSV
    ports = load_ports_from_csv(ports_path)
    ports = list(ports.keys())
    assert len(ports) > 0, f"Found fixture {PORTS_CSV}, but no ports were loaded"

    dataset_ports = []
    with open(ports_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            X = line.split(",")
            dataset_ports.append(X[0])

    assert len(ports) == len(dataset_ports), f"Then number of loaded ports and the dataset ports do not match ({len(ports)} vs {(len(dataset_ports))})"

def test_load_species(contract, config_dir):
    scenario_name = contract.scenario_name
    species_path = config_dir / scenario_name / SPECIES_CSV
    species = load_species_from_csv(species_path)
    assert len(species) > 0, f"Found fixture {SPECIES_CSV}, but no species were loaded"

    dataset_species = []
    with open(species_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            X = line.split(",")
            dataset_species.append(X[0])

    assert len(species) == len(dataset_species), f"Then number of loaded species and the dataset species do not match ({len(species)} vs {(len(dataset_species))})"

def test_initialise_simulation(contract, config_dir):
    test_scenario_name = "non-matching scenario name"
    handler = ElsaHandler("aaaa", test_scenario_name, None, 1)
    response, error = handler.init_simulation(contract.simulation, config_dir)
    assert error, "Non-matching scenario name to the contract should result in an error"
    assert response is None, "Failed initialisation should result in 'response is None'"

    handler = ElsaHandler("bbbb", contract.scenario_name, None, 1)
    response, error = handler.init_simulation(contract.simulation, config_dir)
    assert not error, "Matching and valid scenario name from the contract result in no errors."
    assert response, "A successful initialisaton should result in a valid response."
    assert response.simulation_id == "bbbb", "Initialised simulation does not have the test scenario name 'bbbb'"

# TODO remove this in the future. This is specific to model implementation.
def test_initialise_local_market_price_history(contract, config_dir):
    handler = ElsaHandler("bbbb", contract.scenario_name, None, 1)
    response, error = handler.init_simulation(contract.simulation, config_dir)
    
    # Check that the initial price timeline is correct
    prices_path = Path(config_dir, contract.scenario_name, PRICES_CSV)
    df = pd.read_csv(prices_path)
    data_for_port_a = df[df["market_code"] == "PORT_A"]
    price_history = handler.market_system.local_markets.get("PORT_A").price_time_series
    do_match = len(price_history) == len(data_for_port_a)
    assert do_match, f"The number of initialised prices do not match with the dataset {PRICES_CSV}"

def test_initialise_simulation_find_config_dir(grpc_context, contract, config_dir):
    handlers = {}
    market_service = MarketService("1.0.0-test")

    os.environ[ROOT_CONFIG_DIR] = "hello"
    response = market_service.InitialiseSimulation(contract, grpc_context)
    assert not response, \
        f"Setting the wrong ROOT_CONFIG_DIR, i.e. 'hello', should fail the" + \
        "initialisation."

def test_initialise_market_service_twice(grpc_context, contract, config_dir):
    handlers = {}
    market_service = MarketService("1.0.0-test")

    os.environ[ROOT_CONFIG_DIR] = str(config_dir)
    response = market_service.InitialiseSimulation(contract, grpc_context)
    assert response, \
        f"Initialising a new simulation by ID with a correctly configured " + \
        "'ROOT_CONFIG_DIR' should be successful."

    response = market_service.InitialiseSimulation(contract, grpc_context)
    assert not response, \
        f"Initialisating an existing simulation by ID should fail."