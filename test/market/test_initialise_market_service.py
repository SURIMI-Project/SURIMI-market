from pathlib import Path
import os
from src.SimulationHandler import load_ports_from_csv, \
                                  load_species_from_csv, \
                                  SimulationHandler
from util import consts

def test_load_ports(contract, config_dir):
    # handler = SimulationHandler({}, "1234")
    scenario_name = contract.scenario_name
    ports_path = config_dir / scenario_name / consts.PORTS_CSV
    ports = load_ports_from_csv(ports_path)
    ports = list(ports.keys())
    assert len(ports) > 0, f"Found fixture {consts.PORTS_CSV}, but no ports were loaded"

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
    species_path = config_dir / scenario_name / consts.SPECIES_CSV
    species = load_species_from_csv(species_path)
    assert len(species) > 0, f"Found fixture {consts.SPECIES_CSV}, but no species were loaded"

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
    handler = SimulationHandler("aaaa", test_scenario_name, None, 1)
    response, error = handler.init_simulation(contract.simulation, config_dir)
    assert error, "Non-matching scenario name to the contract should result in an error"

    handler = SimulationHandler("bbbb", contract.scenario_name, None, 1)
    response, error = handler.init_simulation(contract.simulation, config_dir)
    assert not error, "Matching and valid scenario name from the contract result in no errors."