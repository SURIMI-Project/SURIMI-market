import fnmatch
from pathlib import Path
import shutil
import threading
import csv
import os
import logging
import datetime

import grpc
import pandas as pd

from src.MarketSystem import MarketSystem
from src.LocalMarket import LocalMarket
from surimi.v1 import simulation_pb2, initialise_simulation_pb2, simulate_step_pb2, finalise_simulation_pb2, update_sales_pb2
from util import consts
from util.MarketError import MarketError

class SimulationHandler:
    def __init__(self, simulation_id, scenario_id, start_time, step_size):
        """
        Manages the simulation loop, using MarketSystem as the core.
        Here we need to implement the Experiment = munltiple simulations idea of SURIMI controller. 
        Possilbly also change the name to experiment handler or simmilar. 
        Talk with RIK required to see what the controller handles and what the market model needs to handle.
        """
        self.simulation_id = simulation_id
        self.scenario_id = scenario_id
        self.start_time = start_time
        self.step_size = step_size

        self.logger = logging.getLogger(f"{consts.SIMULATION_HANDLER_LOGGER_NAME}.{simulation_id}")

        # Initialize market system with default currency and measurement unit
        self.market_system = MarketSystem(currency=consts.SUPPORTED_CURRENCY, start_time=start_time)

        # Internal dictionaries for quick reference
        self.gear_codes = []

        # Threading event for blocking control
        self.update_event = threading.Event()

        # Cached message for latest prices
        # TODO what is the purpose of this?
        self.latest_market_prices = None

    def init_simulation(self, simulation: simulation_pb2.Simulation) -> tuple[initialise_simulation_pb2.InitialiseSimulationResponse, MarketError]:
        """
        Trigger simulation initialization and respond to controller.
        Loads initialization config (species, ports, historical prices)
        and sets baseline prices for each market.

        TODO add typehint for return type.
        """
        # Copy config from the S3 downloaded configs to the simulation dirs.
        # TODO this is temporary. Should be removed when we can reliably download
        #      from S3.
        simulation_dir = Path(__file__).parent.parent / "simulations" / self.scenario_id / self.simulation_id
        simulation_dir.mkdir(parents=True, exist_ok=True)
        config_src = Path(__file__).parent.parent / "config"
        config_dst = simulation_dir / "config"
        if config_dst.exists():
            shutil.rmtree(config_dst)
        shutil.copytree(config_src, config_dst, ignore=ignore_non_csv)

        # Load configurations and data from CSV
        app_directory = Path(__file__).parent.parent.resolve()
        folder_path = Path(app_directory, "config")

        # Load species configs
        # TODO get the species from prices.csv instead of species.csv
        species_path = Path(folder_path, self.scenario_id, consts.SPECIES_CSV)
        if os.path.exists(species_path):
            self.logger.info(f"Loading species from {species_path}")
        else:
            self.logger.critical(f"{consts.SPECIES_CSV} not found in {folder_path}/{self.scenario_id}")
            return None, MarketError(grpc.StatusCode.NOT_FOUND, f"Configuration file {consts.SPECIES_CSV} was not found for scenario {self.scenario_id}.")
        loaded_species: list[str] = load_species_from_csv(species_path)

        # Load ports configs
        ports_path = Path(folder_path, self.scenario_id, consts.PORTS_CSV)
        if os.path.exists(ports_path):
            self.logger.info(f"Loading ports from {ports_path}")
        else:
            self.logger.critical(f"{consts.PORTS_CSV} not found in {folder_path}/{self.scenario_id}")
            return None, MarketError(grpc.StatusCode.NOT_FOUND, f"Configuration file {consts.PORTS_CSV} was not found for scenario {self.scenario_id}.")
        loaded_markets: dict[str,tuple] = load_ports_from_csv(ports_path)

        # Load prices configs
        prices_path = Path(folder_path, self.scenario_id, consts.PRICES_CSV)
        if os.path.exists(prices_path):
            self.logger.info(f"Loading ports from {prices_path}")
        else:
            self.logger.critical(f"{consts.PRICES_CSV} not found in {folder_path}/{self.scenario_id}")
            return None, MarketError(grpc.StatusCode.NOT_FOUND, f"Configuration file {consts.PRICES_CSV} was not found for scenario {self.scenario_id}.")
        loaded_prices = pd.read_csv(prices_path)

        # Validate ports
        contract_ports = set([port.market_code for port in simulation.items.markets])
        missing_ports = find_missing_ports(contract_ports=contract_ports,
                                           market_ports=set(loaded_markets.keys()))
        initialised_ports = {}
        if len(missing_ports) > 0:
            self.logger.critical(f"Missing the following ports: {missing_ports}")
            err_msg = "The market is missing the following ports:\n"
            for port in missing_ports:
                err_msg = f"  - {port}\n"
            return None, MarketError(grpc.StatusCode.NOT_FOUND, err_msg)
        else:
            # Remove excessive ports
            valid_ports: set = get_valid_ports_only(set(loaded_markets.keys()), contract_ports)
            for port in (set(loaded_markets.keys()) - valid_ports):
                del loaded_markets[port]
            
            # Instantiate new LocalMarket objects
            for port, data in loaded_markets.items():
                initialised_ports[port] = LocalMarket(port, data[0], data[1], data[2])

        # Validate the Market Service's available species
        contract_species = set([species.species_code for species in simulation.items.species])
        missing_species = find_missing_species(contract_species=contract_species,
                                               market_species=set(loaded_species))
        if len(missing_species) > 0:
            self.logger.critical(f"Missing the following species: {missing_species}")
            err_msg = "The market is missing the following species:\n"
            for species in missing_species:
                err_msg = f"  - {species}\n"
            return None, MarketError(grpc.StatusCode.NOT_FOUND, err_msg)
        else:
            # Remove species not in the contract
            loaded_species = list(get_valid_species_only(set(loaded_species), contract_species))
        
        # Define months that we have real data for. Used to decide whether to
        # predict or use real prices.
        self.market_system.real_data_months = list(set(loaded_prices["date"]))

        # Get all unique gear codes from the contract.
        gear_contract = [segment.gear_code for segment in list(simulation.items.fleet_segments)]
        gear_contract = set(gear_contract)

        # Transfer ownerships of loaded data to the MarketSystem
        self.market_system.species_codes = loaded_species
        self.market_system.local_markets = initialised_ports
        self.market_system.gear_codes_contract = gear_contract

        # Initialise baseline prices
        self.market_system.set_local_market_initial_price()
        
        # Elser et al. assumes that the global market price is static.
        self.market_system.update_global_price_history()
        self.market_system.set_missing_local_prices_with_global_fallback()

        # Checks if the species in the configs are in the contract. If some
        # aren't, then fill these in so no species are missing.
        # TODO add unit test.
        missing_species = []
        for species in simulation.items.species:
            if species.species_code not in loaded_species:
                missing_species.append(species.species_code)

        if missing_species:
            raise ValueError(
                f"Species codes from simulation contract not found in loaded species configuration: {missing_species}"
            )
        
        self.logger.info(f"Tick {self.market_system.tick.get_tick()} initialized.")
        return initialise_simulation_pb2.InitialiseSimulationResponse(simulation_id=self.simulation_id), None

    def finalise_simulation(self, folder_path="simulations"):
        """
        Trigger simulation finalisation and respond to controller.
        Writes results into simulations/<simulation_id>.
        """
        self.logger.info("Finalising simulation...")
        self.market_system.export_results(self.simulation_id, Path(folder_path, self.scenario_id))
        self.logger.info(f"Results saved in simulations/{self.scenario_id}/{self.simulation_id}")
        return finalise_simulation_pb2.FinaliseSimulationResponse(simulation_id=self.simulation_id)

    def simulate_step(self, controller_simulation_time: datetime.datetime) \
        -> tuple[simulate_step_pb2.SimulateStepResponse, MarketError]:
        """
        Trigger a simulation step.
        Uses all buffered sales for this tick.
        """
        self.logger.info("Simulate step triggered...")
        if controller_simulation_time != self.market_system.tick.get_datetime():
            return None, MarketError(
                status_code=grpc.StatusCode.NOT_FOUND,
                message=f"The MarketService's time, {self.market_system.tick.get_timestamp()} does not match the input time of {controller_simulation_time.isoformat() + "Z"}"
            )
        self.market_system.step()

        tick = self.market_system.tick.get_tick()
        current_date = self.market_system.tick.get_timestamp()[:10]
        self.logger.info(f"Tick {tick} ({current_date}) completed.")

        return simulate_step_pb2.SimulateStepResponse(simulation_id=self.simulation_id), None

    def get_latest_market_prices(self):
        """
        Get the most recent market prices.
        """
        self.latest_market_prices = self.market_system.get_species_prices_message()
        return self.latest_market_prices

    def update_sales(self, request: update_sales_pb2.UpdateSalesRequest) -> tuple[update_sales_pb2.UpdateSalesResponse, MarketError]:
        """
        Receives external sales (catches) from controller.
        """
        err = self.market_system.update_sales(
            request.sales_summary.market_sales,
            request.start_date_time.seconds
        )
        return update_sales_pb2.UpdateSalesResponse(simulation_id=self.simulation_id), err

def load_ports_from_csv(ports_path: Path) -> dict[str, tuple[str, float, float]]:
    """
    Load local market (port) definitions from CSV into the market system.

    CSV columns: port_code, port_name, lon, lat
    """
    # Return value
    loaded_markets = {}

    with open(ports_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            loaded_markets[row["port_code"].strip()] = (
                row["port_name"].strip(),
                float(row["lon"]),
                float(row["lat"])
            )

    return loaded_markets

def ignore_non_csv(dir, files):
    return {f for f in files if not fnmatch.fnmatch(f, "*.csv")}

def load_species_from_csv(species_path) -> list[str]:
    """
    Load species definitions from CSV file into the market system.

    CSV columns: species_code, species_name

    Returns
    -------
        loaded_species : Collection of all species by their code found in
                            the CSV file.
    """
    logger = logging.getLogger("SimulationHandler")

    # Return value
    loaded_species = []

    # TODO exit if the file was not found
    if os.path.exists(species_path):
        logger.info(f"Loading species from {species_path}")
    else:
        logger.critical(f"{consts.PORTS_CSV} not found in {species_path.parent}") 

    with open(species_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            code = row["species_code"].strip()
            loaded_species.append(code)

    return loaded_species

def find_missing_ports(contract_ports: set, market_ports: set) -> list[str]:
    """
    Find ports that are specified by the contract but missing from  the datasets.
    """
    return contract_ports - set(market_ports)

def find_missing_species(contract_species: set, market_species: set) -> list[str]:
    """
    Find species specified by the contract but missing from the dataset.
    """
    return contract_species - market_species


def get_valid_ports_only(market_ports: set[str], contract_ports: set[str]) -> set[str]:
    """
    Get list of ports from the datasets that adhere to the contract

    Arguments
    ---------
        contract_ports : Set of all ports that must be present.
        market_ports : Set of all ports from the datasets. May include ports
                       not in the contract.
    
    Returns
    -------
        Set of valid ports
    """
    excessive_ports = market_ports - contract_ports
    valid_ports = market_ports - excessive_ports
    return valid_ports

def get_valid_species_only(market_species: set[str], contract_species: set[str]) -> set[str]:
    """
    Get list of species from the datasets that adhere to the contract
    
    Arguments
    ---------
        contract_ports : Set of all species that must be present.
        market_species : Set of all species from the datasets. May include species
                       not in the contract.

    Returns
    -------
        Set of valid species
    """
    excessive_species = market_species - contract_species
    valid_species = market_species - excessive_species
    return valid_species