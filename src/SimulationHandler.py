import fnmatch
from pathlib import Path
import shutil
import threading
import csv
import os
import logging

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

        # Flag to detect concurrent SimulateStep and UpdateSales calls
        # TODO why not use mutex?
        self._step_in_progress = False

    def init_simulation(self, simulation: simulation_pb2.Simulation):
        """
        Trigger simulation initialization and respond to controller.
        Loads initialization config (species, ports, historical prices)
        and sets baseline prices for each market.

        TODO add typehint for return type.
        """
        # Copy config from the S3 downloaded configs to the simulation dirs.
        # TODO this is temporary. Should be removed when we can reliably download
        #      from S3.
        simulation_dir = Path(__file__).parent.parent / "simulations" / self.simulation_id
        simulation_dir.mkdir(parents=True, exist_ok=True)
        config_src = Path(__file__).parent.parent / "config"
        config_dst = simulation_dir / "config"
        if config_dst.exists():
            shutil.rmtree(config_dst)
        shutil.copytree(config_src, config_dst, ignore=ignore_non_csv)

        # TODO add unit testing for each of the three actions.
        # Load configurations and data from CSV 
        loaded_species = self.load_species_from_csv()
        loaded_markets = self.load_ports_from_csv()
        loaded_prices = self.load_prices_from_csv()
        self.load_local_market_price_history(loaded_markets, loaded_prices)

        # Get all unique gear codes from the contract.
        gear_contract = [segment.gear_code for segment in list(simulation.items.fleet_segments)]
        gear_contract = set(gear_contract)

        # Transfer ownerships of loaded data to the MarketSystem
        self.market_system.species_codes = loaded_species
        self.market_system.local_markets = loaded_markets
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
        
        # Set the initial tick 0 price history, using prices from the
        # configuration files.
        self.market_system.update_local_history()
        self.market_system.tick.increase_tick()

        self.logger.info(f"Tick {self.market_system.tick.get_tick()} initialized.")
        return initialise_simulation_pb2.InitialiseSimulationResponse(simulation_id=self.simulation_id)

    def finalise_simulation(self, folder_path="simulations"):
        """
        Trigger simulation finalisation and respond to controller.
        Writes results into simulations/<simulation_id>.
        """
        self.logger.info("Finalising simulation...")
        self.market_system.export_results(self.simulation_id, folder_path)
        self.logger.info(f"Results saved in simulations/{self.simulation_id}")
        return finalise_simulation_pb2.FinaliseSimulationResponse(simulation_id=self.simulation_id)

    def simulate_step(self):
        """
        Trigger a simulation step.
        Uses all buffered sales for this tick.
        """
        self.logger.info("Simulate step triggered...")
        self._step_in_progress = True
        # TODO investigate how this can fail
        try:
            self.market_system.step()

        finally:
            self._step_in_progress = False

        tick = self.market_system.tick.get_tick()
        current_date = self.market_system.tick.get_timestamp()[:10]
        self.logger.info(f"Tick {tick} ({current_date}) completed.")

        return simulate_step_pb2.SimulateStepResponse(simulation_id=self.simulation_id)

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
        err = None
        if not self.market_system.update_sales(request.sales_summary.market_sales):
            err = MarketError(
                status_code=grpc.StatusCode.FAILED_PRECONDITION,
                message="At least one local market does not exist."
            )
        return update_sales_pb2.UpdateSalesResponse(simulation_id=self.simulation_id), err

    def load_species_from_csv(self) -> list[str]:
        """
        Load species definitions from CSV file into the market system.

        CSV columns: species_code, species_name

        Returns
        -------
            loaded_species : Collection of all species by their code found in
                             the CSV file.
        """
        app_directory = Path(__file__).parent.parent.resolve()
        folder_path = str(Path(app_directory) / "config")
        species_path = os.path.join(folder_path, consts.SPECIES_CSV)

        # Return value
        loaded_species = []

        if os.path.exists(species_path):
            self.logger.info(f"Loading species from {species_path}")
        else:
            self.logger.warning(f"{consts.PORTS_CSV} not found in {folder_path}") 

        with open(species_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                code = row["species_code"].strip()
                loaded_species.append(code)

        return loaded_species

    def load_ports_from_csv(self) -> dict[str, LocalMarket]:
        """
        Load local market (port) definitions from CSV into the market system.

        CSV columns: port_code, port_name, lon, lat
        """
        app_directory = Path(__file__).parent.parent.resolve()
        folder_path = str(Path(app_directory) / "config")
        ports_path = os.path.join(folder_path, consts.PORTS_CSV)

        # Return value
        loaded_markets = {}

        if os.path.exists(ports_path):
            self.logger.info(f"Loading species from {ports_path}")
        else:
            self.logger.warning(f"{consts.PORTS_CSV} not found in {folder_path}") 

        with open(ports_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                code = row["port_code"].strip()
                name = row["port_name"].strip()
                lon = float(row["lon"])
                lat = float(row["lat"])

                market = LocalMarket(
                    market_code=code,
                    name=name,
                    lon=lon,
                    lat=lat
                )

                loaded_markets[code] = market

        return loaded_markets

    def load_prices_from_csv(self) -> pd.DataFrame:
        app_directory = Path(__file__).parent.parent.resolve()
        folder_path = str(Path(app_directory) / "config")
        prices_path = os.path.join(folder_path, consts.PRICES_CSV)
        
        # Test what happens if the file does not exist.
        self.logger.info(f"Loading species from {prices_path}")
        return pd.read_csv(prices_path)

    def load_local_market_price_history(self, markets: dict[str, LocalMarket], data: pd.DataFrame):
        """
        Load time-series prices into each local market from CSV file.

        CSV columns: date, market_code, species_code, category_code, price, currency, measurement_unit
        """
        # TODO add check when price is not supported
        for code, found_markets in data.groupby("market_code"):
            market = markets.get(code)
            if markets.get(code) is not None:
                for row in found_markets.itertuples(index=False):
                    date = row[0]
                    species_code = row[2]
                    gear_code = row[3]
                    price = row[4]

                    market.price_time_series[(species_code, gear_code, date)] = price

def ignore_non_csv(dir, files):
    return {f for f in files if not fnmatch.fnmatch(f, "*.csv")}
