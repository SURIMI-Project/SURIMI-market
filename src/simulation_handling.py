import fnmatch
from pathlib import Path
import shutil
import threading
import csv
import os
from collections import defaultdict
import logging

import grpc

from src.MarketSystem import MarketSystem
from src.Species import Species
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

        self.logger = logging.getLogger(f"simulation.{simulation_id}")

        # Initialize market system with default currency and measurement unit
        self.market_system = MarketSystem(currency=consts.SUPPORTED_CURRENCY, start_time=start_time)

        # Internal dictionaries for quick reference
        self.species_dict = {}
        self.local_market_dict = {}
        self.gear_codes = []

        # Threading event for blocking control
        self.update_event = threading.Event()

        # Cached message for latest prices
        # TODO what is the purpose of this?
        self.latest_market_prices = None

        # Aggregated catches for the current tick
        # key = (market_code, species_code) -> total quantity
        self.pending_catch_dict = defaultdict(float)

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
        self.load_species_from_csv()
        self.load_ports_from_csv()
        self.load_prices_from_csv()

        # Get all unique gear codes from the contract.
        gear_codes = [segment.gear_code for segment in list(simulation.items.fleet_segments)]
        gear_codes = set(gear_codes)
        
        # Initialise baseline prices
        self.market_system.set_local_market_initial_price(gear_codes)
        
        # Elser et al. assumes that the global market price is static.
        self.market_system.update_global_price_history(gear_codes)
        self.market_system.set_missing_local_prices_with_global_fallback(gear_codes)

        # Checks if the species in the configs are in the contract. If some
        # aren't, then fill these in so no species are missing.
        # TODO add unit test.
        missing_species = []
        for species in simulation.items.species:
            if species.species_code not in self.species_dict:
                missing_species.append(species.species_code)

        if missing_species:
            raise ValueError(
                f"Species codes from simulation contract not found in loaded species configuration: {missing_species}"
            )
        
        # Tick 0, i.e. the initial tick, stores the prices from configuration.
        self.market_system._update_local_history()
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
        try:
            catches = [(m, s, gc, qty) for (m, s, gc), qty in self.pending_catch_dict.items()]
            self.market_system.step(catches)

            total_qty = sum(qty for _, _, _, qty in catches)
            self.logger.info(f"Processed {len(catches)} market/species pairs, total_qty={total_qty:.2f} kg")

            # TODO need to make sure that values are nullified
            self.pending_catch_dict.clear()
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
        Aggregates them into pending_catch_dict for this tick.
        """
        self.logger.info("Sales update received...")

        # TODO handle race condition gracefully.
        if self._step_in_progress:
            self.logger.warning(
                "UpdateSales called while SimulateStep is in progress. Sale may be lost."
            )

        new_entries = 0


        # Validate the markets' existence.
        market_sales = request.sales_summary.market_sales
        markets_existence = [s.market_code in self.local_market_dict
                             for s in request.sales_summary.market_sales]


        if not all(markets_existence):
            return None, MarketError(
                status_code=grpc.StatusCode.FAILED_PRECONDITION,
                message="At least one local market does not exist."
            )

        for summary in request.sales_summary.market_sales:
            market_code = summary.market_code
            local_market = self.local_market_dict.get(market_code)

            if local_market is None:
                assert local_market, f"Local market {local_market} does not exist. This should not happen."
                continue

            for sale in summary.sales:
                species_code = sale.species.species_code
                quantity = sale.quantity
                gear_code = sale.fleet_segment.gear_code

                local_market.add_monthly_catch(species_code, gear_code, quantity)
                new_entries += 1

                # Use only for debugging. This is a very expensive operation.
                # Optionally, comment it out.
                self.logger.debug(
                    f"Sale: market={market_code}, species={species_code}, gear={gear_code}, qty={quantity:.2f} kg"
                )

        self.logger.info(f"Processed {new_entries} sales.")
        self.logger.info(f"Currently buffered {len(self.pending_catch_dict)} unique market/species combos.")

        return update_sales_pb2.UpdateSalesResponse(simulation_id=self.simulation_id), None

    def load_species_from_csv(self):
        """
        Load species definitions from CSV file into the market system.

        CSV columns: species_code, species_name
        """
        app_directory = Path(__file__).parent.parent.resolve()
        folder_path = str(Path(app_directory) / "config")
        species_path = os.path.join(folder_path, consts.SPECIES_CSV)

        if os.path.exists(species_path):
            self.logger.info(f"Loading species from {species_path}")
        else:
            self.logger.warning(f"{consts.PORTS_CSV} not found in {folder_path}") 

        with open(species_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                code = row["species_code"].strip()
                name = row["species_name"].strip()
                species = Species(species_name=name, species_code=code)
                self.market_system.add_species(species)
                self.species_dict[code] = species

    def load_ports_from_csv(self):
        """
        Load local market (port) definitions from CSV into the market system.

        CSV columns: port_code, port_name, lon, lat
        """
        app_directory = Path(__file__).parent.parent.resolve()
        folder_path = str(Path(app_directory) / "config")
        ports_path = os.path.join(folder_path, consts.PORTS_CSV)

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
                    global_market=self.market_system.global_market,
                    name=name,
                    lon=lon,
                    lat=lat
                )
                self.market_system.add_local_market(market)
                self.local_market_dict[code] = market

    def load_prices_from_csv(self):
        """
        Load time-series prices into each local market from CSV file.

        CSV columns: date, market_code, species_code, category_code, price, currency, measurement_unit
        """
        app_directory = Path(__file__).parent.parent.resolve()
        folder_path = str(Path(app_directory) / "config")
        prices_path = os.path.join(folder_path, consts.PRICES_CSV)

        if os.path.exists(prices_path):
            self.logger.info(f"Loading species from {prices_path}")
        else:
            self.logger.warning(f"{consts.PRICES_CSV} not found in {folder_path}")

        with open(prices_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                date = row["date"].strip()
                market_code = row["market_code"].strip()
                species_code = row["species_code"].strip()
                price = float(row["price"])
                currency = row["currency"].strip()

                # TODO include gear code in the calculation.
                gear_code = row["gear_code"].strip()

                # TODO check if this is broken.
                if currency != consts.SUPPORTED_CURRENCY:
                    self.logger.warning(f"SimulationHandler.load_prices_from_csv(): Currency {currency} was loaded, but only {consts.SUPPORTED_CURRENCY} is supported!")
                    continue

                market = self.local_market_dict.get(market_code)
                if not market:
                    self.logger.warning(f"Market '{market_code}' not found. Skipping price row.")
                    continue

                # TODO include gear code
                market.price_time_series[(species_code, gear_code, date)] = price

def ignore_non_csv(dir, files):
    return {f for f in files if not fnmatch.fnmatch(f, "*.csv")}
