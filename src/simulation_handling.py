import fnmatch
from pathlib import Path
import shutil
import threading
import csv
import os
from collections import defaultdict

from src.market_model import MarketSystem, LocalMarket, Species
from surimi.v1 import workflow_service_pb2, market_provider_service_pb2, simulation_pb2
from logger import get_logger


class SimulationHandler:
    def __init__(self, simulation_id, scenario_id, start_time, step_size):
        """
        Manages the simulation loop, using MarketSystem as the core.
        """
        self.simulation_id = simulation_id
        self.scenario_id = scenario_id
        self.start_time = start_time
        self.step_size = step_size

        self.logger = get_logger(f"simulation.{simulation_id}")

        # Initialize market system with default currency and measurement unit
        self.market = MarketSystem(currency="EUR", start_time=start_time)

        # Internal dictionaries for quick reference
        self.species_dict = {}
        self.local_market_dict = {}

        # Threading event for blocking control
        self.update_event = threading.Event()

        # Cached message for latest prices
        self.latest_market_prices = None

        # Aggregated catches for the current tick
        # key = (market_code, species_code) -> total quantity
        self.pending_catch_dict = defaultdict(float)

        # Flag to detect concurrent SimulateStep and UpdateSales calls
        self._step_in_progress = False

    def init_simulation(self, simulation: simulation_pb2.Simulation):
        """
        Trigger simulation initialization and respond to controller.
        Loads initialization config (species, ports, historical prices)
        and sets baseline prices for each market.
        """
        simulation_dir = Path(__file__).parent.parent / "simulations" / self.simulation_id
        simulation_dir.mkdir(parents=True, exist_ok=True)
        config_src = Path(__file__).parent.parent / "config"
        config_dst = simulation_dir / "config"
        if config_dst.exists():
            shutil.rmtree(config_dst)
        shutil.copytree(config_src, config_dst, ignore=ignore_non_csv)

        self.load_initialization_data("config")

        missing_species = []
        for species in simulation.items.species:
            if species.species_code not in self.species_dict:
                missing_species.append(species.species_code)

        if missing_species:
            raise ValueError(
                f"Species codes from simulation contract not found in loaded species configuration: {missing_species}"
            )

        self.logger.info(f"Tick {self.market.tick.get_tick()} initialized.")
        return workflow_service_pb2.InitialiseResponse(simulation_id=self.simulation_id)

    def finalise_simulation(self, folder_path="simulations"):
        """
        Trigger simulation finalisation and respond to controller.
        Writes results into simulations/<simulation_id>.
        """
        self.logger.info("Finalising simulation...")
        self.market.export_results(self.simulation_id, folder_path)
        self.logger.info(f"Results saved in simulations/{self.simulation_id}")
        return workflow_service_pb2.FinaliseResponse(simulation_id=self.simulation_id)

    def simulate_step(self):
        """
        Trigger a simulation step.
        Uses all buffered sales for this tick.
        """
        self.logger.info("Simulate step triggered...")
        self._step_in_progress = True
        try:
            catches = [(m, s, qty) for (m, s), qty in self.pending_catch_dict.items()]
            self.market.step(catches)

            total_qty = sum(qty for _, _, qty in catches)
            self.logger.info(f"Processed {len(catches)} market/species pairs, total_qty={total_qty:.2f} kg")

            self.pending_catch_dict.clear()
        finally:
            self._step_in_progress = False

        tick = self.market.tick.get_tick()
        current_date = self.market.tick.get_timestamp()[:10]
        self.logger.info(f"Tick {tick} ({current_date}) completed.")

        return workflow_service_pb2.SimulateStepResponse(simulation_id=self.simulation_id)

    def update_prices(self):
        """
        Send current prices to the controller.
        """
        self.latest_market_prices = self.market.get_species_prices_message()
        self.logger.info("Prices sent.")
        return self.latest_market_prices

    def update_sales(self, request):
        """
        Receives external sales (catches) from controller.
        Aggregates them into pending_catch_dict for this tick.
        """
        self.logger.info("Sales update received...")

        if self._step_in_progress:
            self.logger.warning(
                "UpdateSales called while SimulateStep is in progress. Sale may be lost."
            )

        new_entries = 0

        for summary in request.sales_summaries:
            market_code = summary.market_code

            for sale in summary.sales:
                species_code = sale.species.species_code
                quantity = sale.quantity
                gear_code = sale.fleet_segment.gear_code

                key = (market_code, species_code)
                self.pending_catch_dict[key] += quantity
                new_entries += 1

                self.logger.info(
                    f"Sale: market={market_code}, species={species_code}, gear={gear_code}, qty={quantity:.2f} kg"
                )

        self.logger.info(f"Processed {new_entries} sales.")
        self.logger.info(f"Currently buffered {len(self.pending_catch_dict)} unique market/species combos.")

        return market_provider_service_pb2.UpdateSalesResponse(simulation_id=self.simulation_id)

    def load_species_from_csv(self, filepath: str):
        """
        Load species definitions from CSV file into the market system.

        CSV columns: species_code, species_name
        """
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                code = row["species_code"].strip()
                name = row["species_name"].strip()
                species = Species(species_name=name, species_code=code)
                self.market.add_species(species)
                self.species_dict[code] = species

    def load_ports_from_csv(self, filepath: str):
        """
        Load local market (port) definitions from CSV into the market system.

        CSV columns: port_code, port_name, lon, lat
        """
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                code = row["port_code"].strip()
                name = row["port_name"].strip()
                lon = float(row["lon"])
                lat = float(row["lat"])

                market = LocalMarket(
                    market_code=code,
                    global_market=self.market.global_market,
                    name=name,
                    lon=lon,
                    lat=lat
                )
                self.market.add_local_market(market)
                self.local_market_dict[code] = market

    def load_prices_from_csv(self, filepath: str):
        """
        Load time-series prices into each local market from CSV file.

        CSV columns: date, market_code, species_code, category_code, price, currency, measurement_unit
        """
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                date = row["date"].strip()
                market_code = row["market_code"].strip()
                species_code = row["species_code"].strip()
                price = float(row["price"])

                market = self.local_market_dict.get(market_code)
                if not market:
                    self.logger.warning(f"Market '{market_code}' not found. Skipping price row.")
                    continue

                market.price_time_series[(species_code, date)] = price

    def load_initialization_data(self, folder_path="config"):
        """
        Loads all initial simulation input (species, ports, prices),
        then initializes baseline prices using Option B (CSV -> global avg -> fallback).
        """
        app_directory = Path(__file__).parent.parent.resolve()
        folder_path = str(Path(app_directory) / folder_path)

        species_path = os.path.join(folder_path, "species.csv")
        ports_path = os.path.join(folder_path, "ports.csv")
        prices_path = os.path.join(folder_path, "prices.csv")

        if os.path.exists(species_path):
            self.logger.info(f"Loading species from {species_path}")
            self.load_species_from_csv(species_path)
        else:
            self.logger.warning(f"species.csv not found in {folder_path}")

        if os.path.exists(ports_path):
            self.logger.info(f"Loading ports from {ports_path}")
            self.load_ports_from_csv(ports_path)
        else:
            self.logger.warning(f"ports.csv not found in {folder_path}")

        if os.path.exists(prices_path):
            self.logger.info(f"Loading prices from {prices_path}")
            self.load_prices_from_csv(prices_path)
        else:
            self.logger.info("prices.csv not found — skipping price injection")

        self.market._initialize_baseline_prices_from_csv()


def ignore_non_csv(dir, files):
    return {f for f in files if not fnmatch.fnmatch(f, "*.csv")}
