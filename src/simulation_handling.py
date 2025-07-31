import fnmatch
from pathlib import Path
import shutil
import threading
import csv
import os
import sys

from src.market_model import MarketSystem, LocalMarket, Species
from surimi.v1 import workflow_pb2, market_pb2

class SimulationHandler:
    def __init__(self, simulation_id, scenario_id, start_time, step_size):
        """
        Manages the simulation loop, using MarketSystem as the core.
        """
        self.simulation_id = simulation_id
        self.scenario_id = scenario_id
        self.start_time = start_time
        self.step_size = step_size

        # Initialize market system with default currency and measurement unit
        self.market = MarketSystem(currency="EUR", measurement_unit="kg")

        # Internal dictionaries for quick reference
        self.species_dict = {}
        self.local_market_dict = {}

        # Threading event for blocking control
        self.update_event = threading.Event()

        # Cached message for latest prices
        self.latest_market_prices = None

    def init_simulation(self):
        """
        Trigger simulation initialization and respond to controller.
        Loads initialization config (species, ports, historical prices)
        and sets baseline prices for each market.
        """

        # Create simulation directory if it does not exist, and copy config files into it
        simulation_dir = Path(__file__).parent.parent / "simulations" / self.simulation_id
        simulation_dir.mkdir(parents=True, exist_ok=True)
        config_src = Path(__file__).parent.parent / "config"
        config_dst = simulation_dir / "config"
        if config_dst.exists():
            shutil.rmtree(config_dst)
        shutil.copytree(config_src, config_dst, ignore=ignore_non_csv)

        # Todo: This needs to be done via selection from controller. Do we need to change proto message?
        print("-" * 60)
        # Load all baseline config (species, ports, historical prices)
        self.load_initialization_data("config")

        print(f"[Simulation] ID: {self.simulation_id}, Tick {self.market.tick.get_tick()} initialized ...")
        return workflow_pb2.InitialiseResponse(simulation_id=self.simulation_id)

    def finalise_simulation(self, folder_path="simulations"):
        """
        Trigger simulation finalisation and respond to controller.
        Writes results into simulations/<simulation_id>.
        """
        print(f"[Simulation] Finalising {self.simulation_id}...")

        # Export results after final tick
        self.market.export_results(self.simulation_id, folder_path)

        print(f"[Simulation] Results saved in simulations/{self.simulation_id}")
        return workflow_pb2.FinaliseResponse(simulation_id=self.simulation_id)


    def simulate_step(self):
        """
        Trigger a simulation step and respond to the controller.
        """
        print("-" * 60)

        # Perform one monthly simulation step
        self.market.step([])
        tick = self.market.tick.get_tick()
        current_date = self.market.tick.get_timestamp()[:10]
        print(f"[Simulation] ID: {self.simulation_id}, Tick {tick} ({current_date}) triggered ...")

        return workflow_pb2.SimulateStepResponse(simulation_id=self.simulation_id)

    def update_prices(self):
        """
        Send current prices to the controller.
        """
        print("-" * 60)
        self.latest_market_prices = self.market.get_species_prices_message()
        print(f"[Simulation] ID: {self.simulation_id}, Prices sent ...")
        return self.latest_market_prices

    def update_sales(self):
        # TODO: This should be receiving sales config from the controller twice (EWE and Poseidon).
        #  Should be aggregate values until simulate Step is called.
        """
        Placeholder for external catch input (future extension).
        """
        print("-" * 60)
        print(f"[Simulation] ID: {self.simulation_id}, Sales received ...")
        return market_pb2.UpdateSalesResponse(simulation_id=self.simulation_id)

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
                    print(f"⚠️ Market '{market_code}' not found. Skipping price row.")
                    continue

                # Save into the market's price_time_series
                market.price_time_series[(species_code, date)] = price

    def load_initialization_data(self, folder_path="config"):
        """
        Loads all initial simulation input (species, ports, prices),
        then initializes baseline prices using Option B (CSV → global avg → fallback).
        """
        app_directory = Path(__file__).parent.parent.resolve()  # Get the parent directory of the current file
        folder_path = str(Path(app_directory) / folder_path) # Combine the app directory with the folder path

        species_path = os.path.join(folder_path, "species.csv")
        ports_path = os.path.join(folder_path, "ports.csv")
        prices_path = os.path.join(folder_path, "prices.csv")

        if os.path.exists(species_path):
            print(f"📥 Loading species from {species_path}")
            self.load_species_from_csv(species_path)
        else:
            print(f"⚠️ species.csv not found in {folder_path}")

        if os.path.exists(ports_path):
            print(f"📥 Loading ports from {ports_path}")
            self.load_ports_from_csv(ports_path)
        else:
            print(f"⚠️ ports.csv not found in {folder_path}")

        if os.path.exists(prices_path):
            print(f"📥 Loading prices from {prices_path}")
            self.load_prices_from_csv(prices_path)
        else:
            print(f"ℹ️ prices.csv not found — skipping price injection")

        # After loading CSV config, initialize baseline prices
        self.market._initialize_baseline_prices_from_csv()

# Function to ignore non-CSV files when copying config. Like .gitkeep files
def ignore_non_csv(dir, files):
    return {f for f in files if not fnmatch.fnmatch(f, "*.csv")}
