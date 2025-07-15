from src.market_model import MarketSystem, LocalMarket, Species
from surimi.v1 import workflow_pb2, market_pb2

import threading
import csv
import os

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
        """
        print("-" * 60)
        # Todo: This needs to be done via selection from controller. Do we need to change proto message?
        self.load_data_from_folder("../data/input")  # Load species, ports, prices

        print(f"[Simulation] ID: {self.simulation_id}, Month {self.market.tick.get_tick()} has been initialized ... ")
        return workflow_pb2.InitialiseResponse(simulation_id=self.simulation_id)

    def finalise_simulation(self):
        """
        Trigger simulation finalisation and respond to controller.
        """
        return workflow_pb2.FinaliseResponse(simulation_id=self.simulation_id)

    def simulate_step(self, catches=None):
        # TODO: This should be sending a blocking response to the controller if sales we not received.

        """
        Trigger a simulation step and respond to the controller.
        Uses MarketSystem.step() to update all prices and advance time.
        """
        print("-" * 60)

        # Default: no new catches
        if catches is None:
            catches = []

        # Run the actual monthly market update
        self.market.step(catches)

        # Get the current tick and timestamp from this MarketSystem's clock
        tick = self.market.tick.get_tick()
        current_date = self.market.tick.get_timestamp()[:10]  # 'YYYY-MM-DD'

        print(f"[Simulation] ID: {self.simulation_id}, Month {tick} ({current_date}) has been triggered ... ")

        return workflow_pb2.SimulateStepResponse(simulation_id=self.simulation_id)

    def update_prices(self):
        """
        Send current prices to the controller.
        """
        print("-" * 60)
        self.latest_market_prices = self.market.get_species_prices_message()
        print(f"[Simulation] ID: {self.simulation_id}, Prices have been sent ... ")
        return self.latest_market_prices

    def update_sales(self):
        # TODO: This should be receiving sales data from the controller twice (EWE and Poseidon).
        #  Should be aggregate values until simulate Step is called.
        """
        Placeholder for external catch input. In real use, values will come from outside agents or models.
        """
        print("-" * 60)
        print(f"[Simulation] ID: {self.simulation_id}, Sales have been received ... ")
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
                    print(f"⚠️ Market '{market_code}' not found. Skipping.")
                    continue

                if not hasattr(market, "price_time_series"):
                    market.price_time_series = {}

                market.price_time_series[(species_code, date)] = price

    def load_data_from_folder(self, folder_path="data/input"):
        """
        Loads all required simulation input data (species, ports, prices) from the given folder.

        Expects:
        - species.csv
        - ports.csv
        - prices.csv
        """
        folder_path = os.path.abspath(folder_path)

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
