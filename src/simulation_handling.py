from src.market_model import MarketSystem, LocalMarket, Species
from surimi.v1 import workflow_pb2, market_pb2
import threading
import json

class SimulationHandler:
    def __init__(self, simulation_id, scenario_id, start_time, step_size):
        """
        Manages the simulation loop, using MarketSystem as the core.
        """
        self.simulation_id = simulation_id
        self.scenario_id = scenario_id
        self.start_time = start_time
        self.step_size = step_size

        # Initialize market system
        self.market = MarketSystem(currency="EUR", measurement_unit="kg")

        # Add species
        # self.market.add_species(Species("Bogue", "BOG"))
        # self.market.add_species(Species("Pilchard", "PIL"))

        # Add local markets (using UN/LOCODE)
        # self.market.add_local_market(LocalMarket("ESARN", self.market.global_market))  # ARENYS DE MAR, Spain
        # self.market.add_local_market(LocalMarket("ESBLA", self.market.global_market))  # BLANES, Spain


        self.update_event = threading.Event()

        self.latest_market_prices = None
        # TODO: Initialize the market system with species and local markets

    def init_simulation(self):
        """
        Trigger simulation initialization and responds to controller.
        """
        print("-" * 60)
        self.market.tick.increase_tick()
        print(f"[Simulation] ID: {self.simulation_id}, Day {self.market.tick.get_tick()} has been initialized ... ")
        return workflow_pb2.InitialiseResponse(
            simulation_id=self.simulation_id
        )

    def finalise_simulation(self):
        """
        Trigger simulation finalisation and responds to controller.
        """
        return workflow_pb2.FinaliseResponse(
            simulation_id=self.simulation_id
        )

    def simulate_step(self):
        # TODO: This should be sending a blocking response to the controller if sales we not received.
        """
        Trigger a simulation step and respond to the controller.
        """
        print("-" * 60)
        self.market.tick.increase_tick()
        print(f"[Simulation] ID: {self.simulation_id}, Day {self.market.tick.get_tick()} has been triggered ... ")

        # Compute global prices
        for species_code in self.market.species_dict:
            total_quantity = sum(m.monthly_catch[species_code] for m in self.market.local_markets.values())
            self.market.global_market.compute_price(species_code, total_quantity)

        # Compute inter-market averages (for theta_mix)
        avg_prices = {}
        for s in self.market.species_dict.values():
            avg_prices[s.code] = sum(
                m.local_prices.get(s.code, s.last_price) for m in self.market.local_markets.values()
            ) / len(self.market.local_markets)

        # Compute local prices
        for m in self.market.local_markets.values():
            m.calculate_local_price(other_market_avg=avg_prices)

        return workflow_pb2.SimulateStepResponse(
            simulation_id=self.simulation_id
        )

    def update_prices(self):
        """
        Send current prices to the controller.
        """
        print("-" * 60)
        self.latest_market_prices = self.market.update_species_prices_message()
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
        return market_pb2.UpdateSalesResponse(
            simulation_id=self.simulation_id
        )

    def load_from_json(self, file_path: str):
        """
        Import Jason file to initialize the simulation with species and local markets.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        for s in data.get("species", []):
            species = Species(species_name=s["name"], species_code=s["code"])
            self.add_species(species)

        for m in data.get("local_market", []):
            market = LocalMarket(market_name=m["name"], market_code=m["code"], global_market=self.market.global_market)
            self.add_local_market(market)
