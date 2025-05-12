from src.market import Market
import threading
from surimi.v1 import workflow_pb2

class Simulation:
    def __init__(self, simulation_id, scenario_id, start_time, step_size):
        """
        Static simulation parameters, will be replaced later
        """
        self.simulation_id = simulation_id
        self.scenario_id = scenario_id
        self.start_time = start_time
        self.step_size = step_size
        # add scenario parameters
        self.market = Market("Global Seafood Market", "EUR", "kg")
        self.market.add_port("ARENYS DE MAR", "ESARN")  # ARENYS DE MAR, Spain
        self.market.add_port("BLANES", "ESBLA")  # BLANES, Spain
        self.market.add_species("Bogue", "BOG")
        self.market.add_species("Pilchard", "PIL")
        self.update_event = threading.Event()
        self.latest_market_prices = None

    def init_response(self):
        """
        This method will handle simulation initialization
        and streaming updates triggered externally (by the client).
        """
        print("-" * 60)
        self.market.tick.increase_tick()
        print(f"[Simulation] ID: {self.simulation_id}, Day {self.market.tick.get_tick()} has been initialized ... ")

    def simulate_step(self):
        """
        This method will handle simulation steps
        and streaming updates triggered externally (by the client).
        """
        print("-" * 60)
        self.market.tick.increase_tick()
        print(f"[Simulation] ID: {self.simulation_id}, Day {self.market.tick.get_tick()} has been triggered ... ")
        return workflow_pb2.SimulateStepResponse()

    def send_species_prices(self):
        """
        This method will handle prices updates
        and streaming updates triggered externally (by the client).
        """
        print("-" * 60)
        self.latest_market_prices = self.market.update_species_prices_message()
        print(f"[Simulation] ID: {self.simulation_id}, Prices have been sent ... ")
        return self.latest_market_prices




