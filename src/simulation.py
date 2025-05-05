from src.market import Market
import threading
from surimi.v1 import workflow_pb2
from google.protobuf.timestamp_pb2 import Timestamp


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
        self.market.add_port("Barcelona", "ESBCN")  # Barcelona, Spain
        self.market.add_port("Valencia", "ESVLC")  # Valencia, Spain
        self.market.add_species("Tuna", "TUN")
        self.market.add_species("Salmon", "SAL")
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
        currentTimestamp = Timestamp()

        print(f"[Simulation] Current tick: {self.market.tick.get_timestamp()}")
        currentTimestamp.FromJsonString(self.market.tick.get_timestamp())

        return workflow_pb2.SimulateStepResponse(date_time=currentTimestamp)

    def sent_species_prices(self):
        """
        This method will handle prices updates
        and streaming updates triggered externally (by the client).
        """
        print("-" * 60)
        self.latest_market_prices = self.market.update_prices_message()
        print(f"[Simulation] ID: {self.simulation_id}, Prices have been sent ... ")




