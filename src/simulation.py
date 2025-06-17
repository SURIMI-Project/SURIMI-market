from src.market import MarketSystem, LocalMarket, Species
from surimi.v1 import workflow_pb2, market_pb2
import threading

class Simulation:
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
        self.market.add_species(Species("Bogue", "BOG"))
        self.market.add_species(Species("Pilchard", "PIL"))

        # Add local markets (using UN/LOCODE)
        self.market.add_local_market(LocalMarket("ESARN", self.market.global_market))  # ARENYS DE MAR, Spain
        self.market.add_local_market(LocalMarket("ESBLA", self.market.global_market))  # BLANES, Spain

        self.update_event = threading.Event()
        self.latest_market_prices = None

    def init_simulation(self):
        """
        Trigger simulation initialization and responds to controller.
        """
        print("-" * 60)
        self.market.tick.increase_tick()
        print(f"[Simulation] ID: {self.simulation_id}, Day {self.market.tick.get_tick()} has been initialized ... ")
        return workflow_pb2.InitResponse()

    def simulate_step(self):
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

        return workflow_pb2.SimulateStepResponse()

    def update_prices(self):
        """
        Send current prices to the controller.
        """
        print("-" * 60)
        self.latest_market_prices = self.market.update_species_prices_message()
        print(f"[Simulation] ID: {self.simulation_id}, Prices have been sent ... ")
        return self.latest_market_prices

    def update_sales(self):
        """
        Placeholder for external catch input. In real use, values will come from outside agents or models.
        """
        print("-" * 60)
        print(f"[Simulation] ID: {self.simulation_id}, Sales have been received ... ")
        return market_pb2.UpdateSalesResponse()
