from src.market import Market
import threading

class Simulation:
    def __init__(self):
        """
        Static simulation parameters, will be replaced later
        """
        self.market = Market("Global Seafood Market", "EUR", "kg")
        self.market.add_port("Barcelona", "ESBCN")  # Barcelona, Spain
        self.market.add_port("Valencia", "ESVLC")  # Valencia, Spain
        self.market.add_species("Tuna", "TUN")
        self.market.add_species("Salmon", "SAL")
        self.update_event = threading.Event()
        self.latest_market_prices = None

    def run(self):
        """
        Method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def pick_simulation(simulation_type):
        """
        Pick the correct simulation based on the type.
        """
        if simulation_type == "1":
            return Simulation_1()
        elif simulation_type == "2":
            return Simulation_2()
        else:
            raise ValueError("Unknown simulation type")

    def PlotPricesOverTime(self):
        # Plot fish prices over time
        self.market.action_log.plot_data()

# Subclass of Simulation for Simulation 1
class Simulation_1(Simulation):
    def run(self):
        for _ in range(10):
            self.market.tick.increase_tick()
            print("-" * 60)
            print(f"Day {Tick.get_tick()} - {Tick.get_timestamp()}")
            self.latest_market_prices = self.market.send_update_prices_message()


# Subclass of Simulation for Simulation 2
class Simulation_2(Simulation):
    def __init__(self):
        super().__init__()

    def run(self):
        """
        Instead of automatically running for a fixed number of ticks,
        the simulation will wait for a trigger to stream updates.
        """
        print("[Simulation] Simulation_2 is running. Waiting for trigger from client.")

    def trigger_simulation(self):
        """
        This method will handle simulation steps
        and streaming updates triggered externally (by the client).
        """
        print("-" * 60)
        self.market.tick.increase_tick()
        print(f"[Simulation] Day {self.market.tick.get_tick()} has been triggered ")
        self.latest_market_prices = self.market.send_update_prices_message()
