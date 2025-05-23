import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../pb")))
import math
import datetime

from collections import defaultdict
from typing import Dict

from surimi.v1 import species_price_pb2, market_pb2
from google.protobuf.timestamp_pb2 import Timestamp

# tick.py
class Tick:
    """
    A global counter to track time progression in the system.
    """
    tick = 0
    start_time = datetime.datetime.utcnow()

    @staticmethod
    def increase_tick():
        """
        Increases the global tick counter by 1.
        Each tick represents one day in the simulation.
        """
        Tick.tick += 1

    @staticmethod
    def get_tick():
        """
        Returns the current tick value.
        """
        return Tick.tick

    @staticmethod
    def get_timestamp():
        """
        Returns the current simulation timestamp in RFC 3339 format.
        The simulation starts at the current UTC time and progresses by one day per tick.
        """
        current_time = (Tick.start_time + datetime.timedelta(days=Tick.tick)).isoformat() + "Z"
        return current_time


# species.py
class Species:
    """
    Represents a species within the market.
    A species has a name, type, quantity, and a species code.
    """

    def __init__(self, species_name: str, species_code: str):
        self.name = species_name
        self.id = species_code  # UN species code


# port.py
class Port:
    """
    Represents a port in the market.
    Each port has a unique identifier and a UN Location Code.
    """

    def __init__(self, port_name: str, port_id: str, parent_model: "Market"):
        self.name = port_name
        self.id = port_id  # UN Location Code
        self.local_market = LocalMarket(port_id, parent_model)

class GlobalMarket:
    """
    Represents the global market for all species, prices are fixed.
    """
    def __init__(self):
        self.prices: Dict[str, float] = {}  # species_id -> fixed price p

    def set_price(self, species_id: str, price: float):
        self.prices[species_id] = price

    def get_price(self, species_id: str) -> float:
       return self.prices.get(species_id, 0.0)

class LocalMarket:
    """
    Represents a local market at a specific port.
    Each local market has a unique identifier and tracks its own species.
    """

    def __init__(self, port_id: str, parent_model: "Market"):
        self.port_id = port_id
        self.parent_model = parent_model # Needed for cross-market influence
        self.species_price_params = {}  # species_id → { 'd': ..., 'beta': ... }
        self.monthly_catch = defaultdict(float) # species_id → quantity landed

    def set_price_parameters(self, species_id: str, local_base_price: float, beta: float):
        self.species_price_params[species_id] = {
            "d": local_base_price,
            "beta": beta,
        }

    def record_landing(self, species_id: str, quantity: float):
        self.daily_catch[species_id] += quantity

    def calculate_price(self, species_id: str) -> float:
        params = self.species_price_params[species_id]
        d = params["d"]
        beta = params["beta"]

        # Total catch across all ports (influence mechanism)
        total_catch = 0.0
        for port in self.parent_model.ports.values():
            total_catch += port.local_market.daily_catch[species_id]

        return d * math.exp(-beta * total_catch)

    def get_elasticity(self, species_id: str) -> float:
        p = self.parent_model.global_market.get_price(species_id)
        d = self.species_price_params[species_id]["d"]
        return 1 / math.log(p / d)



# market.py
class Market:
    """
    Represents a market that contains multiple ports and species.
    Also maintains an action log for tracking events.
    """

    def __init__(self, currency: str, measurement_unit: str):
        self.tick = Tick()
        self.ports_dict: Dict[str, Port] = {}
        self.species_dict: Dict[str, Species] = {}
        self.currency = currency  # 3-letter ISO 4217 currency code
        self.measurement_unit = measurement_unit  # Unit of measurement (e.g., kg, lb)
        self.global_market = GlobalMarket()


    def add_port(self, port_name: str, port_id: str):
        """
        Adds a new port to the market.
        """
        self.ports_dict.append(Port(port_name, port_id))

    def add_species(self, species_name:str, species_id: str):
        """
        Adds a new species to the market.
        """
        self.species_dict.append(Species(species_name, species_id))

    def set_local_market_parameters(self, port_id: str, species_id: str, d: float, beta: float):
        port = self.ports[port_id]
        port.local_market.set_price_parameters(species_id, d, beta)

    def catch_by_species(self, port_id: str, species_id: str, quantity: float):
        self.ports[port_id].local_market.record_landing(species_id, quantity)

    def calculate_species_price(self) -> float:
        """
        Calculates the price of a given species.
        Placeholder logic assumes price is based on quantity.
        """
        return (80.00 + random.randrange(1, 40))

    def update_species_prices_message(self):
        """
        Generates an update message containing pricing details for all species at all ports.
        """
        messages = []
        timestamp = Timestamp()
        timestamp.FromJsonString(self.tick.get_timestamp())

        for species in self.species_dict:
            for port in self.ports_dict:
                message = species_price_pb2.SpeciesPrice(
                    species_id=species.id,
                    port_id=port.id,
                    price=self.calculate_species_price(),
                    measurement_unit=self.measurement_unit,
                    currency=self.currency,
                    timestamp=timestamp
                )
                messages.append(message)
                print(f"[Market] Generated update: {species.id} at {port.id} - {message.price} "
                      f" {message.currency}/{message.measurement_unit} - {message.timestamp} ")

        return market_pb2.GetSpeciesPricesResponse( prices = messages)
