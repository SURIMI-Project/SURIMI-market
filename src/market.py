import sys
import os
import math
import datetime
import random

from collections import defaultdict
from typing import Dict

from surimi.v1 import species_price_pb2, market_pb2
from google.protobuf.timestamp_pb2 import Timestamp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../pb")))


# market.py
class Market:
    """
    Represents a market that contains multiple ports and species.
    Also maintains an action log for tracking events.
    """

    def __init__(self, currency: str, measurement_unit: str):
        self.tick = Tick()
        self.currency = currency  # 3-letter ISO 4217 currency code
        self.measurement_unit = measurement_unit  # Unit of measurement (e.g., kg, lb)
        self.species_dict: Dict[str, dict] = {} # species_code -> Species
        self.ports_dict: Dict[str, Port] = {} # port_code -> Port
        self.global_market = GlobalMarket() # Global market prices

    def add_species(self, species: object):
        """
        Adds a new species to the market.
        """
        self.species_dict[species.code] = species

    def add_port(self, port: object):
        """
        Adds a new port to the market.
        """
        self.ports_dict[port.code] = port

    def add_species_to_local_market(self, port_code: str, species_code: str):
        port = self.ports_dict[port_code]
        port.local_market.species_dict = self.species_dict.copy() # Copy species to local market

    def catch_by_species(self, port_code: str, species_code: str, quantity: float):
        self.ports_dict[port_code].local_market.record_monthly_catch(species_code, quantity)

    def calculate_species_price(self) -> float:
        """
        Calculates the price of a given species.
        Placeholder logic assumes price is based on quantity.
        """
        return (80.00 + random.randrange(1, 40))
        # TODO: add proper price calculation logic

    def update_species_prices_message(self):
        """
        Generates an update message containing pricing details for all species at all ports.
        """
        messages = []
        timestamp = Timestamp()
        timestamp.FromJsonString(self.tick.get_timestamp())

        for species in self.species_dict.values():
            for port in self.ports_dict.values():
                message = species_price_pb2.SpeciesPrice(
                    species_code=species.code,
                    port_code=port.code,
                    price=port.local_market.local_prices.get(species.code),
                    measurement_unit=self.measurement_unit,
                    currency=self.currency,
                    timestamp=timestamp
                )
                messages.append(message)
                print(f"[Market] Generated update: {species.code} at {port.code} - {message.price} "
                      f" {message.currency}/{message.measurement_unit} - {message.timestamp} ")

        return market_pb2.GetSpeciesPricesResponse( prices = messages)

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
        self.code = species_code  # UN species code


class GlobalMarket:
    """
    Represents the global market for all species, prices are fixed.
    """
    def __init__(self):
        self.global_prices: Dict[str, float] = {}  # species_code -> fixed price p
        self.local_markets: Dict[str, object] = {}  # port_code -> LocalMarket

    def set_global_price(self, species_code: str, price: float):
        self.global_prices[species_code] = price

    def get_global_price(self, species_code: str) -> float:
       return self.global_prices.get(species_code)

# port.py
class Port:
    """
    Represents a port in the market.
    Each port has a unique codeentifier and a UN Location Code.
    """

    def __init__(self, port_name: str, port_code: str, global_market: object):
        self.name = port_name
        self.code = port_code  # UN Location Code
        self.local_market = LocalMarket(port_code, global_market)

class LocalMarket:
    """
    Represents a local market at a specific port.
    Each local market has a unique identifier and tracks its own species.
    """

    def __init__(self, port: str, global_market: object):
        self.port = port
        self.global_market = global_market # Needed for cross-market influence
        self.species_dict = global_market.species_dict.copy() # Dict
        self.monthly_catch = defaultdict(float)  # species_code -> quantity caught
        self.local_prices: Dict[str, float] = {}
        self.price_calc_params = {}


    def set_price_parameters(self, species_code: str, local_base_price: float, beta: float):
        self.price_calc_params[species_code] = {
            "d": local_base_price,
            "beta": beta,
        }

    def record_monthly_catch(self, species_code: str, quantity: float):
        self.monthly_catch[species_code] = quantity

    def calculate_local_price(self, species_code: str):

        for species_code in self.species_dict:
            params = self.price_calc_params[species_code]
            d = params["d"]
            beta = params["beta"]
            q = self.monthly_catch.get(species_code, 0.0)

            price = d * math.exp(-beta * q)
            self.local_prices[species_code] = price

        return price


    def get_elasticity(self, species_code: str) -> float:
        p = self.global_market.get_global_price(species_code)
        d = self.price_calc_params[species_code]["d"]
        return 1 / math.log(p / d)