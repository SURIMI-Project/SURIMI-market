import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../pb")))

from typing import List, Dict
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import datetime
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

    def __init__(self, port_name: str, port_id: str):
        self.name = port_name
        self.id = port_id  # UN Location Code


# market.py
class Market:
    """
    Represents a market that contains multiple ports and species.
    Also maintains an action log for tracking events.
    """

    def __init__(self, name: str, currency: str, measurement_unit: str):
        self.name = name
        self.ports_list: List[Port] = []
        self.species_list: List[Species] = []
        self.currency = currency  # 3-letter ISO 4217 currency code
        self.measurement_unit = measurement_unit  # Unit of measurement (e.g., kg, lb)
        self.tick = Tick()

    def add_port(self, port_name: str, port_id: str):
        """
        Adds a new port to the market.
        """
        self.ports_list.append(Port(port_name, port_id))

    def add_species(self, species_name:str, species_id: str):
        """
        Adds a new species to the market.
        """
        self.species_list.append(Species(species_name, species_id))

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

        for species in self.species_list:
            for port in self.ports_list:
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
