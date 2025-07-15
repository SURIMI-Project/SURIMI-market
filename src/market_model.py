import sys
import os
import math
import datetime
import random

from collections import defaultdict
from typing import Dict, Tuple

from surimi.v1 import species_price_pb2, market_pb2
from google.protobuf.timestamp_pb2 import Timestamp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../pb")))

# TODO: setup via calling Scenario configuaration (via S3 bucket)

class Species:
    """
    Represents a species traded in the market.
    Maintains memory of last price and quantity to implement path-dependent price dynamics.
    Implements Elser et al. (2019) price adjustment:
        p_t = p_{t-1} * (q_{t-1} / q_t)^epsilon
    """
    def __init__(self, species_name: str, species_code: str, initial_price=10.0, initial_quantity=100.0, epsilon=0.5):
        self.name = species_name
        self.code = species_code
        self.last_price = initial_price
        self.last_quantity = initial_quantity
        self.epsilon = epsilon

    def calculate_price(self, quantity: float) -> float:
        """
        Calculates the price of the species based on the last price and quantity.
        """
        price = self.last_price * (self.last_quantity / max(quantity, 1e-6)) ** self.epsilon
        self.last_quantity = quantity
        self.last_price = price
        return price


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
        """
        Tick.tick += 1

    @staticmethod
    def get_tick():
        """
        Returns the current tick value.
        """
        return Tick.tick

    @staticmethod
    def set_timestep(days: int):
        """
        Sets the time step to a number of days
        """
        Tick.step_days = days

    @staticmethod
    def get_timestamp():
        """
        Returns the current simulation timestamp in RFC 3339 format.
        The simulation starts at the current UTC time and progresses one tick per time step.
        """
        current_time = (Tick.start_time + datetime.timedelta(days=Tick.tick * getattr(Tick, 'step_days', 1))).isoformat() + "Z"
        return current_time

    @staticmethod
    def reset():
        """
        Resets the tick counter and start time to initial values.
        """
        Tick.tick = 0
        Tick.start_time = datetime.datetime.utcnow()


class MarketSystem:
    """
    Represents the overall simulation environment for dynamic fish markets.
    Combines multiple LocalMarkets and a GlobalMarket for multi-level pricing.

    Attributes:
        currency (str): 3-letter ISO 4217 currency code.
        measurement_unit (str): Unit of measurement (e.g., kg, lb).
        species_dict (dict): Maps species codes to Species objects.
        local_markets (dict): Maps market codes to LocalMarket objects.
        global_market (GlobalMarket): Shared global market logic across all species.
    """
    def __init__(self, currency: str, measurement_unit: str):
        self.tick = Tick()
        self.currency = currency
        self.measurement_unit = measurement_unit
        self.species_dict: Dict[str, Species] = {}
        self.local_markets: Dict[str, LocalMarket] = {}
        self.global_market = GlobalMarket(self.species_dict)

    def add_species(self, species: Species):
        """
        Adds a new species to the MarketSystem.
        """
        self.species_dict[species.code] = species

    def add_local_market(self, market: object):
        """
        Adds a new local market to the MarketSystem.
        """
        self.local_markets[market.market_code] = market

    def add_species_to_local_market(self, market_code: str, species_code: str):
        """
        Adds a single species to a given local market, if not already present.
        """
        local_market = self.local_markets[market_code]
        species = self.species_dict.get(species_code)
        if species and species_code not in local_market.species_dict:
            local_market.species_dict[species_code] = species

    def catch_by_species(self, market_code: str, species_code: str, quantity: float):
        self.local_markets[market_code].record_monthly_catch(species_code, quantity)

    def get_species_prices_message(self):
        """
        Generates an update message containing pricing details for all species at all ports.
        """
        messages = []
        timestamp = Timestamp()
        timestamp.FromJsonString(self.tick.get_timestamp())

        for species in self.species_dict.values():
            for market in self.local_markets.values():
                message = species_price_pb2.SpeciesPrice(
                    species_code=species.code,
                    market_code=market.market_code,
                    price=market.local_prices.get(species.code),
                    measurement_unit=self.measurement_unit,
                    currency=self.currency,
                    timestamp=timestamp
                )
                messages.append(message)
                print(f"[Market] Generated update: {species.code} at {market.market_code} - {message.price} "
                      f"{message.currency}/{message.measurement_unit} - {message.timestamp} ")

        return market_pb2.GetSpeciesPricesResponse(prices=messages)


class GlobalMarket:
    """
    Represents the global market context, computing global species prices
    based on aggregate supply using species-specific dynamics.
    Prices update each tick based on total quantity and previous state.
    """
    def __init__(self, species_dict):
        self.species_dict = species_dict
        self.global_prices: Dict[str, float] = {}

    def compute_price(self, species_code: str, total_quantity: float) -> float:
        species = self.species_dict[species_code]
        price = species.calculate_price(total_quantity)
        self.global_prices[species_code] = price
        return price

    def get_global_price(self, species_code: str) -> float:
        return self.global_prices.get(species_code, 0.0)


class LocalMarket:
    """
    Represents a local fish market identified by a market code (UN/LOCODE).
    Computes local prices influenced by local supply, global prices (lambda_mix),
    and neighboring markets (theta_mix).

    Attributes:
        market_code (str): ISO 3166-1 alpha-2 + 3-letter port code.
        global_market (GlobalMarket): Reference to the shared global market.
        lambda_mix (float): Influence of global price on local price [0, 1].
        theta_mix (float): Influence of neighboring markets on local price [0, 1].
    """
    def __init__(self, market_code: str, global_market: object,
                 name: str = None, lon: float = None, lat: float = None,
                 lambda_mix=0.0, theta_mix=0.0):
        self.market_code = market_code
        self.market_name = name
        self.lon = lon
        self.lat = lat
        self.global_market = global_market
        self.lambda_mix = lambda_mix
        self.theta_mix = theta_mix

        self.species_dict = global_market.species_dict.copy()
        self.monthly_catch = defaultdict(float)
        self.local_prices: Dict[str, float] = {}
        self.price_history = []
        self.price_time_series: Dict[Tuple[str, str], float] = {}

    def record_monthly_catch(self, species_code: str, quantity: float):
        self.monthly_catch[species_code] = quantity

    def calculate_local_price(self, species_code: str = None, other_market_avg: Dict[str, float] = None):
        """
        Prices are influenced by:
            - Local quantity caught
            - Global market price (weighted by lambda_mix)
            - Average price from other markets (weighted by theta_mix)

        Args:
            species_code (str, optional): If provided, updates only this species. Otherwise, updates all species.
            other_market_avg (Dict[str, float], optional): Average prices from other local markets, used for theta_mix.

        Returns:
            Dict[str, float]: Updated local prices for each species.
        """

        prices = {}
        species_to_process = [species_code] if species_code else list(self.species_dict.keys())

        for code in species_to_process:
            quantity = self.monthly_catch.get(code, 0.0)
            local_species = self.species_dict[code]
            p_local = local_species.calculate_price(quantity)

            if self.lambda_mix > 0:
                p_global = self.global_market.get_global_price(code)
                p_local = (1 - self.lambda_mix) * p_local + self.lambda_mix * p_global

            if other_market_avg and self.theta_mix > 0:
                p_other = other_market_avg.get(code, p_local)
                p_local = (1 - self.theta_mix) * p_local + self.theta_mix * p_other

            self.local_prices[code] = p_local
            prices[code] = p_local

        self.price_history.append((len(self.price_history), self.local_prices.copy()))
        return prices

    def set_price_from_history(self, date_str: str):
        """
        Inject historical prices for the given date into local_prices if available.
        """
        for species_code in self.species_dict:
            key = (species_code, date_str)
            if key in self.price_time_series:
                self.local_prices[species_code] = self.price_time_series[key]

    def get_elasticity(self, species_code: str) -> float:
        p = self.global_market.get_global_price(species_code)
        d = self.species_dict[species_code].last_price
        return 1 / math.log(p / d)
