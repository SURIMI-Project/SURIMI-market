import sys
import os
import math
import datetime
from collections import defaultdict
from typing import Dict, Tuple

from surimi.v1 import species_price_pb2, market_pb2, species_pb2
from google.protobuf.timestamp_pb2 import Timestamp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../pb")))

# TODO: setup via calling Scenario configuaration (via S3 bucket)

class Tick:
    """
    A per-MarketSystem time tracker.
    Each MarketSystem gets its own Tick instance.
    """
    def __init__(self, step_days=30):
        self.tick = 0
        self.start_time = datetime.datetime.utcnow()
        self.step_days = step_days

    def increase_tick(self):
        self.tick += 1

    def get_tick(self):
        return self.tick

    def set_timestep(self, days: int):
        self.step_days = days

    def get_timestamp(self):
        current_time = (self.start_time +
                        datetime.timedelta(days=self.tick * self.step_days)).isoformat() + "Z"
        return current_time

    def reset(self):
        self.tick = 0
        self.start_time = datetime.datetime.utcnow()


class Species:
    """
    Represents a species traded in the market.
    Maintains epsilon for price elasticity.
    Price path memory is handled per LocalMarket.
    """
    def __init__(self, species_name: str, species_code: str, epsilon=0.5):
        self.name = species_name
        self.code = species_code
        self.epsilon = epsilon


class GlobalMarket:
    """
    Represents the global market context, computing global species prices
    as an average of local prices.
    """
    def __init__(self, species_dict):
        self.species_dict = species_dict
        self.global_prices: Dict[str, float] = {}
        self.global_price_history = []  # stores (tick, {species_code: price,...})

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

        self.species_dict = global_market.species_dict  # shared reference
        self.monthly_catch = defaultdict(float)
        self.local_prices: Dict[str, float] = {}

        # Per-species price memory for Elser price adjustment
        self.price_state: Dict[str, Tuple[float, float]] = {}  # species_code -> (last_price, last_quantity)

        # For history
        self.price_history = []  # [(tick, {species: price, ...})]

    def record_monthly_catch(self, species_code: str, quantity: float):
        self.monthly_catch[species_code] += quantity

    def get_last_state(self, s_code: str) -> Tuple[float, float]:
        """
        If no price state yet, fall back to initial price from CSV and dummy qty=1.
        """
        if s_code not in self.price_state:
            init_price = self.local_prices.get(s_code, 1.0)
            return init_price, 1.0
        return self.price_state[s_code]

    def set_new_state(self, s_code: str, price: float, qty: float):
        """
        Save updated price state for next tick.
        """
        self.price_state[s_code] = (price, qty)
        self.local_prices[s_code] = price

    def calculate_local_price(self, species_code: str = None, other_market_avg: Dict[str, float] = None):
        """
        KEPT for potential lambda/theta mixing, but Elser updates are now done centrally.
        """
        prices = {}
        species_to_process = [species_code] if species_code else list(self.species_dict.keys())

        for code in species_to_process:
            # Just use the current stored local price (no Elser recalculation here)
            p_local = self.local_prices.get(code, 0.0)

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


class MarketSystem:
    """
    Represents the overall simulation environment for dynamic fish markets.
    Combines multiple LocalMarkets and a GlobalMarket for multi-level pricing.
    """
    def __init__(self, currency: str, measurement_unit: str):
        self.tick = Tick(step_days=30)  # each MarketSystem has its own timeline
        self.currency = currency
        self.measurement_unit = measurement_unit
        self.species_dict: Dict[str, Species] = {}
        self.local_markets: Dict[str, LocalMarket] = {}
        self.global_market = GlobalMarket(self.species_dict)

    def add_species(self, species: Species):
        self.species_dict[species.code] = species

    def add_local_market(self, market: object):
        self.local_markets[market.market_code] = market

    def add_species_to_local_market(self, market_code: str, species_code: str):
        local_market = self.local_markets[market_code]
        species = self.species_dict.get(species_code)
        if species and species_code not in local_market.species_dict:
            local_market.species_dict[species_code] = species

    def catch_by_species(self, market_code: str, species_code: str, quantity: float):
        self.local_markets[market_code].record_monthly_catch(species_code, quantity)

    def step(self, catches: list):
        """
        One monthly update step:
        1) Record catches
        2) Update local prices
        3) Save history (prices + catches) and clear monthly catches
        4) Compute global prices
        5) Advance simulation tick
        """
        self._record_catches(catches)
        self._update_local_prices()
        self._update_local_history()
        self._compute_global_prices()
        self.tick.increase_tick()

        # Reset monthly_catch after snapshotting (moved here)
        for market in self.local_markets.values():
            market.monthly_catch.clear()


    def _record_catches(self, catches: list):
        """
        Record monthly catches from a list of tuples: (market_code, species_code, qty)
        """
        for m_code, s_code, qty in catches:
            if m_code in self.local_markets:
                self.local_markets[m_code].record_monthly_catch(s_code, qty)

    def _update_local_prices(self):
        """
        Update per-market prices using Elser adjustment based on current catches.
        """
        for m_code, market in self.local_markets.items():
            for s_code, qty in market.monthly_catch.items():
                last_price, last_qty = market.get_last_state(s_code)
                epsilon = self.species_dict[s_code].epsilon

                new_price = last_price * (last_qty / max(qty, 1e-6)) ** epsilon
                market.set_new_state(s_code, new_price, qty)

    def _update_local_history(self):
        """
        After updating prices, snapshot both prices and catches
        for each local market at the current tick,
        then clear monthly_catch for the next month.
        """
        current_tick = self.tick.get_tick()
        for market in self.local_markets.values():
            # Save both prices & catches for this tick
            market.price_history.append(
                (
                    current_tick,
                    {
                        "prices": market.local_prices.copy(),
                        "catches": market.monthly_catch.copy()
                    }
                )
            )

            #Reset catches after snapshotting
            market.monthly_catch.clear()

    def _compute_global_prices(self):
        """
        Compute global average prices and total catches across all local markets
        and snapshot into global history.
        """
        current_tick = self.tick.get_tick()

        global_prices = {}
        global_catches = {}

        for s_code in self.species_dict.keys():
            prices = []
            total_catch = 0.0

            for m in self.local_markets.values():
                if s_code in m.local_prices:
                    prices.append(m.local_prices[s_code])
                total_catch += m.price_history[-1][1]["catches"].get(s_code, 0.0) if m.price_history else 0.0

            if prices:
                global_prices[s_code] = sum(prices) / len(prices)
            if total_catch > 0:
                global_catches[s_code] = total_catch

        # Save current global prices
        self.global_market.global_prices = global_prices

        # Save both avg prices & total catches for this tick
        self.global_market.global_price_history.append(
            (
                current_tick,
                {
                    "avg_prices": global_prices.copy(),
                    "total_catches": global_catches.copy()
                }
            )
        )

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
                    species=species_pb2.Species(
                        species_code=species.code
                    ),
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
