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
    """A per-MarketSystem time tracker."""
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
        self.global_price_history = []  # [(tick, {"avg_prices": {...}, "total_catches": {...}})]

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
    # Todo: Think about what todo with market influence (lambda_mix, theta_mix)
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

        # Price memory for Elser adjustment
        self.price_state: Dict[str, Tuple[float, float]] = {}  # species_code -> (last_price, last_quantity)

        # History
        self.price_history = []  # [(tick, {"prices": {...}, "catches": {...}})]

        # Holds historical CSV prices: (species_code, date) -> price
        self.price_time_series = {}

    def record_monthly_catch(self, species_code: str, quantity: float):
        self.monthly_catch[species_code] += quantity

    def get_last_state(self, s_code: str) -> Tuple[float, float]:
        """
        If no price state yet, fallback to initial price from CSV/global avg and qty=1.
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
        """Lambda/theta mixing if needed (not used in baseline init)."""
        prices = {}
        species_to_process = [species_code] if species_code else list(self.species_dict.keys())
        for code in species_to_process:
            p_local = self.local_prices.get(code, 0.0)
            if self.lambda_mix > 0:
                p_global = self.global_market.get_global_price(code)
                p_local = (1 - self.lambda_mix) * p_local + self.lambda_mix * p_global
            if other_market_avg and self.theta_mix > 0:
                p_other = other_market_avg.get(code, p_local)
                p_local = (1 - self.theta_mix) * p_local + self.theta_mix * p_other

            self.local_prices[code] = p_local
            prices[code] = p_local

        self.price_history.append((len(self.price_history), {"prices": self.local_prices.copy()}))
        return prices


class MarketSystem:
    """
    Represents the overall simulation environment for dynamic fish markets.
    Combines multiple LocalMarkets and a GlobalMarket for multi-level pricing.
    """
    def __init__(self, currency: str, measurement_unit: str):
        self.tick = Tick(step_days=30)
        self.currency = currency
        self.measurement_unit = measurement_unit
        self.species_dict: Dict[str, Species] = {}
        self.local_markets: Dict[str, LocalMarket] = {}
        self.global_market = GlobalMarket(self.species_dict)

    def add_species(self, species: Species):
        self.species_dict[species.code] = species

    def add_local_market(self, market: object):
        self.local_markets[market.market_code] = market

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
                if m.price_history:
                    total_catch += m.price_history[-1][1]["catches"].get(s_code, 0.0)

            if prices:
                global_prices[s_code] = sum(prices) / len(prices)
            if total_catch > 0:
                global_catches[s_code] = total_catch

        # Save current global prices
        self.global_market.global_prices = global_prices

        # Save both avg prices & total catches for this tick
        self.global_market.global_price_history.append(
            (current_tick, {
                "avg_prices": global_prices.copy(),
                "total_catches": global_catches.copy()
            })
        )

    def _initialize_baseline_prices_from_csv(self):
        """
        Fill local markets with earliest CSV price if available,
        then compute a global baseline and fill missing markets/species.
        """
        print("🔄 Initializing baseline prices from earliest CSV config...")

        # Phase 1: fill from CSV if available
        for market in self.local_markets.values():
            earliest_per_species = {}
            if hasattr(market, "price_time_series") and market.price_time_series:
                for (species_code, date), price in market.price_time_series.items():
                    if species_code not in earliest_per_species or date < earliest_per_species[species_code][0]:
                        earliest_per_species[species_code] = (date, price)

            for s_code in self.species_dict.keys():
                if s_code in earliest_per_species:
                    _, price = earliest_per_species[s_code]
                    market.local_prices[s_code] = price
                    # print(f"✅ Baseline CSV for {market.market_code}/{s_code}: {price} EUR/kg")

        # Phase 2: compute global baseline avg
        print("📊 Computing initial GLOBAL baseline from filled local markets...")
        self._compute_global_prices()

        # Phase 3: fill missing using global avg
        for market in self.local_markets.values():
            for s_code in self.species_dict.keys():
                if s_code not in market.local_prices or market.local_prices[s_code] == 0:
                    global_fallback = self.global_market.get_global_price(s_code)
                    if global_fallback > 0:
                        market.local_prices[s_code] = global_fallback
                        #print(f"➡️ Fallback {market.market_code}/{s_code} using GLOBAL avg {global_fallback:.2f} EUR/kg")
                    else:
                        market.local_prices[s_code] = 1.0
                        print(f"⚠️ No CSV/global price for {market.market_code}/{s_code}, default 1.0 EUR/kg")

        print("✅ Baseline initialization complete!")

    def get_species_prices_message(self):
        """
        Generates an update message containing pricing details for all species at all ports.
        """
        messages = []
        timestamp = Timestamp()
        timestamp.FromJsonString(self.tick.get_timestamp())

        for species in self.species_dict.values():
            for market in self.local_markets.values():
                price = market.local_prices.get(species.code, 1.0)  # safe fallback
                message = species_price_pb2.SpeciesPrice(
                    species=species_pb2.Species(species_code=species.code),
                    market_code=market.market_code,
                    price=price,
                    measurement_unit=self.measurement_unit,
                    currency=self.currency,
                    timestamp=timestamp
                )
                messages.append(message)
                print(f"[Market] {species.code}@{market.market_code} -> {price:.2f} {self.currency}/{self.measurement_unit}")

        return market_pb2.GetSpeciesPricesResponse(prices=messages)
