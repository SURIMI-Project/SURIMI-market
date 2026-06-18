from collections import defaultdict
import threading
import logging
from itertools import product

class LocalMarket:
    """
    Represents a local fish market identified by a market code (UN/LOCODE).
    Computes local prices influenced by local supply, global prices (lambda_mix),
    and neighboring markets (theta_mix).

    Attributes:
        market_code (str): ISO 3166-1 alpha-2 + 3-letter port code.
        lambda_mix (float): Influence of global price on local price [0, 1].
        theta_mix (float): Influence of neighboring markets on local price [0, 1].
        Todo: Think about how to calibrate market influence through
            - lambda_mix: this will be especially relevant when dealing with the fleets from EwE, as they are not selling to specific local markets at the moment.
            - theta_mix: this might require some spatial representation of the market locations.
    """

    def __init__(self, market_code: str,
                 name: str = None, lon: float = None, lat: float = None,
                 lambda_mix=0.0, theta_mix=0.0):
        self.market_code = market_code
        self.market_name = name
        self.lon = lon
        self.lat = lat
        self.lambda_mix = lambda_mix
        self.theta_mix = theta_mix

        self._monthly_catch: dict[tuple[str, str], float] = defaultdict(float)
        self.local_prices: dict[tuple[str, str], float | None] = {}  # species_code -> price or None

        # Price memory for Elser adjustment
        self.price_state: dict[tuple[str, str], tuple[float, float]] = {}  # species_code -> (last_price, last_quantity)

        # History
        self.price_history = []  # [(tick, {"prices": {...}, "catches": {...}})]

        # Holds historical CSV prices: (species_code, date) -> price
        self.price_time_series = {}

        self.lock = threading.Lock()
        self.logger = logging.getLogger("LocalMarket")

    def add_monthly_catch(self, species_code, gear_code, quantity):
        with self.lock:
            self._monthly_catch[(species_code, gear_code)] += quantity

    def clear_monthly_catch(self):
        with self.lock:
            self._monthly_catch.clear()
    
    def get_monthly_catch_snapshot(self):
        with self.lock:
            return self._monthly_catch.copy()

    def get_last_state(self, s_code: str, g_code) -> tuple[float, float]:
        """
        If no price state yet, fallback to initial price if available, otherwise price=1.0, qty=1.
        """
        if (s_code, g_code) not in self.price_state:
            init_price = self.local_prices.get((s_code, g_code), 1.0)
            return init_price, 1.0
        return self.price_state[(s_code, g_code)]

    def set_new_state(self, s_code: str, g_code: str, price: float, qty: float):
        """Save updated price state for next tick."""
        self.price_state[(s_code, g_code)] = (price, qty)
        self.local_prices[(s_code, g_code)] = price

    # def calculate_market_influences_price(self, species_code: str = None, other_market_avg: dict[str, float] = None):
    #     """
    #     Lambda/theta mixing if needed (not used in currenct scenarios).
    #     Will only apply if global/neighbor prices are valid (not None).
    #     """
    #     prices = {}
    #     species_to_process = [species_code] if species_code else list(self.species_dict.keys())
    #     for code in species_to_process:
    #         p_local = self.local_prices.get(code)
    #         if p_local is None:
    #             continue

    #         if self.lambda_mix > 0:
    #             p_global = self.global_market.get_global_price(code)
    #             if p_global is not None and p_global > 0:
    #                 p_local = (1 - self.lambda_mix) * p_local + self.lambda_mix * p_global

    #         if other_market_avg and self.theta_mix > 0:
    #             p_other = other_market_avg.get(code, p_local)
    #             if p_other is not None and p_other > 0:
    #                 p_local = (1 - self.theta_mix) * p_local + self.theta_mix * p_other

    #         self.local_prices[code] = p_local
    #         prices[code] = p_local

    #     self.price_history.append((len(self.price_history), {"prices": self.local_prices.copy()}))
    #     return prices
    
    def set_initial_price_to_earliest_available(self, species: list[str], gear_codes: list[str]):
        """
        Fill local markets with earliest price if available.

        On initialising the local market, we receive a list of datapoints for
        this market. The purpose of this method is to 
        """
        self.logger.info("Initializing baseline prices from earliest price config...")

        # Species prices depend on the gear code, so we a use Cartesian product
        # to combine them.
        lookup_keys = list(product(species, gear_codes))

        # Phase 1: fill from CSV if available
        earliest_per_species = {}

        for (species_code, gear_code, date), price in self.price_time_series.items():
            # TODO interpolate between the two datapoints that the date falls between.
            if species_code not in earliest_per_species or date < earliest_per_species[species_code][0]:
                earliest_per_species[(species_code, gear_code)] = (date, price)

        for lookup_key in lookup_keys:
            if lookup_key in earliest_per_species:
                _, price = earliest_per_species[lookup_key]
                if price is not None and price > 0:
                    self.local_prices[lookup_key] = price
                else:
                    self.logger.warning(f"CSV price is invalid for {self.market_code}/{lookup_key[0]} with gear code {lookup_key[1]}, ignoring.")
