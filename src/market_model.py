import sys
from pathlib import Path
import datetime
from dateutil.relativedelta import *
from collections import defaultdict
from typing import Dict, Tuple

from surimi.v1 import species_price_pb2, market_service_pb2, species_pb2, get_species_prices_pb2
from google.protobuf.timestamp_pb2 import Timestamp
from logger import get_logger

# Define repo root & setup import/export paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "pb"))

logger = get_logger("market_model")


def _is_valid_price(price: float | None) -> bool:
    return price is not None and price > 0


class Tick:
    """A per-MarketSystem time tracker."""
    def __init__(self, start_time: datetime.datetime, step_months=1):
        self.tick = 0
        self.start_time = start_time
        self.step_months = step_months

    def increase_tick(self):
        self.tick += 1

    def get_tick(self):
        return self.tick

    def set_timestep(self, days: int):
        self.step_months = days

    def get_timestamp(self):
        # TODO unit test this
        current_time = (self.start_time +
                        relativedelta(months=self.tick * self.step_months)).isoformat() + "Z"
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
    as an average of valid local prices (preliminary solution).
    """
    def __init__(self, species_dict):
        self.species_dict = species_dict
        self.global_prices: Dict[str, float | None] = {}
        self.global_price_history = []  # [(tick, {"avg_prices": {...}, "total_catches": {...}})]

    def get_global_price(self, species_code: str) -> float | None:
        return self.global_prices.get(species_code)
    
    # TODO the global market computations should happen here.


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
        Todo: Think about how to calibrate market influence through
            - lambda_mix: this will be especially relevant when dealing with the fleets from EwE, as they are not selling to specific local markets at the moment.
            - theta_mix: this might require some spatial representation of the market locations.
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
        self.local_prices: Dict[str, float | None] = {}  # species_code -> price or None

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
        If no price state yet, fallback to initial price if available, otherwise price=1.0, qty=1.
        """
        if s_code not in self.price_state:
            init_price = self.local_prices.get(s_code, 1.0)
            return init_price, 1.0
        return self.price_state[s_code]

    def set_new_state(self, s_code: str, price: float, qty: float):
        """Save updated price state for next tick."""
        self.price_state[s_code] = (price, qty)
        self.local_prices[s_code] = price

    def calculate_market_influences_price(self, species_code: str = None, other_market_avg: Dict[str, float] = None):
        """
        Lambda/theta mixing if needed (not used in currenct scenarios).
        Will only apply if global/neighbor prices are valid (not None).
        """
        prices = {}
        species_to_process = [species_code] if species_code else list(self.species_dict.keys())
        for code in species_to_process:
            p_local = self.local_prices.get(code)
            if p_local is None:
                continue

            if self.lambda_mix > 0:
                p_global = self.global_market.get_global_price(code)
                if _is_valid_price(p_global):
                    p_local = (1 - self.lambda_mix) * p_local + self.lambda_mix * p_global

            if other_market_avg and self.theta_mix > 0:
                p_other = other_market_avg.get(code, p_local)
                if _is_valid_price(p_other):
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
    def __init__(self, currency: str, start_time: datetime.datetime):
        self.tick = Tick(step_months=1, start_time=start_time)
        self.currency = currency
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
        logger.debug(f"MarketSystem STEP triggered — {len(catches)} catch entries")
        for m, s, q in catches:
            logger.debug(f"  Market={m}, Species={s}, Qty={q:.2f} kg")
        if not catches:
            logger.debug("No catches provided this tick (prices will remain unchanged).")

        self._record_catches(catches)
        # TODO this is the method to compute local prices. Should be renamed to
        #      self.compute_local_prices()
        self._update_local_prices()
        self._update_local_history()
        self.tick.increase_tick()

        logger.debug("MarketSystem STEP completed")

    def _record_catches(self, catches: list):
        """Record monthly catches from a list of tuples: (market_code, species_code, qty)"""
        logger.info("Recording monthly catches...")
        for m_code, s_code, qty in catches:
            if m_code in self.local_markets:
                try:
                    self.local_markets[m_code].record_monthly_catch(s_code, qty)
                except Exception as e:
                    logger.warning(f"Error recording catch for market={m_code}, species={s_code}: {e}")

    def _update_local_prices(self):
        """
        Update per-market prices using Elser adjustment based on current catches.
        
        TODO it seems like this does not do anything, because we only get static
             prices (issue 34: https://github.com/Official-EwE/SURIMI-market/issues/34)
        """
        logger.info("Updating local prices using Elser adjustment...")
        for m_code, market in self.local_markets.items():
            for s_code, qty in market.monthly_catch.items():
                try:
                    last_price, last_qty = market.get_last_state(s_code)
                    epsilon = self.species_dict[s_code].epsilon
                    new_price = last_price * (last_qty / max(qty, 1e-6)) ** epsilon
                    market.set_new_state(s_code, new_price, qty)
                except Exception as e:
                    logger.warning(f"Error updating price for market={m_code}, species={s_code}: {e}")

    def _update_local_history(self):
        """
        After updating prices, snapshot both prices and catches
        for each local market at the current tick,
        then clear monthly_catch for the next month.
        """
        logger.info("Snapshotting local history and clearing monthly catches...")
        current_tick = self.tick.get_tick()
        for market in self.local_markets.values():
            market.price_history.append(
                (
                    current_tick,
                    {
                        "prices": market.local_prices.copy(),
                        "catches": market.monthly_catch.copy()
                    }
                )
            )
            market.monthly_catch.clear()

    def _compute_global_prices(self):
        """
        Compute global average prices and total catches across all local markets.
        - Ignores markets with missing prices (None).
        - If no valid local prices exist, global price = None.
        """
        logger.info("Computing global average prices...")
        current_tick = self.tick.get_tick()
        global_prices = {}
        global_catches = {}

        valid_global_count = 0
        missing_global_count = 0

        for s_code in self.species_dict.keys():
            valid_prices = []
            total_catch = 0.0

            for m in self.local_markets.values():
                price = m.local_prices.get(s_code)
                if _is_valid_price(price):
                    valid_prices.append(price)

                if m.price_history:
                    total_catch += m.price_history[-1][1]["catches"].get(s_code, 0.0)

            if valid_prices:
                # TODO improve this, as it shouldnt just do a mean of the local market prices.
                # TODO take inflation as a parameter.
                global_prices[s_code] = sum(valid_prices) / len(valid_prices)
                valid_global_count += 1
            else:
                global_prices[s_code] = None
                missing_global_count += 1

            if total_catch > 0:
                global_catches[s_code] = total_catch

        self.global_market.global_prices = global_prices

        self.global_market.global_price_history.append(
            (current_tick, {
                "avg_prices": global_prices.copy(),
                "total_catches": global_catches.copy()
            })
        )

        logger.debug(
            f"Global prices (tick={current_tick}): valid={valid_global_count}, missing={missing_global_count}"
        )
        for s, p in global_prices.items():
            if p is not None:
                logger.debug(f"  {s}: {p:.2f} EUR/kg")
            else:
                logger.debug(f"  {s}: None (no valid local prices)")

    def _initialize_baseline_prices_from_csv(self):
        """
        TODO fix the naming
        Fill local markets with earliest CSV price if available,
        then compute a global baseline and fill only truly missing prices.
        """
        logger.info("Initializing baseline prices from earliest CSV config...")

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
                    if _is_valid_price(price):
                        market.local_prices[s_code] = price
                    else:
                        logger.warning(f"CSV price is invalid for {market.market_code}/{s_code}, ignoring.")

        # Compute initial GLOBAL baseline without recording history
        logger.info("Computing initial GLOBAL baseline from valid local markets...")

        # Phase 3: fill missing using valid global avg
        # TODO use actual global value instead of using global mean/avg. This
        #      code was written probably because our data was
        #      bad/lacking datapoints/lacking information
        for market in self.local_markets.values():
            for s_code in self.species_dict.keys():
                if s_code not in market.local_prices:
                    global_fallback = self.global_market.get_global_price(s_code)
                    if _is_valid_price(global_fallback):
                        market.local_prices[s_code] = global_fallback
                    else:
                        logger.warning(
                            f"No CSV/global price for {market.market_code}/{s_code}, leaving as missing (None)."
                        )

        logger.info("Baseline initialization complete.")

    def get_species_prices_message(self):
        """
        Generates an update message containing pricing details for all species at all ports.
        Only includes markets/species with valid prices (skips None).
        """
        messages = []
        timestamp = Timestamp()
        timestamp.FromJsonString(self.tick.get_timestamp())

        valid_count = 0
        missing_count = 0

        for species in self.species_dict.values():
            for market in self.local_markets.values():
                price = market.local_prices.get(species.code)

                if price is not None:
                    valid_count += 1
                    message = species_price_pb2.SpeciesPrice(
                        species=species_pb2.Species(species_code=species.code),
                        market_code=market.market_code,
                        price=price,
                        currency=self.currency
                    )
                    messages.append(message)
                else:
                    missing_count += 1
                    logger.debug(f"{species.code}@{market.market_code} has no valid price, skipped")

        total_species = len(self.species_dict)
        total_markets = len(self.local_markets)
        logger.info(
            f"Price summary — species: {total_species}, markets: {total_markets}, "
            f"combinations: {total_species * total_markets}, valid: {valid_count}, missing: {missing_count}"
        )

        return get_species_prices_pb2.GetSpeciesPricesResponse(
            species_price_summary=species_price_pb2.SpeciesPriceSummary(species_prices=messages)
        )

    def export_results(self, simulation_id: str, folder_path: str):
        """
        Export final simulation results to:
        <BASE_DIR>/<folder_path>/<simulation_id>/global_prices.csv
        """
        output_folder = BASE_DIR / folder_path / simulation_id
        output_folder.mkdir(parents=True, exist_ok=True)

        # 1) Export GLOBAL prices
        global_path = output_folder / "global_prices.csv"
        with global_path.open("w", newline="") as f:
            f.write("tick,species,avg_price,total_catch\n")
            for tick_id, data in self.global_market.global_price_history:
                avg_prices = data["avg_prices"]
                total_catches = data["total_catches"]
                for s_code, avg_price in avg_prices.items():
                    total_catch = total_catches.get(s_code, 0.0)
                    avg_price_str = "" if avg_price is None else f"{avg_price:.4f}"
                    f.write(f"{tick_id},{s_code},{avg_price_str},{total_catch:.4f}\n")

        # 2) Export LOCAL market history
        local_path = output_folder / "local_prices.csv"
        with local_path.open("w", newline="") as f:
            f.write("tick,market,species,price,catch\n")
            for m_code, market in self.local_markets.items():
                for tick_id, snapshot in market.price_history:
                    prices = snapshot["prices"]
                    catches = snapshot["catches"]
                    for s_code, price in prices.items():
                        catch_val = catches.get(s_code, 0.0)
                        price_str = "" if price is None else f"{price:.4f}"
                        f.write(f"{tick_id},{m_code},{s_code},{price_str},{catch_val:.4f}\n")

        logger.info(f"Export complete for simulation {simulation_id}")