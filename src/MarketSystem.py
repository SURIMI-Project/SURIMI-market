import sys
from pathlib import Path
import datetime
import logging
from itertools import product

from surimi.v1 import species_price_pb2, species_pb2, get_species_prices_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from .Tick import Tick
from .LocalMarket import LocalMarket
from .GlobalMarket import GlobalMarket
from util import consts

class MarketSystem:
    """
    Represents the overall simulation environment for dynamic fish markets.
    Combines multiple LocalMarkets and a GlobalMarket for multi-level pricing.

    TODO give the market system a better name?
    """
    def __init__(self, currency: str, start_time: datetime.datetime):
        self.tick = Tick(step_months=1, start_time=start_time)
        self.currency = currency
        self.species_codes: list[str] = {}
        self.local_markets: dict[str, LocalMarket] = {}
        self.global_market = GlobalMarket()
        self.gear_codes_contract: list[str] = []
        self.logger = logging.getLogger(consts.MARKET_SYSTEM_LOGGER_NAME)

    def add_local_market(self, market: object):
        self.local_markets[market.market_code] = market

    def step(self):
        """
        One monthly update step:
        1) Compute local prices
        2) Save history (prices + catches) and clear monthly catches
        3) Advance simulation tick
        """
        self.compute_local_prices()
        self.update_local_history()
        self.tick.increase_tick()

        self.logger.debug("MarketSystem STEP completed")

    def compute_local_prices(self):
        """
        Update per-market prices using Elser adjustment based on current catches.
        """
        self.logger.info("Updating local prices using Elser adjustment...")
        epsilon = 0.5
        for m_code, market in self.local_markets.items():
            for (s_code, g_code), qty in market.get_monthly_catch_snapshot().items():
                try:
                    last_price, last_qty = market.get_last_state(s_code, g_code)
                    new_price = last_price * (last_qty / max(qty, 1e-6)) ** epsilon
                    market.set_new_state(s_code, g_code, new_price, qty)
                except Exception as e:
                    self.logger.warning(f"Error updating price for market={m_code}, species={s_code}: {e}")

    def update_local_history(self):
        """
        After updating prices, snapshot both prices and catches
        for each local market at the current tick,
        then clear monthly_catch for the next month.
        """
        self.logger.info("Snapshotting local history and clearing monthly catches...")
        current_tick = self.tick.get_tick()
        for market in self.local_markets.values():
            market.price_history.append(
                (
                    current_tick,
                    {
                        "prices": market.local_prices.copy(),
                        "catches": market.get_monthly_catch_snapshot()
                    }
                )
            )
            market.clear_monthly_catch()

    def update_global_price_history(self):
        """
        Compute global average prices and total catches across all local markets.
        - Ignores markets with missing prices (None).
        - If no valid local prices exist, global price = None.
        """
        self.logger.info("Computing global average prices...")

        prices, catches = self.global_market.compute_prices(
            local_markets=self.local_markets,
            species_codes=self.species_codes,
            gear_codes=self.gear_codes_contract
        )

        current_tick = self.tick.get_tick()
        self.global_market.global_price_history.append(
            (current_tick, {
                "avg_prices": prices,
                "total_catches": catches
            })
        )

    def set_local_market_initial_price(self):
        """
        Set local markets with earliest price if available.

        On initialising a local market, we receive a list of datapoints for
        this market. The purpose of this method is to set each local market's
        initial species price to the earliest available.
        """
        self.logger.info("Initializing baseline prices from earliest price config...")

        # Phase 1: fill from CSV if available
        for market in self.local_markets.values():
            market.set_initial_price_to_earliest_available(self.species_codes, self.gear_codes_contract)

        self.logger.info("Local markets baseline prices initialization complete.")


    def set_missing_local_prices_with_global_fallback(self):
        """
        If there is no local price for a species, use the global price as
        a fallback.

        The most recent global price is used as the fallback price.
        """
        # Phase 3: fill missing using valid global avg
        # TODO use actual global value instead of using global mean/avg. This
        #      code was written probably because our data was
        #      bad/lacking datapoints/lacking information
        for market in self.local_markets.values():

            # Prices depend on both species code and gear code
            for (s_code, g_code) in product(self.species_codes, self.gear_codes_contract):
                if (s_code, g_code) not in market.local_prices:
                    global_fallback = self.global_market.get_most_recent_price(s_code, g_code)
                    if global_fallback is not None and global_fallback > 0:
                        market.local_prices[(s_code, g_code)] = global_fallback
                    else:
                        self.logger.debug(
                            f"No CSV/global price for {market.market_code}/{s_code}, leaving as missing (None)."
                        )

        self.logger.info("Missing local prices are populated with global prices.")


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

        for (s_code, g_code) in product(self.species_codes, self.gear_codes_contract):
            for market in self.local_markets.values():
                price = market.local_prices.get((s_code, g_code))

                if price is not None:
                    valid_count += 1
                    message = species_price_pb2.SpeciesPrice(
                        species=species_pb2.Species(species_code=s_code),
                        market_code=market.market_code,
                        price=price,
                        currency=self.currency
                    )
                    messages.append(message)
                else:
                    missing_count += 1
                    self.logger.debug(f"{s_code}@{market.market_code} has no valid price, skipped")

        total_species = len(self.species_codes)
        total_markets = len(self.local_markets)
        self.logger.info(
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
        # Define repo root & setup import/export paths
        base_dir = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(base_dir / "pb"))

        output_folder = base_dir / folder_path / simulation_id
        output_folder.mkdir(parents=True, exist_ok=True)

        # 1) Export GLOBAL prices
        global_path = output_folder / "global_prices.csv"
        with global_path.open("w", newline="") as f:
            f.write("tick,species,gear_code,avg_price,total_catch\n")
            for tick_id, data in self.global_market.global_price_history:
                avg_prices = data["avg_prices"]
                total_catches = data["total_catches"]
                for (s_code, g_code), avg_price in avg_prices.items():
                    total_catch = total_catches.get((s_code, g_code), 0.0)
                    avg_price_str = "" if avg_price is None else f"{avg_price:.4f}"
                    f.write(f"{tick_id},{s_code},{g_code},{avg_price_str},{total_catch:.4f}\n")

        # 2) Export LOCAL market history
        local_path = output_folder / "local_prices.csv"
        with local_path.open("w", newline="") as f:
            f.write("tick,market,species,gear_code,price,catch\n")
            for m_code, market in self.local_markets.items():
                for tick_id, snapshot in market.price_history:
                    prices = snapshot["prices"]
                    catches = snapshot["catches"]
                    for (s_code, g_code), price in prices.items():
                        catch_val = catches.get((s_code, g_code), 0.0)
                        price_str = "" if price is None else f"{price:.4f}"
                        f.write(f"{tick_id},{m_code},{s_code},{g_code},{price_str},{catch_val:.4f}\n")

        self.logger.info(f"Export complete for simulation {simulation_id}")