import sys
from pathlib import Path
import datetime
import logging
from itertools import product

from surimi.v1 import species_price_pb2, species_pb2, get_species_prices_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from .Species import Species
from .Tick import Tick
from .LocalMarket import LocalMarket
from .GlobalMarket import GlobalMarket

class MarketSystem:
    """
    Represents the overall simulation environment for dynamic fish markets.
    Combines multiple LocalMarkets and a GlobalMarket for multi-level pricing.

    TODO give the market system a better name?
    """
    def __init__(self, currency: str, start_time: datetime.datetime):
        self.tick = Tick(step_months=1, start_time=start_time)
        self.currency = currency
        self.species_dict: dict[str, Species] = {}
        self.local_markets: dict[str, LocalMarket] = {}
        self.global_market = GlobalMarket(self.species_dict)
        self.logger = logging.getLogger("MarketSystem")

    def add_species(self, species: Species):
        self.species_dict[species.code] = species

    def add_local_market(self, market: object):
        self.local_markets[market.market_code] = market

    def step(self, catches: list):
        """
        One monthly update step:
        1) Record catches
        2) Update local prices
        3) Save history (prices + catches) and clear monthly catches
        4) Compute global prices
        5) Advance simulation tick
        """
        self.logger.debug(f"MarketSystem STEP triggered — {len(catches)} catch entries")

        for m, s, q in catches:
            self.logger.debug(f"  Market={m}, Species={s}, Qty={q:.2f} kg")
        if not catches:
            self.logger.debug("No catches provided this tick (prices will remain unchanged).")

        self.compute_local_prices()
        self._update_local_history()
        self.tick.increase_tick()

        self.logger.debug("MarketSystem STEP completed")

    def compute_local_prices(self):
        """
        Update per-market prices using Elser adjustment based on current catches.
        """
        self.logger.info("Updating local prices using Elser adjustment...")
        for m_code, market in self.local_markets.items():
            for (s_code, g_code), qty in market.get_monthly_catch_snapshot().items():
                try:
                    last_price, last_qty = market.get_last_state(s_code, g_code)
                    epsilon = self.species_dict[s_code].epsilon
                    new_price = last_price * (last_qty / max(qty, 1e-6)) ** epsilon
                    market.set_new_state(s_code, g_code, new_price, qty)
                except Exception as e:
                    self.logger.warning(f"Error updating price for market={m_code}, species={s_code}: {e}")

    def _update_local_history(self):
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

    def compute_global_prices(self, gear_codes: list[str]):
        """
        Compute global average prices and total catches across all local markets.
        - Ignores markets with missing prices (None).
        - If no valid local prices exist, global price = None.
        """
        self.logger.info("Computing global average prices...")
        current_tick = self.tick.get_tick()
        global_prices = {}
        global_catches = {}

        valid_global_count = 0
        missing_global_count = 0

        # Species prices depend on the gear code, so we a use Cartesian product
        # to combine them.
        lookup_keys = list(product(self.species_dict.keys(), gear_codes))

        for key in lookup_keys:
            # key[n] = (species_code, gear_code)
            valid_prices = []
            total_catch = 0.0

            for m in self.local_markets.values():
                price = m.local_prices.get(key)
                if price is not None and price > 0:
                    valid_prices.append(price)

                if len(m.price_history) > 0:
                    # TODO investigate whether this actually works.
                    # TODO if the price is invalid, why do we still append to total catches?
                    total_catch += m.price_history[-1][1]["catches"].get(key, 0.0)

            if len(valid_prices) > 0:
                # TODO improve this, as it shouldnt just do a mean of the local market prices.
                # TODO take inflation as a parameter.
                global_prices[key] = sum(valid_prices) / len(valid_prices)
                valid_global_count += 1
            else:
                global_prices[key] = None
                missing_global_count += 1

            if total_catch > 0:
                global_catches[key] = total_catch

        self.global_market.global_prices = global_prices

        self.global_market.global_price_history.append(
            (current_tick, {
                "avg_prices": global_prices.copy(),
                "total_catches": global_catches.copy()
            })
        )

        self.logger.debug(
            f"Global prices (tick={current_tick}): valid={valid_global_count}, missing={missing_global_count}"
        )
        for s, p in global_prices.items():
            if p is not None:
                self.logger.debug(f"  {s}: {p:.2f} EUR/kg")
            else:
                self.logger.debug(f"  {s}: None (no valid local prices)")

    def set_local_market_initial_price(self, gear_codes: list[str]):
        """
        Set local markets with earliest price if available.

        On initialising a local market, we receive a list of datapoints for
        this market. The purpose of this method is to set each local market's
        initial species price to the earliest available.
        """
        self.logger.info("Initializing baseline prices from earliest price config...")

        # Phase 1: fill from CSV if available
        for market in self.local_markets.values():
            market.set_initial_price_to_earliest_available(self.species_dict.keys(), gear_codes)

        self.logger.info("Local markets baseline prices initialization complete.")


    def populate_missing_local_prices(self, gear_codes: list[str]):
        """
        Compute a global baseline and fill only truly missing prices.
        """
        # Compute initial GLOBAL baseline without recording history
        self.logger.info("Computing initial GLOBAL baseline from valid local markets...")

        # Phase 3: fill missing using valid global avg
        # TODO use actual global value instead of using global mean/avg. This
        #      code was written probably because our data was
        #      bad/lacking datapoints/lacking information
        for market in self.local_markets.values():
            for (s_code, g_code) in product(self.species_dict.keys(), gear_codes):
                if (s_code, g_code) not in market.local_prices:
                    global_fallback = self.global_market.get_global_price(s_code, g_code)
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
                    self.logger.debug(f"{species.code}@{market.market_code} has no valid price, skipped")

        total_species = len(self.species_dict)
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
                for (s_code, gear_code), avg_price in avg_prices.items():
                    total_catch = total_catches.get(s_code, 0.0)
                    avg_price_str = "" if avg_price is None else f"{avg_price:.4f}"
                    f.write(f"{tick_id},{s_code},{gear_code},{avg_price_str},{total_catch:.4f}\n")

        # 2) Export LOCAL market history
        local_path = output_folder / "local_prices.csv"
        with local_path.open("w", newline="") as f:
            f.write("tick,market,species,gear_code,price,catch\n")
            for m_code, market in self.local_markets.items():
                for tick_id, snapshot in market.price_history:
                    prices = snapshot["prices"]
                    catches = snapshot["catches"]
                    for (s_code, gear_code), price in prices.items():
                        catch_val = catches.get(s_code, 0.0)
                        price_str = "" if price is None else f"{price:.4f}"
                        f.write(f"{tick_id},{m_code},{s_code},{gear_code},{price_str},{catch_val:.4f}\n")

        self.logger.info(f"Export complete for simulation {simulation_id}")