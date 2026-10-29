import sys
from pathlib import Path
import datetime
import logging
from itertools import product

from surimi.v1 import species_price_pb2, species_pb2, get_species_prices_pb2
from google.protobuf.timestamp_pb2 import Timestamp
import pandas as pd
import grpc

from MarketError import MarketError

from .Tick import Tick
from .LocalMarket import LocalMarket
from .GlobalMarket import GlobalMarket
from . import consts


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
        self.category_codes_contract: list[str] = []
        self.logger = logging.getLogger(consts.MARKET_SYSTEM_LOGGER_NAME)

        # Collection of months that we have real data for. Used to compute
        # species price if real data is available.
        self.real_data_months = []
        self.pending_sales_summary = {}


    def step(self):
        """
        One monthly update step:
        1) Check if there are pending catches that should occur in this month
        2) Check if we have the real price. If so,
          2a) Compute local prices
        3) Save history (prices + catches) and clear monthly catches
        4) Advance simulation tick
        """
        # 1) If there are pending catches for this month, collect and use them
        for date in list(self.pending_sales_summary.keys()):
            if date == self.tick.get_datetime().strftime("%Y-%m-%d"):
                for sale in self.pending_sales_summary[date]:
                    self._accumulate_local_market_catches(sale)
                
                del self.pending_sales_summary[date]

        # 2) Use real data if we have it
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
                # If there exists real price, then use it. Otherwise, compute the price.
                new_price = market.price_time_series.get((s_code, g_code, str(self.tick.get_date())))
                if new_price is not None:
                    market.set_new_state(s_code, g_code, new_price, qty)
                else:
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
            category_codes=self.category_codes_contract
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
            market.set_initial_price_to_earliest_available(self.species_codes, self.category_codes_contract)

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

            # Prices depend on both species code and category code
            for (s_code, g_code) in product(self.species_codes, self.category_codes_contract):
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

        for (s_code, g_code) in product(self.species_codes, self.category_codes_contract):
            for market in self.local_markets.values():
                price = market.local_prices.get((s_code, g_code))

                if price is not None:
                    valid_count += 1
                    message = species_price_pb2.SpeciesPrice(
                        species=species_pb2.Species(species_code=s_code),
                        market_code=market.market_code,
                        price=price,
                        currency=self.currency,
                        category_code=g_code
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
            f.write("tick,species,category_code,avg_price,total_catch\n")
            for tick_id, data in self.global_market.global_price_history:
                avg_prices = data["avg_prices"]
                total_catches = data["total_catches"]
                for (s_code, g_code), avg_price in avg_prices.items():
                    total_catch = total_catches.get((s_code, g_code), 0.0)
                    avg_price_str = "" if avg_price is None else f"{avg_price:.4f}"
                    f.write(f"{tick_id+1},{s_code},{g_code},{avg_price_str},{total_catch:.4f}\n")

        # 2) Export LOCAL market history
        local_path = output_folder / "local_prices.csv"
        with local_path.open("w", newline="") as f:
            f.write("tick,market,species,category_code,price,catch\n")
            for m_code, market in self.local_markets.items():
                for tick_id, snapshot in market.price_history:
                    prices = snapshot["prices"]
                    catches = snapshot["catches"]
                    for (s_code, g_code), price in prices.items():
                        catch_val = catches.get((s_code, g_code), 0.0)
                        price_str = "" if price is None else f"{price:.4f}"
                        f.write(f"{tick_id+1},{m_code},{s_code},{g_code},{price_str},{catch_val:.4f}\n")

        self.logger.info(f"Export complete for simulation {simulation_id}")

    def update_sales(self, market_sales, start_date_time: int) -> MarketError:
        """
        Receives external sales (catches) from controller.

        Arguments
        ---------
            market_sales : Sales summary from the UpdateSales request.
    
        Returns
        -------
            MarketError : If an error occurred, then the MarketError is returned
        """
        self.logger.info("Sales update received...")

        # Validate input
        for summary in market_sales:
            if summary.market_code not in self.local_markets:
                return MarketError(
                    status_code=grpc.StatusCode.FAILED_PRECONDITION,
                    message="At least one local market does not exist."
                )

            for sale in summary.sales:
                if sale.quantity < 1e-323:
                    return MarketError(
                        status_code=grpc.StatusCode.INVALID_ARGUMENT,
                        message=f"Quantity for market {summary.market_code}," + \
                                f"species {sale.species.species_code} and" + \
                                f"category code {sale.category_code} is " + \
                                f"{sale.quantity}. It must be positive"
                    )
            
        # If the start_date_time occurs at a later stage, then save it. If it
        # occurs before the current tick, return an error about it.
        requested_time = datetime.datetime.fromtimestamp(start_date_time, tz=datetime.UTC)
        current_time = self.tick.get_datetime()
        
        if requested_time > current_time:
            # Post current time
            date: str = requested_time.strftime("%Y-%m-%d")
            if self.pending_sales_summary.get(date) is None:
                self.pending_sales_summary[date] = []
            self.pending_sales_summary[date].append(market_sales)
            return None
        elif requested_time < current_time:
            # Pre current time
            return MarketError(
                status_code=grpc.StatusCode.INVALID_ARGUMENT,
                message="'start_date_time cannot be earlier than the current time'" 
            )
        else:
            # Same time as current time
            self._accumulate_local_market_catches(market_sales)

        return None
    

    def _accumulate_local_market_catches(self, market_sales):
        new_entries = 0

        for summary in market_sales:
            market_code = summary.market_code
            local_market = self.local_markets.get(market_code)

            if local_market is None:
                assert local_market, f"Local market {local_market} does not exist. This should not happen."
                continue

            for sale in summary.sales:
                species_code = sale.species.species_code
                quantity = sale.quantity
                category_code = sale.category_code
                
                local_market.add_monthly_catch(species_code, category_code, quantity)
                new_entries += 1

                # Use only for debugging. This is a very expensive operation.
                # Optionally, comment it out.
                self.logger.debug(
                    f"Sale: market={market_code}, species={species_code}, category={category_code}, qty={quantity:.2f} kg"
                )

        self.logger.info(f"Processed {new_entries} sales.")

    def load_local_market_price_history(self, prices_df: pd.DataFrame):
        """
        Load time-series prices into each local market from CSV file.

        CSV columns: date, market_code, species_code, category_code, price, currency, measurement_unit
        """
        # TODO add check when price is not supported
        for code, found_markets in prices_df.groupby("market_code"):
            market = self.local_markets.get(code)
            if self.local_markets.get(code) is not None:
                for row in found_markets.itertuples(index=False):
                    date = row[0]
                    species_code = row[2]
                    category_code = row[3]
                    # NOTE: with the current prices.csv column order (date, market_code,
                    # species_code, category_code, landings_kg, price, currency,
                    # measurement_unit, size), row[4] is actually landings_kg, not price --
                    # this positional read is reading the wrong column. Documented here,
                    # not fixed, per a deliberate decision made while building the
                    # src/models/market/ plugin (out of scope for that work; left for a
                    # separate, explicit fix).
                    price = row[4]

                    market.price_time_series[(species_code, category_code, date)] = price