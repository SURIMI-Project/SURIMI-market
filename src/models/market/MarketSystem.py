"""
Orchestrator (spec section 3): owns the Tick, the CombinedMarkets, and the
dict of LocalMarkets; derives Local<->Combined topology (section 4), drives
cross-port init-time fitting (Task 12), the per-tick flow (Task 13), and
history/export (Task 14).
"""
import datetime
import logging

import grpc
import pandas as pd

from MarketError import MarketError

from . import consts
from .Tick import Tick
from .LocalMarket import LocalMarket
from .CombinedMarket import CombinedMarket
from .distance import build_distance_lookup
from .level_volume_model import build_complete_monthly_series
from .stacked_model import compute_high_tier_ports, distance_weighted_neighbor_signal, build_static_anchor_series


def build_prefix_to_locals(markets_df: pd.DataFrame) -> dict[str, list[str]]:
    """Union of explicit 2-letter combined codes and prefixes implied by the
    5-letter local codes present (spec section 4) -- identical logic to
    Plan 1's data_prep/run.py equivalent."""
    codes = markets_df["market_code"].dropna().unique().tolist()
    local_codes = [c for c in codes if len(c) == 5]
    explicit_prefixes = {c for c in codes if len(c) == 2}
    implied_prefixes = {c[:2] for c in local_codes}
    prefixes = explicit_prefixes | implied_prefixes

    return {prefix: [c for c in local_codes if c.startswith(prefix)] for prefix in prefixes}


class MarketSystem:
    def __init__(self, currency: str, start_time: datetime.datetime):
        self.tick = Tick(step_months=1, start_time=start_time)
        self.currency = currency
        self.species_codes: list[str] = []
        self.category_codes_contract: list[str] = []

        self.local_markets: dict[str, LocalMarket] = {}
        self.combined_markets: dict[str, CombinedMarket] = {}
        self.prefix_to_locals: dict[str, list[str]] = {}
        self.distance_lookup: dict[tuple[str, str], float] = {}

        # (species_code, category_code) -> set of High-tier port codes, frozen
        # at fit_all time (Task 12) and reused every tick (Task 13) to build
        # each port's live neighbor signal without recomputing volume shares.
        self.high_tier_ports_by_key: dict[tuple[str, str], set[str]] = {}

        # (market_code, (species_code, category_code)) -> the exact
        # port_monthly/neighbor_signal/anchor_series Step 2 was fit with
        # (Task 12). Retained so Task 21's open-items report can refit with a
        # DIFFERENT anchor scheme against the identical neighbor signal,
        # rather than approximating it.
        self.step2_fit_inputs: dict[tuple[str, tuple[str, str]], dict] = {}

        self.logger = logging.getLogger(consts.MARKET_SYSTEM_LOGGER_NAME)
        self.pending_sales_summary = {}

    def setup_topology(self, markets_df: pd.DataFrame) -> MarketError | None:
        self.prefix_to_locals = build_prefix_to_locals(markets_df)
        self.distance_lookup = build_distance_lookup(markets_df)

        local_rows = markets_df[markets_df["market_code"].str.len() == 5]
        missing_coords = local_rows[local_rows["lon"].isna() | local_rows["lat"].isna()]
        if len(missing_coords) > 0:
            return MarketError(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Local market(s) missing lon/lat -- this is a reference-data bug "
                f"(spec section 9.3), not an expected gap: {missing_coords['market_code'].tolist()}",
            )

        for row in local_rows.itertuples(index=False):
            self.local_markets[row.market_code] = LocalMarket(row.market_code, row.market_name, row.lon, row.lat)

        for prefix in self.prefix_to_locals:
            self.combined_markets[prefix] = CombinedMarket(prefix)

        return None

    def update_sales(self, market_sales, start_date_time: int) -> MarketError | None:
        for summary in market_sales:
            market_code = summary.market_code
            prefix = market_code[:2]
            is_exact_local = market_code in self.local_markets
            is_known_prefix = prefix in self.combined_markets

            if not is_exact_local and not is_known_prefix:
                return MarketError(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"Market code {market_code} does not match any known local market "
                    "or combined-market prefix.",
                )

            for sale in summary.sales:
                if sale.quantity < 1e-323:
                    return MarketError(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Quantity for market {market_code}, species {sale.species.species_code} "
                        f"and category code {sale.category_code} is {sale.quantity}. "
                        "It must be positive",
                    )

        current_time = self.tick.get_datetime()
        # start_date_time is always a real UTC epoch (gRPC convention), so it
        # must always be resolved in UTC first -- a bare fromtimestamp() would
        # silently reinterpret it using the HOST machine's local timezone,
        # not UTC. self.tick's start_time may itself be naive (this plan's
        # own setup_topology tests, Task 10; also Tick.reset()'s utcnow(),
        # which is naive-but-semantically-UTC) or UTC-aware (Elsa
        # convention), so tzinfo is stripped afterwards only to match
        # current_time's own awareness, never to change the resolved instant.
        requested_time = datetime.datetime.fromtimestamp(start_date_time, tz=datetime.UTC)
        if current_time.tzinfo is None:
            requested_time = requested_time.replace(tzinfo=None)

        if requested_time > current_time:
            date = requested_time.strftime("%Y-%m-%d")
            self.pending_sales_summary.setdefault(date, []).append(market_sales)
            return None
        elif requested_time < current_time:
            return MarketError(
                grpc.StatusCode.INVALID_ARGUMENT,
                "'start_date_time cannot be earlier than the current time'",
            )

        self._accumulate_market_catches(market_sales)
        return None

    def _accumulate_market_catches(self, market_sales):
        for summary in market_sales:
            market_code = summary.market_code
            local_market = self.local_markets.get(market_code)

            for sale in summary.sales:
                species_code = sale.species.species_code
                category_code = sale.category_code
                quantity = sale.quantity

                if local_market is not None:
                    local_market.add_monthly_catch(species_code, category_code, quantity)
                    continue

                prefix = market_code[:2]
                combined_market = self.combined_markets[prefix]
                combined_market.add_direct_catch(species_code, category_code, quantity)
                self.logger.warning(
                    f"Sale routed via catch-all: market_code={market_code} has no exact "
                    f"local match, folded into CombinedMarket {prefix}'s direct accumulator."
                )

    def fit_all(self, prices_df: pd.DataFrame, latest_date: pd.Timestamp):
        blended = prices_df[prices_df["size"].isna()]
        if self.species_codes:
            blended = blended[blended["species_code"].isin(self.species_codes)]
        if self.category_codes_contract:
            blended = blended[blended["category_code"].isin(self.category_codes_contract)]

        for prefix, combined_market in self.combined_markets.items():
            rows = blended[blended["market_code"] == prefix]
            monthly_by_key = self._build_monthly_by_key(rows)
            combined_market.fit(monthly_by_key, latest_date)
            combined_market.load_price_time_series(prices_df)

        for market_code, local_market in self.local_markets.items():
            rows = blended[blended["market_code"] == market_code]
            monthly_by_key = self._build_monthly_by_key(rows)
            local_market.fit_step1(monthly_by_key, latest_date)
            local_market.load_price_time_series(prices_df)

        self._fit_step2_for_all_ports(blended, latest_date)

    def _build_monthly_by_key(self, rows: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
        result = {}
        for (species_code, category_code), group in rows.groupby(["species_code", "category_code"]):
            subset = group[["date", "price", "landings_kg"]]
            try:
                result[(species_code, category_code)] = build_complete_monthly_series(subset)
            except ValueError:
                continue
        return result

    def _fit_step2_for_all_ports(self, blended: pd.DataFrame, latest_date: pd.Timestamp):
        all_keys = set()
        for local_market in self.local_markets.values():
            all_keys.update(local_market.step1_models.keys())

        for key in all_keys:
            species_code, category_code = key

            volume_rows = blended[
                (blended["species_code"] == species_code) & (blended["category_code"] == category_code)
            ]
            volumes_by_port = volume_rows.groupby("market_code")["landings_kg"].sum().to_dict()
            high_tier_ports = compute_high_tier_ports(volumes_by_port)
            self.high_tier_ports_by_key[key] = high_tier_ports

            predicted_series_by_port = {}
            for code, local_market in self.local_markets.items():
                series = local_market.get_step1_historical_series(species_code, category_code)
                if series is not None:
                    predicted_series_by_port[code] = series

            for market_code, local_market in self.local_markets.items():
                if key not in local_market.step1_models:
                    continue

                prefix = market_code[:2]
                combined_market = self.combined_markets.get(prefix)
                baseline_table = (
                    combined_market.get_baseline_price_table(species_code, category_code)
                    if combined_market is not None
                    else None
                )
                if baseline_table is None:
                    continue

                rows = blended[blended["market_code"] == market_code]
                port_monthly = self._build_monthly_by_key(rows).get(key)
                if port_monthly is None:
                    continue

                neighbor_signal = distance_weighted_neighbor_signal(
                    market_code, high_tier_ports, predicted_series_by_port, self.distance_lookup
                )
                if neighbor_signal is None:
                    self.logger.warning(
                        f"LocalMarket {market_code}: no High-tier neighbor available for {key}, "
                        "falling back to Step-1-only (spec section 9 point 3)"
                    )
                    continue

                anchor_series = build_static_anchor_series(port_monthly["date"], baseline_table)
                local_market.fit_step2(
                    species_code, category_code, port_monthly, anchor_series, neighbor_signal, latest_date
                )
                self.step2_fit_inputs[(market_code, key)] = {
                    "port_monthly": port_monthly,
                    "neighbor_signal": neighbor_signal,
                }
