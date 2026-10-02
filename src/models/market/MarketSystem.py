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
