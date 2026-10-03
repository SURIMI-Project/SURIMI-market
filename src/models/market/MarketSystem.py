"""
Orchestrator (spec section 3): owns the Tick, the CombinedMarkets, and the
dict of LocalMarkets; derives Local<->Combined topology (section 4), drives
cross-port init-time fitting (Task 12), the per-tick flow (Task 13), and
history/export (Task 14).
"""
import datetime
import logging
from pathlib import Path

import grpc
import pandas as pd

from surimi.v1 import species_price_pb2, species_pb2, get_species_prices_pb2

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

        # Built once per port (O(ports*keys)), not once per (port, key)
        # (O(ports*keys^2)) -- the previous per-key rebuild was the dominant
        # term in init_simulation's super-linear growth with contract species
        # count, confirmed by the final whole-branch review (measured 23.7s /
        # 56.5s / 134.2s at 5 / 10 / 20 species).
        monthly_by_key_by_port = {
            market_code: self._build_monthly_by_key(blended[blended["market_code"] == market_code])
            for market_code in self.local_markets
        }

        for key in all_keys:
            species_code, category_code = key

            volume_rows = blended[
                (blended["species_code"] == species_code)
                & (blended["category_code"] == category_code)
                & (blended["market_code"].isin(self.local_markets))
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

                port_monthly = monthly_by_key_by_port[market_code].get(key)
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

    def step(self):
        current_date = pd.Timestamp(self.tick.get_date())
        date_str = self.tick.get_datetime().strftime("%Y-%m-%d")

        if date_str in self.pending_sales_summary:
            for sale in self.pending_sales_summary.pop(date_str):
                self._accumulate_market_catches(sale)

        local_volumes = {code: lm.get_monthly_catch_snapshot() for code, lm in self.local_markets.items()}
        combined_direct_volumes = {
            code: cm.get_direct_catch_snapshot() for code, cm in self.combined_markets.items()
        }

        # Step 2 of spec section 5: CombinedMarket prices + total tick volumes.
        combined_prices = {}
        combined_total_volumes = {}
        for prefix, combined_market in self.combined_markets.items():
            children_codes = self.prefix_to_locals.get(prefix, [])
            for key in combined_market.models.keys():
                total_volume = combined_direct_volumes[prefix].get(key, 0.0)
                total_volume += sum(local_volumes[c].get(key, 0.0) for c in children_codes)
                combined_prices[(prefix, key)] = combined_market.get_price(
                    key[0], key[1], current_date, total_volume
                )
                combined_total_volumes[(prefix, key)] = total_volume

        # Step 3: Step 1 for every LocalMarket, every key, unconditionally
        # (plan header decision 6 -- needed as neighbor fuel regardless of
        # whether this port itself has real data this tick).
        step1_predictions = {}
        for code, local_market in self.local_markets.items():
            for key in local_market.step1_models.keys():
                volume = local_volumes[code].get(key, 0.0)
                step1_predictions[(code, key)] = local_market.predict_step1(key[0], key[1], current_date, volume)

        # Step 4: final reported price per LocalMarket (real-price-first;
        # Step 2 only computed/used when this port itself lacks real data).
        local_prices = {code: {} for code in self.local_markets}
        for code, local_market in self.local_markets.items():
            prefix = code[:2]
            combined_market = self.combined_markets.get(prefix)

            for key in local_market.step1_models.keys():
                real_price = local_market.get_real_price(key[0], key[1], current_date)
                if pd.notna(real_price):
                    local_prices[code][key] = real_price
                    continue

                final_price = step1_predictions.get((code, key))

                if key in local_market.step2_stacked_models and combined_market is not None:
                    baseline_table = combined_market.get_baseline_price_table(key[0], key[1])
                    if baseline_table is not None and current_date.month in baseline_table.index:
                        anchor_value = baseline_table[current_date.month]
                        high_tier_ports = self.high_tier_ports_by_key.get(key, set())
                        neighbor_candidates = {
                            other_code: predicted
                            for (other_code, other_key), predicted in step1_predictions.items()
                            if other_key == key and other_code != code and predicted is not None
                        }
                        neighbor_value = self._weighted_neighbor_value(code, high_tier_ports, neighbor_candidates)
                        if neighbor_value is not None:
                            step2_price = local_market.predict_step2(
                                key[0], key[1], current_date, local_volumes[code].get(key, 0.0),
                                anchor_value, neighbor_value,
                            )
                            if step2_price is not None:
                                final_price = step2_price

                local_prices[code][key] = final_price

        # Step 5: adaptive volume-baseline updates (real observed volume only).
        for code, local_market in self.local_markets.items():
            for key, volume in local_volumes[code].items():
                local_market.update_volume_baseline(key[0], key[1], current_date, volume)
        for prefix, combined_market in self.combined_markets.items():
            for (p, key), volume in combined_total_volumes.items():
                if p == prefix:
                    combined_market.update_volume_baseline(key[0], key[1], current_date, volume)

        # Step 6: save history.
        current_tick = self.tick.get_tick()
        for code, local_market in self.local_markets.items():
            local_market.record_history(current_tick, local_prices[code], local_volumes[code])
        for prefix, combined_market in self.combined_markets.items():
            prices_for_prefix = {
                key: price for (p, key), price in combined_prices.items() if p == prefix
            }
            volumes_for_prefix = {
                key: volume for (p, key), volume in combined_total_volumes.items() if p == prefix
            }
            combined_market.record_history(current_tick, prices_for_prefix, volumes_for_prefix)

        # Step 7: clear accumulators, advance tick.
        for local_market in self.local_markets.values():
            local_market.clear_monthly_catch()
        for combined_market in self.combined_markets.values():
            combined_market.clear_direct_catch()
        self.tick.increase_tick()

    def _weighted_neighbor_value(
        self, target_port: str, high_tier_ports: set, neighbor_candidates: dict[str, float]
    ) -> float | None:
        candidates = [
            port for port in high_tier_ports
            if port != target_port and port in neighbor_candidates and (target_port, port) in self.distance_lookup
        ]
        if not candidates:
            return None

        weights = {port: 1.0 / self.distance_lookup[(target_port, port)] for port in candidates}
        total_weight = sum(weights.values())
        if total_weight == 0:
            return None
        return sum(weights[port] * neighbor_candidates[port] for port in candidates) / total_weight

    def get_species_prices_message(self) -> get_species_prices_pb2.GetSpeciesPricesResponse:
        messages = []
        valid_count = 0
        missing_count = 0

        for code, local_market in self.local_markets.items():
            if not local_market.price_history:
                continue
            _, latest_snapshot = local_market.price_history[-1]
            for (species_code, category_code), price in latest_snapshot["prices"].items():
                if price is None:
                    missing_count += 1
                    continue
                valid_count += 1
                messages.append(species_price_pb2.SpeciesPrice(
                    species=species_pb2.Species(species_code=species_code),
                    market_code=code,
                    price=price,
                    currency=self.currency,
                    category_code=category_code,
                ))

        self.logger.info(f"Price summary -- valid: {valid_count}, missing: {missing_count}")
        return get_species_prices_pb2.GetSpeciesPricesResponse(
            species_price_summary=species_price_pb2.SpeciesPriceSummary(species_prices=messages)
        )

    def export_results(self, simulation_id: str, folder_path: str):
        output_folder = Path(folder_path) / simulation_id
        output_folder.mkdir(parents=True, exist_ok=True)

        combined_path = output_folder / "combined_prices.csv"
        with combined_path.open("w", newline="") as f:
            f.write("tick,market_code,species,category_code,price,volume\n")
            for prefix, combined_market in self.combined_markets.items():
                for tick_id, snapshot in combined_market.price_history:
                    for (species_code, category_code), price in snapshot["prices"].items():
                        volume = snapshot["volumes"].get((species_code, category_code), 0.0)
                        price_str = "" if price is None else f"{price:.4f}"
                        f.write(f"{tick_id},{prefix},{species_code},{category_code},{price_str},{volume:.4f}\n")

        local_path = output_folder / "local_prices.csv"
        with local_path.open("w", newline="") as f:
            f.write("tick,market_code,species,category_code,price,catch\n")
            for code, local_market in self.local_markets.items():
                for tick_id, snapshot in local_market.price_history:
                    for (species_code, category_code), price in snapshot["prices"].items():
                        catch = snapshot["catches"].get((species_code, category_code), 0.0)
                        price_str = "" if price is None else f"{price:.4f}"
                        f.write(f"{tick_id},{code},{species_code},{category_code},{price_str},{catch:.4f}\n")

        self.logger.info(f"Export complete for simulation {simulation_id}")
