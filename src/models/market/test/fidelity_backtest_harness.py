"""
Fidelity backtest harness. Runs the real MarketHandler against a real
prices.csv with start_time set to a deliberately early as_of_date, feeds
it real historical landings tick by tick as simulated catches, and
compares its reported prices against the real recorded prices for the
held-out window that follows.
"""
import datetime
import tempfile
from pathlib import Path
from typing import Callable

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.models.market.MarketHandler import MarketHandler
from src.models.market.level_volume_model import score_mape
from src.models.market.test.fake_simulation import (
    FakeMarket as _Market,
    FakeSpecies as _Species,
    FakePriceCategory as _PriceCategory,
    FakeItems as _Items,
    FakeSimulation as _Simulation,
)


def _build_fake_simulation(market_codes, species_codes, category_codes):
    return _Simulation(_Items(
        markets=[_Market(m) for m in market_codes],
        species=[_Species(s) for s in species_codes],
        price_categories=[_PriceCategory(c) for c in category_codes],
    ))


def _live_anchor_resolver_from(combined_monthly_by_key: dict) -> Callable:
    """Builds an anchor_resolver (MarketSystem.fit_all's injection point)
    that anchors on CombinedMarket's own live tick-by-tick prediction
    instead of its frozen seasonal baseline -- analysis-only, used to
    compare against the production default. Moved here from the
    now-deleted CombinedMarket.get_live_anchor_series; identical math,
    just relocated out of the production class since nothing in
    production ever used it."""
    def resolver(combined_market, species_code, category_code, port_monthly):
        key = (species_code, category_code)
        own_monthly = combined_monthly_by_key.get(combined_market.market_code, {}).get(key)
        if own_monthly is None or key not in combined_market.models:
            return None
        predicted = combined_market.models[key].predict_series(own_monthly).copy()
        predicted.index = own_monthly["date"]
        return predicted.reindex(pd.DatetimeIndex(port_monthly["date"]))
    return resolver


def run_backtest_split(
    root_config_dir: str,
    scenario_id: str,
    as_of_date: pd.Timestamp,
    test_horizon_years: int,
    species_codes: list[str],
    return_handler: bool = False,
    anchor_scheme: str = "static",
):
    prices_df = pd.read_csv(f"{root_config_dir}/{scenario_id}/prices.csv", parse_dates=["date"])
    if "size" not in prices_df.columns:
        prices_df["size"] = None
    markets_df = pd.read_csv(f"{root_config_dir}/{scenario_id}/markets.csv")
    species_df = pd.read_csv(f"{root_config_dir}/{scenario_id}/species.csv")

    local_codes = markets_df[markets_df["market_code"].str.len() == 5]["market_code"].tolist()
    category_codes = prices_df["category_code"].dropna().unique().tolist()

    start_time = as_of_date.to_pydatetime().replace(tzinfo=None)

    # MarketHandler only gets a TRUNCATED view of prices.csv (rows strictly
    # before as_of_date), for both fitting and its own real-price-first
    # lookup. Otherwise latest_date would resolve to the file's true end
    # date and real-price-first would return the real recorded price for
    # every "held out" tick, producing a meaningless 0.0 MAPE. The full,
    # untruncated prices_df is still used below for feeding real landings
    # and for the final ground-truth comparison.
    truncated_prices = prices_df[prices_df["date"] < as_of_date]

    with tempfile.TemporaryDirectory() as tmp_root:
        scenario_dir = Path(tmp_root) / scenario_id
        scenario_dir.mkdir(parents=True)
        truncated_prices.to_csv(scenario_dir / "prices.csv", index=False)
        markets_df.to_csv(scenario_dir / "markets.csv", index=False)
        species_df.to_csv(scenario_dir / "species.csv", index=False)

        handler = MarketHandler("backtest", scenario_id, start_time, 1)
        simulation = _build_fake_simulation(local_codes, species_codes, category_codes)
        response, err = handler.init_simulation(simulation, tmp_root)
        if err is not None:
            raise RuntimeError(f"backtest init_simulation failed: {err.message}")

        if anchor_scheme != "static":
            # init_simulation always fits with the production default
            # (static). Re-fit Step 2 with the requested alternate scheme on
            # the same truncated data/latest_date before stepping, so only
            # Step 2's models change -- Step 1 and CombinedMarket don't
            # depend on anchor_scheme.
            latest_date = pd.Timestamp(truncated_prices["date"].max())
            blended_for_resolver = truncated_prices[truncated_prices["size"].isna()]
            combined_monthly_by_key = {
                prefix: handler.market_system._build_monthly_by_key(
                    blended_for_resolver[blended_for_resolver["market_code"] == prefix]
                )
                for prefix in handler.market_system.combined_markets
            }
            resolver = _live_anchor_resolver_from(combined_monthly_by_key)
            handler.market_system.fit_all(truncated_prices, latest_date, anchor_resolver=resolver)

        # Step the market_system forward, feeding it this window's real
        # landings as simulated catch input so Step 1/2 see real volumes.
        test_end = as_of_date + pd.DateOffset(years=test_horizon_years)
        blended = prices_df[prices_df["size"].isna()]  # full, untruncated

        current_date = pd.Timestamp(handler.market_system.tick.get_date())
        while current_date < test_end:
            tick_rows = blended[blended["date"] == current_date]
            for row in tick_rows.itertuples(index=False):
                if row.species_code not in species_codes:
                    continue
                local_market = handler.market_system.local_markets.get(row.market_code)
                if local_market is not None and pd.notna(row.landings_kg):
                    local_market.add_monthly_catch(row.species_code, row.category_code, row.landings_kg)

            handler.market_system.step()
            current_date = pd.Timestamp(handler.market_system.tick.get_date())

        # Compare against REAL recorded prices for the same (date, market, species).
        real_lookup = {
            (r.date, r.market_code, r.species_code, r.category_code): r.price
            for r in blended.itertuples(index=False)
        }

        by_key_market = {}
        for code, local_market in handler.market_system.local_markets.items():
            for tick_id, snapshot in local_market.price_history:
                tick_date = start_time_plus_months(start_time, tick_id)
                for key, predicted_price in snapshot["prices"].items():
                    real_price = real_lookup.get((tick_date, code, key[0], key[1]))
                    if real_price is None or predicted_price is None:
                        continue
                    by_key_market.setdefault((key[0], code), []).append((predicted_price, real_price))

        rows = []
        for (species_code, market_code), pairs in by_key_market.items():
            predicted = pd.Series([p for p, _ in pairs])
            real = pd.Series([r for _, r in pairs])
            mape = score_mape(predicted, real)
            rows.append({
                "species_code": species_code, "market_code": market_code,
                "n_test_months": len(pairs), "mape": mape,
            })

        results_df = pd.DataFrame(rows)
        if return_handler:
            return results_df, handler
        return results_df


def start_time_plus_months(start_time: datetime.datetime, n_months: int) -> pd.Timestamp:
    return pd.Timestamp(start_time + relativedelta(months=n_months))
