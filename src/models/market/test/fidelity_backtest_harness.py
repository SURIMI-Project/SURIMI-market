"""
Fidelity backtest harness (spec section 11 layer 3). Runs the real
MarketHandler against a real prices.csv with start_time set to a
deliberately early as_of_date, feeds it real historical landings tick by
tick as simulated catches, and compares its reported prices against the
real recorded prices for the held-out window that follows.
"""
import datetime
import tempfile
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.models.market.MarketHandler import MarketHandler
from src.models.market.level_volume_model import score_mape


class _Market:
    def __init__(self, market_code):
        self.market_code = market_code


class _Species:
    def __init__(self, species_code):
        self.species_code = species_code


class _PriceCategory:
    def __init__(self, category_code):
        self.category_code = category_code


class _Items:
    def __init__(self, markets, species, price_categories):
        self.markets = markets
        self.species = species
        self.price_categories = price_categories


class _Simulation:
    def __init__(self, items):
        self.items = items


def _build_fake_simulation(market_codes, species_codes, category_codes):
    return _Simulation(_Items(
        markets=[_Market(m) for m in market_codes],
        species=[_Species(s) for s in species_codes],
        price_categories=[_PriceCategory(c) for c in category_codes],
    ))


def run_backtest_split(
    root_config_dir: str,
    scenario_id: str,
    as_of_date: pd.Timestamp,
    test_horizon_years: int,
    species_codes: list[str],
    return_handler: bool = False,
):
    prices_df = pd.read_csv(f"{root_config_dir}/{scenario_id}/prices.csv", parse_dates=["date"])
    if "size" not in prices_df.columns:
        prices_df["size"] = None
    markets_df = pd.read_csv(f"{root_config_dir}/{scenario_id}/markets.csv")
    species_df = pd.read_csv(f"{root_config_dir}/{scenario_id}/species.csv")

    local_codes = markets_df[markets_df["market_code"].str.len() == 5]["market_code"].tolist()
    category_codes = prices_df["category_code"].dropna().unique().tolist()

    start_time = as_of_date.to_pydatetime().replace(tzinfo=None)

    # Give MarketHandler a TRUNCATED view of prices.csv (rows strictly before
    # as_of_date) for fitting AND for its own real-price-first lookup
    # (load_price_time_series) -- otherwise init_simulation's
    # latest_date = prices_df["date"].max() would resolve to the real file's
    # true end date regardless of as_of_date, and real-price-first would find
    # a match for every tick in the nominal "held out" test window, silently
    # substituting the real recorded price for the model's own prediction at
    # every single comparison point (a perfect, meaningless 0.0 MAPE). The
    # full, untruncated prices_df is still used below for feeding real
    # landings tick-by-tick and for the final ground-truth comparison -- only
    # what MarketHandler itself gets to see at init time is restricted.
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

        # From here on, the test window is driven by stepping the
        # market_system forward and feeding it that same window's REAL
        # landings as the simulated catch input, so Step 1/2 see real volumes
        # exactly as a live simulation would.
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
