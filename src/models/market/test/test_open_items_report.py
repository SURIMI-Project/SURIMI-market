import os

import pandas as pd
import pytest

from src.models.market import consts
from src.models.market.test.fidelity_backtest_harness import run_backtest_split

VALIDATED_SPECIES = ["ANE", "PIL", "HKE", "SAA", "MUT"]
CONFIG_DIR = "config/northwestern_med"


@pytest.mark.skipif(
    not os.path.exists(f"{CONFIG_DIR}/prices.csv"),
    reason="requires config/northwestern_med/prices.csv (Plan 1's data-prep output)",
)
def test_open_items_report():
    prices_df = pd.read_csv(f"{CONFIG_DIR}/prices.csv", parse_dates=["date"])
    # Same normalization run_backtest_split itself applies (fidelity_backtest_harness.py):
    # the real prices.csv has no "size" column at all (it's an optional, not-yet-produced
    # Plan 1 output), so without this, filtering to "size is NaN" blended rows would raise
    # a KeyError here instead of correctly treating every row as blended.
    if "size" not in prices_df.columns:
        prices_df["size"] = None
    blended = prices_df[prices_df["size"].isna()]

    print("\n=== 1. Real data extent and transition-jump size (spec section 13) ===")
    last_real_dates = {}
    for species_code in VALIDATED_SPECIES:
        species_rows = blended[(blended["species_code"] == species_code) & (blended["market_code"].str.len() == 2)]
        if species_rows.empty:
            print(f"{species_code}: no combined-market rows found, skipping")
            continue
        last_real_date = species_rows["date"].max()
        last_real_dates[species_code] = last_real_date
        print(f"{species_code}: real data extends to {last_real_date.date()}")

    if last_real_dates:
        shared_as_of_date = pd.Timestamp(min(last_real_dates.values()))
        if len(set(last_real_dates.values())) > 1:
            print(f"\nNote: species have DIFFERENT last-real-data dates; using the "
                  f"earliest ({shared_as_of_date.date()}) as a shared as_of_date for "
                  f"the transition-jump comparison below -- some species may show a "
                  f"jump measured slightly before their own true data boundary.")

        # Fit exactly at the shared real-data boundary (as_of_date excludes that
        # date itself from fitting/real-price-lookup -- the same truncation
        # run_backtest_split already does for its own test window), then compare
        # the frozen model's retrospective prediction for that one held-out tick
        # against what the real recorded value actually was. This is the actual
        # "how large is the real-to-predicted transition jump" measurement spec
        # section 13 requires -- not just how far real data extends.
        jump_results, jump_handler = run_backtest_split(
            root_config_dir=CONFIG_DIR.rsplit("/", 1)[0],
            scenario_id=CONFIG_DIR.split("/")[-1],
            as_of_date=shared_as_of_date,
            test_horizon_years=1,
            species_codes=list(last_real_dates.keys()),
            return_handler=True,
        )

        print(f"\nTransition jump at {shared_as_of_date.date()} (CombinedMarket 'ES' level):")
        combined_market = jump_handler.market_system.combined_markets.get("ES")
        if combined_market is not None and combined_market.price_history:
            tick_id, snapshot = combined_market.price_history[0]
            for species_code in last_real_dates:
                key = (species_code, "Fresh - Whole")
                predicted_value = snapshot["prices"].get(key)
                real_rows = blended[
                    (blended["species_code"] == species_code)
                    & (blended["market_code"] == "ES")
                    & (blended["date"] == shared_as_of_date)
                ]
                if predicted_value is None or real_rows.empty:
                    print(f"  {species_code}: no comparable prediction/real value at this tick, skipping")
                    continue
                real_value = real_rows["price"].iloc[0]
                abs_jump = abs(predicted_value - real_value)
                pct_jump = (abs_jump / real_value * 100) if real_value else float("nan")
                print(f"  {species_code}: predicted={predicted_value:.4f}, real={real_value:.4f}, "
                      f"abs_diff={abs_jump:.4f} ({pct_jump:.1f}%)")
        else:
            print("  no CombinedMarket price_history available to compare")

    results, handler = run_backtest_split(
        root_config_dir=CONFIG_DIR.rsplit("/", 1)[0],
        scenario_id=CONFIG_DIR.split("/")[-1],
        as_of_date=pd.Timestamp("2022-01-01"),
        test_horizon_years=2,
        species_codes=VALIDATED_SPECIES,
        return_handler=True,
    )

    print("\n=== 2. Combined/Local consistency drift (spec section 10) ===")
    for species_code in VALIDATED_SPECIES:
        key = (species_code, "Fresh - Whole")
        combined_market = handler.market_system.combined_markets.get("ES")
        if combined_market is None or key not in combined_market.models:
            continue

        gaps = []
        for tick_id, snapshot in combined_market.price_history:
            combined_price = snapshot["prices"].get(key)
            if combined_price is None:
                continue

            local_prices_this_tick = []
            local_volumes_this_tick = []
            for code, local_market in handler.market_system.local_markets.items():
                if tick_id >= len(local_market.price_history):
                    continue
                _, local_snapshot = local_market.price_history[tick_id]
                price = local_snapshot["prices"].get(key)
                volume = local_snapshot["catches"].get(key, 0.0)
                if price is not None:
                    local_prices_this_tick.append(price)
                    local_volumes_this_tick.append(volume)

            if not local_prices_this_tick or sum(local_volumes_this_tick) == 0:
                continue
            weighted_local = sum(
                p * v for p, v in zip(local_prices_this_tick, local_volumes_this_tick)
            ) / sum(local_volumes_this_tick)
            gaps.append(abs(weighted_local - combined_price) / combined_price * 100)

        if gaps:
            print(f"{species_code}: mean |Local(volume-weighted) - Combined| gap = {sum(gaps)/len(gaps):.2f}%")

    print("\n=== 3. Static vs. live Step-2 anchor comparison (spec sections 5/14) ===")
    # A REAL head-to-head: two independent, fully-stepped backtest runs over
    # the identical as_of_date/test_horizon/species, one with each anchor
    # scheme (MarketSystem.fit_all(..., anchor_scheme=...), both now
    # production-supported paths -- STATIC remains the only scheme the live
    # gRPC path ever uses by default). This replaces the previous
    # gamma-coefficient-only comparison, which confounded anchor-scheme
    # choice with window-length choice (the live refit always used full
    # history, static used its own adaptively selected window) and could
    # only observe divergence, never say which scheme was actually better.
    live_results, live_handler = run_backtest_split(
        root_config_dir=CONFIG_DIR.rsplit("/", 1)[0],
        scenario_id=CONFIG_DIR.split("/")[-1],
        as_of_date=pd.Timestamp("2022-01-01"),
        test_horizon_years=2,
        species_codes=VALIDATED_SPECIES,
        anchor_scheme=consts.ANCHOR_SCHEME_LIVE,
        return_handler=True,
    )

    merged = results.merge(
        live_results, on=["species_code", "market_code"], suffixes=("_static", "_live"), how="inner"
    )
    if merged.empty:
        print("No (species, port) pairs comparable between the two anchor-scheme runs.")
    else:
        merged["mape_diff"] = merged["mape_live"] - merged["mape_static"]  # negative => live wins

        def _used_step2(h, species_code, market_code):
            key = (species_code, "Fresh - Whole")
            local_market = h.market_system.local_markets.get(market_code)
            return local_market is not None and key in local_market.step2_stacked_models

        merged["step2_both_schemes"] = merged.apply(
            lambda r: _used_step2(handler, r["species_code"], r["market_code"])
            and _used_step2(live_handler, r["species_code"], r["market_code"]),
            axis=1,
        )

        static_wins = int((merged["mape_diff"] > 0).sum())
        live_wins = int((merged["mape_diff"] < 0).sum())
        ties = int((merged["mape_diff"] == 0).sum())
        print(
            f"Overall (all {len(merged)} comparable pairs, regardless of whether Step 2 was used "
            f"under both schemes): static better in {static_wins}, live better in {live_wins}, "
            f"tied in {ties}. Mean MAPE -- static: {merged['mape_static'].mean():.2f}%, "
            f"live: {merged['mape_live'].mean():.2f}%."
        )

        clean = merged[merged["step2_both_schemes"]]
        if clean.empty:
            print("No pairs had Step 2 fitted under BOTH schemes -- cannot isolate anchor scheme alone.")
        else:
            clean_static_wins = int((clean["mape_diff"] > 0).sum())
            clean_live_wins = int((clean["mape_diff"] < 0).sum())
            clean_ties = int((clean["mape_diff"] == 0).sum())
            print(
                f"\nIsolated to the {len(clean)} pairs where Step 2 was actually fitted under BOTH "
                f"schemes (the fair, confound-free comparison): static better in {clean_static_wins}, "
                f"live better in {clean_live_wins}, tied in {clean_ties}. Mean MAPE -- "
                f"static: {clean['mape_static'].mean():.2f}%, live: {clean['mape_live'].mean():.2f}%."
            )
            print(clean.sort_values("mape_diff")[
                ["species_code", "market_code", "mape_static", "mape_live", "mape_diff"]
            ].to_string(index=False))
