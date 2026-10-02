import os

import numpy as np
import pandas as pd
import pytest

from src.models.market.test.fidelity_backtest_harness import run_backtest_split
from src.models.market.level_volume_model import predict_level_volume
from src.models.market.stacked_model import fit_stacked_model, fit_distance_only_model

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
    for species_code in VALIDATED_SPECIES:
        species_rows = blended[(blended["species_code"] == species_code) & (blended["market_code"].str.len() == 2)]
        if species_rows.empty:
            print(f"{species_code}: no combined-market rows found, skipping")
            continue
        last_real_date = species_rows["date"].max()
        print(f"{species_code}: real data extends to {last_real_date.date()}")

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
    for species_code in VALIDATED_SPECIES:
        key = (species_code, "Fresh - Whole")
        combined_market = handler.market_system.combined_markets.get("ES")
        if combined_market is None or key not in combined_market.models:
            continue

        combined_rows = blended[
            (blended["market_code"] == "ES") & (blended["species_code"] == species_code)
        ][["date", "price", "landings_kg"]].sort_values("date")
        if combined_rows.empty:
            continue
        live_anchor_full = predict_level_volume(combined_rows, combined_market.models[key])
        live_anchor_full.index = combined_rows["date"]

        for code, local_market in handler.market_system.local_markets.items():
            fit_inputs = handler.market_system.step2_fit_inputs.get((code, key))
            if fit_inputs is None or key not in local_market.step2_stacked_models:
                continue

            port_monthly = fit_inputs["port_monthly"]
            neighbor_signal = fit_inputs["neighbor_signal"]
            train_end = port_monthly["date"].max()  # the same "latest_date" used at fit time

            try:
                live_stacked = fit_stacked_model(port_monthly, live_anchor_full, neighbor_signal, None, train_end)
                live_distance_only = fit_distance_only_model(
                    port_monthly, live_anchor_full, neighbor_signal, None, train_end
                )
            except (ValueError, np.linalg.LinAlgError) as e:
                print(f"{species_code}@{code}: live-anchor variant failed to fit ({e}) -- "
                      "static-anchor scheme is the only usable option for this port/species")
                continue

            static_stacked = local_market.step2_stacked_models[key]
            print(
                f"{species_code}@{code}: static-anchor gamma={static_stacked['gamma']:.3f} vs. "
                f"live-anchor gamma={live_stacked['gamma']:.3f} "
                f"(both fit successfully; relative MAPE comparison is test_fidelity_backtest.py's "
                f"job for the static scheme -- this confirms the live scheme is at least a stable, "
                f"fittable alternative worth backtesting head-to-head as a follow-up if these diverge "
                f"substantially)"
            )
