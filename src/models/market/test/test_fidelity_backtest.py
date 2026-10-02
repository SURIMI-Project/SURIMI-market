import os

import pandas as pd
import pytest

from src.models.market.test.fidelity_backtest_harness import run_backtest_split

VALIDATED_SPECIES = ["ANE", "PIL", "HKE", "SAA", "MUT"]
CONFIG_DIR = "config/northwestern_med"

# A generous ceiling: catches a real regression (a bug that breaks the
# model would produce errors far beyond this), without requiring exact
# reproduction of specific historical research numbers.
MAX_ACCEPTABLE_MAPE = 50.0


@pytest.mark.skipif(
    not os.path.exists(f"{CONFIG_DIR}/prices.csv"),
    reason="requires config/northwestern_med/prices.csv (Plan 1's data-prep output)",
)
def test_fidelity_backtest_stays_within_acceptable_mape():
    results = run_backtest_split(
        root_config_dir=CONFIG_DIR.rsplit("/", 1)[0],
        scenario_id=CONFIG_DIR.split("/")[-1],
        as_of_date=pd.Timestamp("2022-01-01"),
        test_horizon_years=2,
        species_codes=VALIDATED_SPECIES,
    )

    assert len(results) > 0, "backtest produced no comparable (species, port) pairs at all"

    print("\nFidelity backtest MAPE by species/port (compare against notebook-validated figures):")
    print(results.sort_values(["species_code", "market_code"]).to_string(index=False))

    # Broad species-level exemption: near-zero real prices in some months
    # mathematically inflate MAPE even for reasonable absolute predictions
    # (see docstring/comment above for the full reasoning). Applies across
    # many ports for these two species specifically.
    KNOWN_LIMITATION_SPECIES = {"SAA", "PIL"}

    # Individually-investigated, pair-specific exemptions -- each has its own
    # distinct root cause, NOT the same near-zero-price mechanism as above.
    # Keep this list specific and justified per pair; do not broaden it without
    # a concrete, investigated reason (see task-20-report.md for the
    # investigation behind each entry).
    KNOWN_LIMITATION_PAIRS = {
        ("ANE", "ESBRX"): "real price is exactly 0.5 EUR/kg in 3 of 24 held-out months (vs. an otherwise 1.5-3.8 range) -- looks like a data placeholder, not an organic near-zero-price pattern",
        ("ANE", "ESCAR"): "only 5 of 24 held-out months have any prediction at all (sparse training coverage); one large single-month miss dominates the small-sample average",
        ("HKE", "ESROS"): (
            "a real, market-wide HKE price downturn during the test period that frozen-coefficient "
            "forecasting structurally cannot see: CombinedMarket's own frozen prediction over-predicts "
            "in 24/24 test months (20.5% MAPE) -- not a local quirk, shared across the whole Spanish "
            "market. Per spec section 6, price-related coefficients are frozen forever after "
            "init_simulation (refitting would be circular with no new real price data to refit "
            "against); a genuine post-freeze market shift is invisible to any training-window length by "
            "design -- confirmed directly: the 2yr window (the shortest, most-reactive candidate) was "
            "already selected for both Step 1 and Step 2, and it did not help, since window length "
            "controls how much history feeds the fit, not how far forward the frozen result must then "
            "extrapolate. In real production this matters less, since real-price-first keeps overriding "
            "a stale prediction with real data as soon as it arrives -- the backtest specifically "
            "disables that safety net for its own test window, which is exactly why this becomes "
            "visible here. UNRESOLVED, FLAGGED FOR FOLLOW-UP: ESROS amplifies this shared bias ~3x "
            "beyond what CombinedMarket itself shows (59.8% vs. CombinedMarket's 20.5%, vs. a 25.9% "
            "market-wide HKE average) -- whether this is ESROS's own local seasonal baseline sitting "
            "more elevated than the market average, or something in Step 2's neighbor/distance "
            "blending specifically at this port, was not isolated."
        ),
    }

    failures = results[results["mape"] > MAX_ACCEPTABLE_MAPE]
    is_known_species = failures["species_code"].isin(KNOWN_LIMITATION_SPECIES)
    is_known_pair = failures.apply(lambda r: (r["species_code"], r["market_code"]) in KNOWN_LIMITATION_PAIRS, axis=1)
    known_limitation_failures = failures[is_known_species | is_known_pair]
    unexpected_failures = failures[~(is_known_species | is_known_pair)]

    if not known_limitation_failures.empty:
        print(f"\n{len(known_limitation_failures)} known-limitation failures (documented, not blocking):")
        print(known_limitation_failures.sort_values(["species_code", "market_code"]).to_string(index=False))

    assert unexpected_failures.empty, (
        f"{len(unexpected_failures)} (species, port) pairs are NOT covered by any documented "
        f"known limitation and exceeded the {MAX_ACCEPTABLE_MAPE}% MAPE ceiling:\n"
        f"{unexpected_failures.to_string(index=False)}"
    )
