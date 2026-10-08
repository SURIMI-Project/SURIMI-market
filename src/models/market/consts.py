"""
Constants used throughout the market model plugin: config filenames,
logger names, and the empirically-grounded thresholds/candidate sets
from the validated research (spec sections 4, 6, 7, 9).
"""

# Configurations (matches Elsa's consts.py convention)
SPECIES_CSV = "species.csv"
PORTS_CSV = "markets.csv"
PRICES_CSV = "prices.csv"
SIZE_RATIOS_CSV = "size_ratios.csv"

SUPPORTED_CURRENCY = "EUR"
SUPPORTED_MEASUREMENT_UNIT = "kg"

# Logging (one logger per class, matching Elsa's named-logger-per-class pattern)
MARKET_SYSTEM_LOGGER_NAME = "MarketMarketSystem"
COMBINED_MARKET_LOGGER_NAME = "CombinedMarket"
LOCAL_MARKET_LOGGER_NAME = "MarketLocalMarket"
SIMULATION_HANDLER_LOGGER_NAME = "market_simulation"

# Distance (spec section 4): hand-rolled haversine, WGS84 equatorial radius,
# confirmed to match R's geosphere::distHaversine() default and
# .cache/test/port_distances.csv to 1e-10m precision.
EARTH_RADIUS_M = 6_378_137

# Volume-share tiers for High-tier neighbor determination (spec section 9.3)
MID_TIER_MIN_VOLUME_SHARE_PCT = 0.2
HIGH_TIER_MIN_VOLUME_SHARE_PCT = 2.0

# Dynamic Stacked/DistanceOnly switch thresholds (spec section 5 step 4, section 9.1)
LOW_VOLUME_RATIO_THRESHOLD = 0.10
HIGH_VOLUME_RATIO_THRESHOLD = 2.0

# Adaptive training-window candidate set (spec section 7) — deliberately
# small, empirically tuned to this dataset, not a fine-grained sweep.
WINDOW_CANDIDATE_YEARS = {"2yr": 2, "3yr": 3, "4yr": 4, "6yr": 6, "Full": None}

# Rolling-origin validation methodology (spec section 7) used to score
# each window candidate. Anchored at the latest available real data point,
# never at start_time (see plan header, decision 5).
N_ROLLING_SPLITS = 3
ROLLING_SPLIT_STEP_YEARS = 1
ROLLING_TEST_HORIZON_YEARS = 2

# Step-2 anchor schemes (spec section 5's open item, resolved in the plan
# header decision 1-2). "static" is the only scheme used in the live
# production path; "live" exists so the fidelity backtest can compare both.
ANCHOR_SCHEME_STATIC = "static"
ANCHOR_SCHEME_LIVE = "live"

# Trend confidence (spec 2026-10-08-combined-market-trend-term-v2-design.md
# section 3, Step 2): per-calendar-month OLS on the FULL real historical
# record (no train/test split), continuous -- NOT a pass/fail gate (v1's
# TREND_GATE_MIN_SIGNIFICANT_MONTHS threshold is gone; see
# trend_confidence.py). confidence = (months individually significant in
# the majority sign direction) / 12.
TREND_CONFIDENCE_T_THRESHOLD = 2.0

# Trend rate (spec v2 section 3, Step 3): sequential block size, in years,
# used to measure the historical rate from block-average log-prices
# (numpy.polyfit across all blocks that fit in the complete real record
# before train_end, built forward from the oldest available date). 3yr
# was empirically found to balance a long-enough fair test horizon
# against enough blocks to fit a meaningful slope -- a larger block size
# needs more lead-time history before its first usable cutoff, which
# shortens how far into the future that configuration can be fairly
# tested (confirmed directly: 4yr blocks could only be tested out to 1-7
# years vs. 4-11 years for 2-3yr blocks) -- see the v2 spec section 2.
TREND_RATE_BLOCK_YEARS = 3
