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

# Trend-term gate (spec 2026-10-07-combined-market-trend-term-design.md,
# section 3): per-calendar-month OLS on the FULL real historical record
# (no train/test split), used to decide whether a (species_code,
# category_code) key's CombinedMarket fit should include a trend term at
# all. Deliberately strict -- chosen to cleanly separate the one
# validated real case (ANE@ES: 12/12 months agree in sign, 11/12
# individually significant) from the nearest false positive (HKE@FR:
# 12/12 sign agreement, but only 2/12 significant).
TREND_GATE_T_THRESHOLD = 2.0
TREND_GATE_MIN_SIGNIFICANT_MONTHS = 9
