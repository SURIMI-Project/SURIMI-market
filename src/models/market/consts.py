"""
Constants used throughout the market model plugin: config filenames,
logger names, and empirically-grounded thresholds and candidate sets.
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

# Hand-rolled haversine, WGS84 equatorial radius; matches R's
# geosphere::distHaversine() default to 1e-10m precision.
EARTH_RADIUS_M = 6_378_137

# Step 2 neighbor-donor eligibility: a port may donate its Step-1
# predicted series only if it's among the target port's K nearest
# (haversine distance) -- replaces an earlier volume-share tier gate.
TOP_K_NEAREST_DONORS = 5

# Dynamic Stacked/DistanceOnly switch thresholds
LOW_VOLUME_RATIO_THRESHOLD = 0.10
HIGH_VOLUME_RATIO_THRESHOLD = 2.0

# Adaptive training-window candidate set — deliberately small,
# empirically tuned to this dataset, not a fine-grained sweep.
WINDOW_CANDIDATE_YEARS = {"2yr": 2, "3yr": 3, "4yr": 4, "6yr": 6, "Full": None}

# Rolling-origin validation methodology used to score each window
# candidate. Anchored at the latest available real data point, never
# at start_time.
N_ROLLING_SPLITS = 3
ROLLING_SPLIT_STEP_YEARS = 1
ROLLING_TEST_HORIZON_YEARS = 2

# Anchor schemes: "static" is the only one used in the live production
# path; "live" exists so the fidelity backtest can compare both.
ANCHOR_SCHEME_STATIC = "static"
ANCHOR_SCHEME_LIVE = "live"

# Trend confidence: per-calendar-month OLS on the full real historical
# record (no train/test split). Continuous, not a pass/fail gate --
# confidence = (months individually significant in the majority sign
# direction) / 12. See trend_confidence.py.
TREND_CONFIDENCE_T_THRESHOLD = 2.0

# Trend rate: sequential block size, in years, used to measure the
# historical rate from block-average log-prices (numpy.polyfit across
# all blocks that fit in the complete real record before train_end,
# built backward from train_end). 3yr empirically balances a
# long-enough fair test horizon against enough blocks to fit a
# meaningful slope -- a larger block size needs more lead-time history
# before its first usable cutoff, shortening how far into the future
# it can be fairly tested (4yr blocks tested out to only 1-7 years
# vs. 4-11 years for 2-3yr blocks).
TREND_RATE_BLOCK_YEARS = 3
