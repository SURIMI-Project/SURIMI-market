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

# Neighbor-donor eligibility: a port may donate its base-model predicted
# series only if it's among the target port's K nearest (haversine
# distance) -- replaces an earlier volume-share tier gate.
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

# Trend confidence: per-calendar-month OLS (year -> price) on the full
# real historical record before train_end. Continuous, not a pass/fail
# gate -- confidence = (months individually significant, in the majority
# sign direction, against a degrees-of-freedom-scaled critical value --
# see trend_confidence_critical_value below, not a fixed threshold) / 12.
# See trend_confidence.py.

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


# Two-tailed 95% critical t-values by degrees of freedom (df 1-20),
# backing trend_confidence_critical_value below. scipy is NOT a declared
# dependency of this project (requirements.txt/pyproject.toml have no
# scipy entry; the only `import scipy` hits anywhere in the repo are in
# scripts/fisheries_data/*.ipynb notebooks, not in any package code
# pytest exercises), so this is a small hardcoded lookup table rather
# than scipy.stats.t.ppf(0.975, df).
_T_CRITICAL_95_BY_DF = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086,
}
_T_CRITICAL_95_ASYMPTOTIC = 1.960  # standard-normal fallback for df > 20


def trend_confidence_critical_value(df: int) -> float:
    """
    Two-tailed 95% critical t-value for `df` degrees of freedom --
    replaces a FIXED |t| > 2.0 threshold that was statistically wrong for
    low-df months (e.g. a month with only 3 years of data has df=1, where
    the true 95% critical value is ~12.7, not 2.0 -- thin-data months
    were getting flagged "significant" far too easily under the old flat
    threshold, while well-sampled months were tested closer to a real
    bar). Covers df 1-20 -- comfortably past the ~1-13 range the real
    ~15-year dataset's per-calendar-month tests can ever produce
    (ARCHITECTURE.md) -- falling back to the standard-normal asymptotic
    value (1.960) for any df beyond the table, where the t-distribution
    has already converged close enough to it not to matter.
    """
    if df < 1:
        return _T_CRITICAL_95_ASYMPTOTIC
    return _T_CRITICAL_95_BY_DF.get(df, _T_CRITICAL_95_ASYMPTOTIC)
