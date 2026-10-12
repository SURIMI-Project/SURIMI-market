"""
Seasonal baseline + volume-response model, fit in log-price space against a
relative volume anomaly: log(price) = baseline_log_price[month] + alpha +
beta * vol_anom, where vol_anom = (volume - baseline_volume[month]) /
baseline_volume[month] (a dimensionless "% above/below normal for this
month" anomaly, not a raw kilogram difference). Predicted price = exp(that).

Log-price as the fit target keeps predicted price from bending toward zero
instead of extrapolating through it under a sustained, far-outside-training
volume swing, rather than going negative. Using a relative (not absolute)
volume anomaly makes beta a uniform "% price change per % volume change"
across every calendar month, rather than skewed by each month's own typical
volume level. vol_anom is clipped to the min/max relative anomaly actually
observed during training before applying beta, since beta is only validated
within the range of data it was fit on.

Used identically for CombinedMarket's Model 2 and LocalMarket's base model --
the same fitting/prediction logic, just applied to combined-market-level or
per-port-level monthly data; a shared, single implementation, not forked
per caller. baseline_price (the plain arithmetic mean) is still computed
and returned alongside baseline_log_price, unused by fitting/prediction
here, purely so LocalMarket's neighbor-informed model's static anchor keeps
reading the same key with the same (linear-space) meaning it always has.
"""
import numpy as np
import pandas as pd

from . import consts


def with_month_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a "month" column derived from its "date"
    column -- the same two-line pattern repeated across this package's
    fit/predict functions, collapsed into one place."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    return df


def fit_level_only_model(
    monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp
) -> dict:
    """
    Fallback fit with no volume term at all, used when there isn't enough
    data/variance to fit the full level-volume model.
    """
    monthly = with_month_column(monthly)

    train = monthly[monthly["date"] < train_end]
    if train_start is not None:
        train = train[train["date"] >= train_start]

    baseline_price = train.groupby("month")["price"].mean()
    if baseline_price.dropna().empty:
        raise ValueError("not enough usable training rows to fit level-only model")

    return {"baseline_price": baseline_price, "alpha": 0.0, "beta": 0.0, "baseline_volume": None}


def predict_level_only(monthly: pd.DataFrame, params: dict) -> pd.Series:
    monthly = with_month_column(monthly)
    predicted = monthly["month"].map(params["baseline_price"]) + params["alpha"]
    predicted.index = monthly.index
    return predicted


def fit_level_volume_model(
    monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp
) -> dict:
    monthly = with_month_column(monthly)
    monthly["log_price"] = np.log(monthly["price"])

    train = monthly[monthly["date"] < train_end]
    if train_start is not None:
        train = train[train["date"] >= train_start]

    baseline_price = train.groupby("month")["price"].mean()
    baseline_log_price = train.groupby("month")["log_price"].mean()
    baseline_volume = train.groupby("month")["landings_kg"].mean()

    baseline_volume_mapped = train["month"].map(baseline_volume)
    train_price_anom = train["log_price"] - train["month"].map(baseline_log_price)
    train_vol_anom = (train["landings_kg"] - baseline_volume_mapped) / baseline_volume_mapped
    fit_mask = train_price_anom.notna() & train_vol_anom.notna() & (baseline_volume_mapped > 0)

    if fit_mask.sum() < 2 or train_vol_anom[fit_mask].var() == 0:
        raise ValueError("not enough usable training rows to fit level-volume model")

    beta, alpha = np.polyfit(train_vol_anom[fit_mask], train_price_anom[fit_mask], 1)

    return {
        "baseline_price": baseline_price,
        "baseline_log_price": baseline_log_price,
        "baseline_volume": baseline_volume,
        "alpha": float(alpha),
        "beta": float(beta),
        "vol_anom_min": float(train_vol_anom[fit_mask].min()),
        "vol_anom_max": float(train_vol_anom[fit_mask].max()),
    }


def predict_level_volume(monthly: pd.DataFrame, params: dict) -> pd.Series:
    monthly = with_month_column(monthly)

    baseline_volume_mapped = monthly["month"].map(params["baseline_volume"])
    vol_anom = (monthly["landings_kg"] - baseline_volume_mapped) / baseline_volume_mapped
    # A non-positive baseline (e.g. a recorded closed season every year in
    # training) makes the anomaly undefined -- NaN, not +-inf silently
    # clipped into looking like a real, confident reading.
    vol_anom = vol_anom.where(baseline_volume_mapped > 0)
    vol_anom = vol_anom.clip(lower=params["vol_anom_min"], upper=params["vol_anom_max"])
    predicted_log = (
        monthly["month"].map(params["baseline_log_price"]) + params["alpha"] + params["beta"] * vol_anom
    )
    predicted = np.exp(predicted_log)
    predicted.index = monthly.index
    return predicted


def fit_trend_rate_model(monthly: pd.DataFrame, train_end: pd.Timestamp) -> dict:
    """
    Historical rate from sequential TREND_RATE_BLOCK_YEARS-year blocks,
    anchored backward from train_end so the most recent data is always
    fully represented; any leftover partial year nearest the oldest
    observation is discarded. Each block's mean log-price is one data
    point; rate is the slope of a straight line through those points
    (numpy.polyfit), with the x-axis in years relative to train_end
    (negative for historical blocks, matching predict_trend_correction's
    trend_years convention exactly). Completely independent of window
    selection -- this function never reads or influences which window
    CombinedMarket.fit() picks.

    Block centers are the nominal midpoint of each block's time window,
    not the centroid of the data actually inside it -- a block that's
    only partially populated (e.g. the most recent one, when this
    function is called with a train_end shared across multiple series
    with different data-coverage end dates) introduces a small, usually
    negligible bias in its effective x-coordinate.

    Returns {"rate": 0.0, ...} (a harmless no-op, not a raised error) when
    fewer than 2 blocks fit -- a straight line needs at least 2 points,
    and real history shorter than 2*TREND_RATE_BLOCK_YEARS years is a
    normal, expected case (e.g. a newly-onboarded species/market), not a
    fitting failure.
    """
    history = monthly.dropna(subset=["price"]).copy()
    history = history[history["price"] > 0]
    history = history[history["date"] < train_end].sort_values("date")
    if history.empty:
        return {"rate": 0.0, "years_trained": 0.0}

    history["log_price"] = np.log(history["price"])
    block_days = pd.Timedelta(days=consts.TREND_RATE_BLOCK_YEARS * 365.25)

    oldest = history["date"].min()

    block_centers = []
    block_averages = []
    block_end = train_end
    while block_end - block_days >= oldest:
        block_start = block_end - block_days
        block = history[(history["date"] >= block_start) & (history["date"] < block_end)]
        if len(block) > 0:
            center_date = block_start + (block_end - block_start) / 2
            block_centers.append((center_date - train_end).days / 365.25)
            block_averages.append(block["log_price"].mean())
        block_end = block_start

    if len(block_centers) < 2:
        return {"rate": 0.0, "years_trained": 0.0}

    rate, _ = np.polyfit(block_centers, block_averages, 1)
    years_trained = len(block_centers) * consts.TREND_RATE_BLOCK_YEARS
    return {"rate": float(rate), "years_trained": float(years_trained)}


def predict_trend_correction(monthly: pd.DataFrame, params: dict) -> pd.Series:
    """
    Multiplicative trend correction:
    exp(confidence * rate * clip(trend_years, 0, years_trained)). Returns
    a correction FACTOR, not a price -- callers multiply this onto
    predict_level_volume's (or predict_level_only's) own output.
    trend_years is computed from params["trend_train_end"], clamped to
    [0, years_trained] (not symmetric): the rate here is measured purely
    from block averages with no per-row regression against trend_years,
    so there's no fitting-time assumption an asymmetric clamp would
    contradict. Any date at or before train_end gets a correction of
    exactly 1.0 (no-op), matching that every historical date is covered
    by get_price's real-price-first path before this model is ever
    reached anyway.
    """
    trend_years = (monthly["date"] - params["trend_train_end"]).dt.days / 365.25
    trend_years = trend_years.clip(lower=0.0, upper=params["trend_years_trained"])
    correction = np.exp(params["trend_confidence"] * params["trend_rate"] * trend_years)
    correction.index = monthly.index
    return correction


def score_mape(predicted: pd.Series, actual: pd.Series) -> float | None:
    """Mean absolute percentage error, ignoring rows where either side is NaN or actual is 0."""
    valid = predicted.notna() & actual.notna() & (actual != 0)
    if valid.sum() == 0:
        return None
    errors = (predicted[valid] - actual[valid]).abs() / actual[valid].abs()
    return float(errors.mean() * 100)


def build_complete_monthly_series(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex a sparse set of (date, price, landings_kg) rows -- e.g. a
    single market_code/species_code/category_code's blended rows from
    prices.csv -- to a complete, gap-filled monthly date range. Missing
    months get NaN price/landings_kg, so fit/predict functions never need
    to handle irregular gaps.
    """
    if rows.empty:
        raise ValueError("cannot build a monthly series from zero rows")

    rows = rows.sort_values("date")
    full_range = pd.date_range(rows["date"].min(), rows["date"].max(), freq="MS")
    monthly = rows.set_index("date").reindex(full_range)
    monthly.index.name = "date"
    return monthly.reset_index()


def compute_volume_baseline_stats(
    monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp
) -> dict[int, tuple[float, int]]:
    """
    Shared by CombinedMarket and LocalMarket (identical logic in both):
    per-calendar-month (sum, count) of non-null landings_kg within
    [train_start, train_end).
    """
    monthly = with_month_column(monthly)
    train = monthly[monthly["date"] < train_end]
    if train_start is not None:
        train = train[train["date"] >= train_start]

    stats: dict[int, tuple[float, int]] = {}
    for month, group in train.groupby("month"):
        valid = group["landings_kg"].dropna()
        if len(valid) > 0:
            stats[month] = (float(valid.sum()), int(len(valid)))
    return stats


def populate_price_time_series(
    price_time_series: dict[tuple[str, str, pd.Timestamp], float],
    prices_df: pd.DataFrame,
    market_code: str,
) -> None:
    """
    Shared by CombinedMarket and LocalMarket (identical logic in both).
    Filters to blended rows only (size is NaN) itself, rather than
    assuming the caller pre-filtered, so this holds regardless of what's
    passed in.
    """
    rows = prices_df[(prices_df["market_code"] == market_code) & (prices_df["size"].isna())]
    for row in rows.itertuples(index=False):
        price_time_series[(row.species_code, row.category_code, row.date)] = row.price
