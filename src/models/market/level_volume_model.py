"""
Seasonal baseline + volume-response model (spec section 3's "Model 2"), fit
in log-price space against a RELATIVE volume anomaly (spec section 9's
negative-price-extrapolation fix, extended):
log(price) = baseline_log_price[month] + alpha + beta * vol_anom, where
vol_anom = (volume - baseline_volume[month]) / baseline_volume[month] (a
dimensionless "% above/below normal for this month" anomaly, not a raw
kilogram difference) -- predicted price = exp(that).

Three distinct fixes are stacked here, found and validated separately:
  - Log-price (not raw price) as the fit target means a sustained,
    far-outside-training volume swing bends the predicted price towards
    zero instead of extrapolating straight through it -- confirmed
    empirically (never goes negative, even under a 20-year sustained-growth
    stress scenario where the original linear fit went negative).
  - RELATIVE (not absolute) volume anomaly means beta means "% price change
    per % volume change", uniformly across every calendar month, instead of
    "% price change per extra KG" -- which, with a single frozen beta,
    made the model's sensitivity to the SAME relative catch scenario wildly
    uneven across months purely because each month's own typical volume
    differs (confirmed both mathematically and visibly in a 20-year
    scenario forecast: MUT's Sep/Oct/Nov, its lowest-price/highest-volume
    months, swung 2-6x harder than its other months for the same scenario).
  - CLAMPING vol_anom to the min/max relative anomaly actually observed
    during training, before applying beta: beta is only validated within
    the range of data it was fit on, so a real or synthetic volume anomaly
    beyond that range holds steady at the most extreme validated response
    instead of extrapolating beta further outward. This is the direct fix
    for the root cause behind every extrapolation-driven finding in this
    investigation, on BOTH sides: the original MUT near-zero-price stress
    scenario (bounded far more comfortably with clamping than without) and
    a newly-found real-data case (HKE@ESQAJ/ESROS/ESSAG, where an actual
    2022-2023 catch collapse far outside training experience caused a
    large price overshoot -- substantially reduced, though not fully
    resolved, by clamping, e.g. ESSAG's held-out MAPE 97.1% -> 68.6%).
All three are a net improvement in real-data accuracy for most, though not
all, validated species (see
docs/superpowers/specs/2026-10-02-market-model-handler-design.md section 15).

Used identically for CombinedMarket's Model 2 and LocalMarket's Step 1 --
the same fitting/prediction logic, just applied to combined-market-level
or per-port-level monthly data (spec section 3). Deliberately applied to
BOTH: this is a shared, single implementation, not forked per caller.
baseline_price (the plain arithmetic mean) is still computed and returned
alongside baseline_log_price, unused by fitting/prediction here, purely so
LocalMarket Step 2's static anchor (get_baseline_price_table/
build_static_anchor_series) keeps reading the same key with the same
(linear-space) meaning it always has -- Step 2 itself (its own separate
delta/gamma regression, with its own separate, still-absolute volume
anomaly term) is explicitly out of scope for this change; it still has no
price-floor protection and no relative-volume fix of its own.

Adapted from the validated fit_level_volume_model/predict_level_volume
functions from the research (which also explored a log-price fit for the
per-port Stacked model), with column names adapted to the production
schema (date/price/landings_kg instead of Date/Price_EUR_kg/Volume_Kg).
"""
import numpy as np
import pandas as pd


def fit_level_only_model(
    monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp
) -> dict:
    """
    Fallback fit with no volume term at all (spec section 9 point 2's
    "level-offset-only" fallback step) -- used when there isn't enough
    data/variance to fit the full level-volume model.
    """
    monthly = monthly.copy()
    monthly["month"] = monthly["date"].dt.month

    train = monthly[monthly["date"] < train_end]
    if train_start is not None:
        train = train[train["date"] >= train_start]

    baseline_price = train.groupby("month")["price"].mean()
    if baseline_price.dropna().empty:
        raise ValueError("not enough usable training rows to fit level-only model")

    return {"baseline_price": baseline_price, "alpha": 0.0, "beta": 0.0, "baseline_volume": None}


def predict_level_only(monthly: pd.DataFrame, params: dict) -> pd.Series:
    monthly = monthly.copy()
    monthly["month"] = monthly["date"].dt.month
    predicted = monthly["month"].map(params["baseline_price"]) + params["alpha"]
    predicted.index = monthly.index
    return predicted


def fit_level_volume_model(
    monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp
) -> dict:
    monthly = monthly.copy()
    monthly["month"] = monthly["date"].dt.month
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
    monthly = monthly.copy()
    monthly["month"] = monthly["date"].dt.month

    baseline_volume_mapped = monthly["month"].map(params["baseline_volume"])
    vol_anom = (monthly["landings_kg"] - baseline_volume_mapped) / baseline_volume_mapped
    vol_anom = vol_anom.clip(lower=params["vol_anom_min"], upper=params["vol_anom_max"])
    predicted_log = (
        monthly["month"].map(params["baseline_log_price"]) + params["alpha"] + params["beta"] * vol_anom
    )
    predicted = np.exp(predicted_log)
    predicted.index = monthly.index
    return predicted


def fit_level_volume_trend_model(monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp) -> dict:
    """
    Trend-augmented sibling of fit_level_volume_model (spec
    2026-10-07-combined-market-trend-term-design.md, section 4). Extends
    the existing 1-D vol_anom -> price_anom regression to a 2-D
    least-squares fit adding trend_years = (date - train_end) / 365.25 as
    a second predictor (zero at train_end, negative across the training
    window). Only ever called for a key that has already passed
    trend_gate.passes_trend_gate -- this function itself does not re-check
    eligibility.
    """
    monthly = monthly.copy()
    monthly["month"] = monthly["date"].dt.month
    monthly["log_price"] = np.log(monthly["price"])
    monthly["trend_years"] = (monthly["date"] - train_end).dt.days / 365.25

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

    if fit_mask.sum() < 3 or train_vol_anom[fit_mask].var() == 0:
        raise ValueError("not enough usable training rows to fit level-volume-trend model")

    X = np.column_stack([
        np.ones(int(fit_mask.sum())),
        train_vol_anom[fit_mask].to_numpy(),
        train["trend_years"][fit_mask].to_numpy(),
    ])
    y = train_price_anom[fit_mask].to_numpy()
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta, gamma = coeffs

    years_trained = (
        (train_end - train["date"].min()).days / 365.25 if train_start is None
        else (train_end - train_start).days / 365.25
    )

    return {
        "baseline_price": baseline_price,
        "baseline_log_price": baseline_log_price,
        "baseline_volume": baseline_volume,
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "vol_anom_min": float(train_vol_anom[fit_mask].min()),
        "vol_anom_max": float(train_vol_anom[fit_mask].max()),
        "trend_years_max": float(years_trained),
        "train_end": train_end,
    }


def predict_level_volume_trend(monthly: pd.DataFrame, params: dict) -> pd.Series:
    """
    Trend-augmented sibling of predict_level_volume. trend_years is
    computed from params["train_end"] (so this function, like
    predict_level_volume, only ever needs (monthly, params)) and clamped
    to [-params["trend_years_max"], params["trend_years_max"]] -- the trend cannot be extrapolated
    beyond the training window span in either direction, mirroring the
    existing vol_anom_min/max clamp exactly.
    """
    monthly = monthly.copy()
    monthly["month"] = monthly["date"].dt.month
    monthly["trend_years"] = (monthly["date"] - params["train_end"]).dt.days / 365.25
    monthly["trend_years"] = monthly["trend_years"].clip(lower=-params["trend_years_max"], upper=params["trend_years_max"])

    baseline_volume_mapped = monthly["month"].map(params["baseline_volume"])
    vol_anom = (monthly["landings_kg"] - baseline_volume_mapped) / baseline_volume_mapped
    vol_anom = vol_anom.clip(lower=params["vol_anom_min"], upper=params["vol_anom_max"])

    predicted_log = (
        monthly["month"].map(params["baseline_log_price"])
        + params["alpha"]
        + params["beta"] * vol_anom
        + params["gamma"] * monthly["trend_years"]
    )
    predicted = np.exp(predicted_log)
    predicted.index = monthly.index
    return predicted


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
    months get NaN price/landings_kg, matching the research's build_monthly
    helper, so fit/predict functions never need to handle irregular gaps.
    """
    if rows.empty:
        raise ValueError("cannot build a monthly series from zero rows")

    rows = rows.sort_values("date")
    full_range = pd.date_range(rows["date"].min(), rows["date"].max(), freq="MS")
    monthly = rows.set_index("date").reindex(full_range)
    monthly.index.name = "date"
    return monthly.reset_index()
