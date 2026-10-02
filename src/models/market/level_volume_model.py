"""
Seasonal baseline + volume-response model (spec section 3's "Model 2"):
price = baseline_price[month] + alpha + beta * (volume - baseline_volume[month]).

Used identically for CombinedMarket's Model 2 and LocalMarket's Step 1 --
the same fitting/prediction logic, just applied to combined-market-level
or per-port-level monthly data (spec section 3).

Direct port of the validated fit_level_volume_model/predict_level_volume
functions from the research, with column names adapted to the production
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

    train = monthly[monthly["date"] < train_end]
    if train_start is not None:
        train = train[train["date"] >= train_start]

    baseline_price = train.groupby("month")["price"].mean()
    baseline_volume = train.groupby("month")["landings_kg"].mean()

    train_price_anom = train["price"] - train["month"].map(baseline_price)
    train_vol_anom = train["landings_kg"] - train["month"].map(baseline_volume)
    fit_mask = train_price_anom.notna() & train_vol_anom.notna()

    if fit_mask.sum() < 2 or train_vol_anom[fit_mask].var() == 0:
        raise ValueError("not enough usable training rows to fit level-volume model")

    beta, alpha = np.polyfit(train_vol_anom[fit_mask], train_price_anom[fit_mask], 1)

    return {
        "baseline_price": baseline_price,
        "baseline_volume": baseline_volume,
        "alpha": float(alpha),
        "beta": float(beta),
    }


def predict_level_volume(monthly: pd.DataFrame, params: dict) -> pd.Series:
    monthly = monthly.copy()
    monthly["month"] = monthly["date"].dt.month

    vol_anom = monthly["landings_kg"] - monthly["month"].map(params["baseline_volume"])
    predicted = (
        monthly["month"].map(params["baseline_price"]) + params["alpha"] + params["beta"] * vol_anom
    )
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
