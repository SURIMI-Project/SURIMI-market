"""
High-tier neighbor determination, distance-weighted neighbor signal
construction, and the Stacked/DistanceOnly fit/predict functions.
"""
import numpy as np
import pandas as pd

from . import consts
from .level_volume_model import with_month_column


def compute_volume_shares(volumes_by_port: dict[str, float]) -> dict[str, float]:
    total = sum(volumes_by_port.values())
    if total <= 0:
        return {port: 0.0 for port in volumes_by_port}
    return {port: (volume / total) * 100 for port, volume in volumes_by_port.items()}


def compute_tier(volume_share_pct: float) -> str:
    """Matches pd.cut(bins=[-0.01, 0.2, 2, 100]) right-inclusive-lower-bin semantics exactly."""
    if volume_share_pct > consts.HIGH_TIER_MIN_VOLUME_SHARE_PCT:
        return "High"
    if volume_share_pct > consts.MID_TIER_MIN_VOLUME_SHARE_PCT:
        return "Mid"
    return "Low"


def compute_high_tier_ports(volumes_by_port: dict[str, float]) -> set[str]:
    shares = compute_volume_shares(volumes_by_port)
    return {port for port, share in shares.items() if compute_tier(share) == "High"}


def distance_weighted_neighbor_signal(
    target_port: str,
    source_ports: set[str],
    predicted_series_by_port: dict[str, pd.Series],
    distance_lookup: dict[tuple[str, str], float],
) -> pd.Series | None:
    """
    Inverse-distance-weighted average of sibling ports' Step-1 PREDICTED
    series, never their real recorded prices. Returns None if there are
    no valid source ports (no distance known, or the only candidate is
    the target itself).
    """
    candidates = [
        port for port in source_ports
        if port != target_port
        and port in predicted_series_by_port
        and (target_port, port) in distance_lookup
    ]
    if not candidates:
        return None

    weights = pd.Series({port: 1.0 / distance_lookup[(target_port, port)] for port in candidates})
    matrix = pd.DataFrame({port: predicted_series_by_port[port] for port in candidates})

    valid = matrix.notna()
    weighted_sum = (matrix.fillna(0) * weights).sum(axis=1)
    weight_total = (valid * weights).sum(axis=1)

    neighbor_signal = weighted_sum / weight_total
    neighbor_signal[weight_total == 0] = float("nan")
    return neighbor_signal


def build_static_anchor_series(dates, baseline_price_table: pd.Series) -> pd.Series:
    """Broadcast CombinedMarket's frozen per-month baseline across every
    date; the production default anchor scheme."""
    dates = pd.DatetimeIndex(dates)
    months = pd.Series(dates.month, index=dates)
    return months.map(baseline_price_table)


def fit_stacked_model(
    port_monthly: pd.DataFrame,
    anchor_series: pd.Series,
    neighbor_signal: pd.Series,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
) -> dict:
    port = with_month_column(port_monthly)
    port["anchor"] = port["date"].map(anchor_series)
    port["neighbor"] = port["date"].map(neighbor_signal)

    train = port[port["date"] < train_end]
    if train_start is not None:
        train = train[train["date"] >= train_start]

    port_baseline_volume = train.groupby("month")["landings_kg"].mean()
    neighbor_baseline = train.groupby("month")["neighbor"].mean()

    y = train["price"] - train["anchor"]
    x1 = train["landings_kg"] - train["month"].map(port_baseline_volume)
    x2 = train["neighbor"] - train["month"].map(neighbor_baseline)

    mask = (y.notna() & x1.notna() & x2.notna()).to_numpy()
    if mask.sum() < 4:
        raise ValueError("not enough usable training rows to fit stacked model")

    X = np.column_stack([np.ones(mask.sum()), x1.to_numpy()[mask], x2.to_numpy()[mask]])
    yv = y.to_numpy()[mask]
    if np.linalg.matrix_rank(X) < 3:
        raise ValueError("design matrix is rank-deficient")

    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    delta, beta, gamma = coef

    return {
        "delta": float(delta),
        "beta": float(beta),
        "gamma": float(gamma),
        "port_baseline_volume": port_baseline_volume,
        "neighbor_baseline": neighbor_baseline,
    }


def predict_stacked_model(
    port_monthly: pd.DataFrame, anchor_series: pd.Series, neighbor_signal: pd.Series, params: dict
) -> pd.Series:
    port = with_month_column(port_monthly)
    port["anchor"] = port["date"].map(anchor_series)
    port["neighbor"] = port["date"].map(neighbor_signal)

    vol_anom = port["landings_kg"] - port["month"].map(params["port_baseline_volume"])
    neighbor_anom = port["neighbor"] - port["month"].map(params["neighbor_baseline"])

    predicted = port["anchor"] + params["delta"] + params["beta"] * vol_anom + params["gamma"] * neighbor_anom
    predicted.index = port_monthly.index
    return predicted


def fit_distance_only_model(
    port_monthly: pd.DataFrame,
    anchor_series: pd.Series,
    neighbor_signal: pd.Series,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
) -> dict:
    port = with_month_column(port_monthly)
    port["anchor"] = port["date"].map(anchor_series)
    port["neighbor"] = port["date"].map(neighbor_signal)

    train = port[port["date"] < train_end]
    if train_start is not None:
        train = train[train["date"] >= train_start]

    neighbor_baseline = train.groupby("month")["neighbor"].mean()

    y = train["price"] - train["anchor"]
    x = train["neighbor"] - train["month"].map(neighbor_baseline)

    mask = (y.notna() & x.notna()).to_numpy()
    if mask.sum() < 3 or x.to_numpy()[mask].var() == 0:
        raise ValueError("not enough usable training rows to fit distance-only model")

    gamma, delta = np.polyfit(x.to_numpy()[mask], y.to_numpy()[mask], 1)

    return {"delta": float(delta), "gamma": float(gamma), "neighbor_baseline": neighbor_baseline}


def predict_distance_only_model(
    port_monthly: pd.DataFrame, anchor_series: pd.Series, neighbor_signal: pd.Series, params: dict
) -> pd.Series:
    port = with_month_column(port_monthly)
    port["anchor"] = port["date"].map(anchor_series)
    port["neighbor"] = port["date"].map(neighbor_signal)

    neighbor_anom = port["neighbor"] - port["month"].map(params["neighbor_baseline"])
    predicted = port["anchor"] + params["delta"] + params["gamma"] * neighbor_anom
    predicted.index = port_monthly.index
    return predicted
