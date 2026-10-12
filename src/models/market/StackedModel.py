"""
The frozen neighbor-informed port price model: `StackedModel`, log-space,
anchor + the port's own (relative, clipped) volume anomaly + a
distance-weighted neighbor anomaly. There used to be a dynamic
Stacked/DistanceOnly switch here (and a `NeighborInformedModel` wrapper
composing the two sub-models plus a captured `VolumeBaselineTracker`
reference purely to compute the switch's ratio) -- removed once the
switch itself was removed (see SIMPLIFICATION_PROPOSAL.md section
3.2.B/D): `StackedModel`'s own-volume term is now relative and clipped
exactly like the base model's, so the instability the switch existed to
work around is handled the same way the base model already handles it,
and there's nothing left for a second sub-model or a switch to do.

Also holds distance-weighted neighbor-signal construction (nearest-K
donor selection) and the model's own fit/predict free functions. These
used to live in a separate stacked_model.py, split from this class the
same way level_volume_model.py is split from price_models.py -- but
unlike that pair, fit_stacked_model/predict_stacked_model have no second
consumer that needs them independent of this class (no window-selection
scoring loop calls them -- LocalMarket.fit_neighbor_models reuses the
base model's own window instead, see ARCHITECTURE.md section 3.3), so
splitting them was leftover fragmentation rather than a load-bearing
separation. Merged here, with `.fit()` added so `StackedModel` matches
`LevelOnlyModel`/`LevelVolumeModel`'s own fit/predict pattern.
"""
import numpy as np
import pandas as pd

from . import consts
from .level_volume_model import with_month_column
from .scalar_predict import predict_scalar


def compute_inverse_distance_weights(
    target_port: str,
    source_ports: set[str],
    available_ports,
    distance_lookup: dict[tuple[str, str], float],
) -> dict[str, float]:
    """
    Inverse-distance weight (1/distance) for each of the target's
    TOP_K_NEAREST_DONORS geographically closest eligible ports -- replaces
    the old volume-share "High-tier" gate (simpler: no percentage-share
    math, no 3-way classification, just "closest K"). A port counts as a
    candidate for the distance ranking if it isn't the target itself and
    has a known distance to the target; THEN, after narrowing to the
    nearest K by distance alone, the availability filter (present in
    whatever the caller is weighting: a dict of per-tick scalar values, or
    a dict of historical Series -- `available_ports` just needs to support
    `in`) is applied -- this order matters and was the exact one
    empirically validated, not an equivalent-looking alternative. Returns
    an empty dict if nothing survives both filters; callers treat that the
    same way they already treat "no neighbor signal".
    """
    with_known_distance = [
        port for port in source_ports
        if port != target_port and (target_port, port) in distance_lookup
    ]
    nearest = sorted(
        with_known_distance, key=lambda port: distance_lookup[(target_port, port)]
    )[:consts.TOP_K_NEAREST_DONORS]
    candidates = [port for port in nearest if port in available_ports]
    return {port: 1.0 / distance_lookup[(target_port, port)] for port in candidates}


def distance_weighted_neighbor_signal(
    target_port: str,
    source_ports: set[str],
    predicted_series_by_port: dict[str, pd.Series],
    distance_lookup: dict[tuple[str, str], float],
) -> pd.Series | None:
    """
    Inverse-distance-weighted average of sibling ports' base-model PREDICTED
    series, never their real recorded prices. Returns None if there are
    no valid source ports (no distance known, or the only candidate is
    the target itself).
    """
    weight_dict = compute_inverse_distance_weights(
        target_port, source_ports, predicted_series_by_port, distance_lookup
    )
    if not weight_dict:
        return None

    weights = pd.Series(weight_dict)
    matrix = pd.DataFrame({port: predicted_series_by_port[port] for port in weight_dict})

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
    """
    Neighbor-informed price model: log(price) - log(anchor) = delta +
    beta*vol_anom + gamma*neighbor_anom. Mirrors
    level_volume_model.fit_level_volume_model's own formula shape exactly
    (log-space, vol_anom relative to the port's own baseline volume and
    clipped to the observed training range at predict time) -- this used
    to be a separate, inconsistent linear-space formula with an
    unclipped, absolute volume anomaly, which needed a whole extra
    Stacked/DistanceOnly switch to compensate for instability the base
    model already handles with a one-line clip. Being in log-space also
    means a prediction can never be non-positive, by construction --
    unlike the old linear-space formula, which needed a separate
    sanity-check/retry safety net this version doesn't need at all.
    """
    port = with_month_column(port_monthly)
    port["anchor"] = port["date"].map(anchor_series)
    port["neighbor"] = port["date"].map(neighbor_signal)

    train = port[port["date"] < train_end]
    if train_start is not None:
        train = train[train["date"] >= train_start]

    port_baseline_volume = train.groupby("month")["landings_kg"].mean()
    neighbor_baseline = train.groupby("month")["neighbor"].mean()

    baseline_volume_mapped = train["month"].map(port_baseline_volume)
    y = np.log(train["price"]) - np.log(train["anchor"])
    x1 = (train["landings_kg"] - baseline_volume_mapped) / baseline_volume_mapped
    x2 = train["neighbor"] - train["month"].map(neighbor_baseline)

    mask = (y.notna() & x1.notna() & x2.notna() & (baseline_volume_mapped > 0)).to_numpy()
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
        "vol_anom_min": float(x1.to_numpy()[mask].min()),
        "vol_anom_max": float(x1.to_numpy()[mask].max()),
    }


def predict_stacked_model(
    port_monthly: pd.DataFrame, anchor_series: pd.Series, neighbor_signal: pd.Series, params: dict
) -> pd.Series:
    port = with_month_column(port_monthly)
    port["anchor"] = port["date"].map(anchor_series)
    port["neighbor"] = port["date"].map(neighbor_signal)

    baseline_volume_mapped = port["month"].map(params["port_baseline_volume"])
    vol_anom = (port["landings_kg"] - baseline_volume_mapped) / baseline_volume_mapped
    vol_anom = vol_anom.clip(lower=params["vol_anom_min"], upper=params["vol_anom_max"])
    neighbor_anom = port["neighbor"] - port["month"].map(params["neighbor_baseline"])

    predicted_log = (
        np.log(port["anchor"]) + params["delta"] + params["beta"] * vol_anom + params["gamma"] * neighbor_anom
    )
    predicted = np.exp(predicted_log)
    predicted.index = port_monthly.index
    return predicted


class StackedModel:
    def __init__(self, params: dict):
        self.delta = params["delta"]
        self.beta = params["beta"]
        self.gamma = params["gamma"]
        self.port_baseline_volume = params["port_baseline_volume"]
        self.neighbor_baseline = params["neighbor_baseline"]
        self.vol_anom_min = params["vol_anom_min"]
        self.vol_anom_max = params["vol_anom_max"]

    @classmethod
    def fit(
        cls,
        port_monthly: pd.DataFrame,
        anchor_series: pd.Series,
        neighbor_signal: pd.Series,
        train_start: pd.Timestamp | None,
        train_end: pd.Timestamp,
    ) -> "StackedModel":
        params = fit_stacked_model(port_monthly, anchor_series, neighbor_signal, train_start, train_end)
        return cls(params)

    def predict(self, date: pd.Timestamp, volume: float, anchor_value: float, neighbor_value: float) -> float | None:
        return predict_scalar(
            predict_stacked_model, date, volume, self._as_params(), anchor_value, neighbor_value
        )

    def _as_params(self) -> dict:
        return {
            "delta": self.delta,
            "beta": self.beta,
            "gamma": self.gamma,
            "port_baseline_volume": self.port_baseline_volume,
            "neighbor_baseline": self.neighbor_baseline,
            "vol_anom_min": self.vol_anom_min,
            "vol_anom_max": self.vol_anom_max,
        }
