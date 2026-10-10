"""
The frozen neighbor-informed port price model: `StackedModel`, log-space,
anchor + the port's own (relative, clipped) volume anomaly + a
distance-weighted neighbor anomaly. There used to be a dynamic
Stacked/DistanceOnly switch here (and a `NeighborInformedModel` wrapper
composing the two sub-models plus a captured `VolumeBaselineTracker`
reference purely to compute the switch's ratio) -- removed once the
switch itself was removed (see stacked_model.py's module docstring and
SIMPLIFICATION_PROPOSAL.md section 3.2.B/D): `StackedModel`'s own-volume
term is now relative and clipped exactly like the base model's, so the
instability the switch existed to work around is handled the same way
the base model already handles it, and there's nothing left for a second
sub-model or a switch to do.

`StackedModel` wraps stacked_model.py's fit_stacked_model/predict_stacked_model,
which stay the free functions used during fit-time window-selection
scoring; this class is the frozen, single, final model produced once a
window is chosen (now reused from LocalMarket's own base model -- see
LocalMarket.fit_neighbor_models -- rather than independently re-selected).
"""
import pandas as pd

from .stacked_model import predict_stacked_model
from .scalar_predict import predict_scalar


class StackedModel:
    def __init__(self, params: dict):
        self.delta = params["delta"]
        self.beta = params["beta"]
        self.gamma = params["gamma"]
        self.port_baseline_volume = params["port_baseline_volume"]
        self.neighbor_baseline = params["neighbor_baseline"]
        self.vol_anom_min = params["vol_anom_min"]
        self.vol_anom_max = params["vol_anom_max"]

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
