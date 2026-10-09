"""
Volume + distance-weighted-neighbor-signal price model -- see
stacked_model.py's fit_stacked_model/predict_stacked_model for the actual
math, which stay the free functions used during fit-time window-selection
scoring; this class is the frozen, single, final model produced once a
window is chosen.
"""
import pandas as pd

from .stacked_model import predict_stacked_model


class StackedModel:
    def __init__(self, params: dict):
        self.delta = params["delta"]
        self.beta = params["beta"]
        self.gamma = params["gamma"]
        self.port_baseline_volume = params["port_baseline_volume"]
        self.neighbor_baseline = params["neighbor_baseline"]

    def predict(self, date: pd.Timestamp, volume: float, anchor_value: float, neighbor_value: float) -> float | None:
        single_row = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
        anchor_series = pd.Series([anchor_value], index=[date])
        neighbor_signal = pd.Series([neighbor_value], index=[date])
        predicted = predict_stacked_model(single_row, anchor_series, neighbor_signal, self._as_params())
        value = predicted.iloc[0]
        return float(value) if pd.notna(value) else None

    def _as_params(self) -> dict:
        return {
            "delta": self.delta,
            "beta": self.beta,
            "gamma": self.gamma,
            "port_baseline_volume": self.port_baseline_volume,
            "neighbor_baseline": self.neighbor_baseline,
        }
