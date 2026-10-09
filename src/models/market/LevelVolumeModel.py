"""
Seasonal baseline + volume-response model -- see level_volume_model.py's
module docstring for the full math. Wraps fit_level_volume_model/predict_level_volume,
which stay the free functions used during fit-time window-selection scoring;
this class is the frozen, single, final model produced once a window is chosen.
"""
import pandas as pd

from .PriceModel import PriceModel
from .TrendCorrection import TrendCorrection
from .level_volume_model import fit_level_volume_model, predict_level_volume


class LevelVolumeModel(PriceModel):
    def __init__(
        self,
        baseline_price: pd.Series,
        baseline_log_price: pd.Series,
        baseline_volume: pd.Series,
        alpha: float,
        beta: float,
        vol_anom_min: float,
        vol_anom_max: float,
        trend: TrendCorrection,
    ):
        self.baseline_price = baseline_price
        self.baseline_log_price = baseline_log_price
        self.baseline_volume = baseline_volume
        self.alpha = alpha
        self.beta = beta
        self.vol_anom_min = vol_anom_min
        self.vol_anom_max = vol_anom_max
        self.trend = trend

    @classmethod
    def fit(
        cls, monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp,
        trend: TrendCorrection,
    ) -> "LevelVolumeModel":
        params = fit_level_volume_model(monthly, train_start, train_end)
        return cls(
            baseline_price=params["baseline_price"],
            baseline_log_price=params["baseline_log_price"],
            baseline_volume=params["baseline_volume"],
            alpha=params["alpha"],
            beta=params["beta"],
            vol_anom_min=params["vol_anom_min"],
            vol_anom_max=params["vol_anom_max"],
            trend=trend,
        )

    def predict(self, date: pd.Timestamp, volume: float) -> float | None:
        monthly = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
        predicted = predict_level_volume(monthly, self._as_params()) * self.trend.factor(date)
        value = predicted.iloc[0]
        return float(value) if pd.notna(value) else None

    def predict_series(self, monthly: pd.DataFrame) -> pd.Series:
        predicted = predict_level_volume(monthly, self._as_params())
        return predicted * self.trend.factor_series(monthly["date"])

    def update_baseline_volume(self, month: int, new_mean: float) -> None:
        self.baseline_volume[month] = new_mean

    def _as_params(self) -> dict:
        return {
            "baseline_price": self.baseline_price,
            "baseline_log_price": self.baseline_log_price,
            "baseline_volume": self.baseline_volume,
            "alpha": self.alpha,
            "beta": self.beta,
            "vol_anom_min": self.vol_anom_min,
            "vol_anom_max": self.vol_anom_max,
        }
