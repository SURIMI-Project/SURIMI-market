"""
Fallback price model with no volume term -- same seasonal baseline every
year, used when there isn't enough data/variance to fit LevelVolumeModel.
Wraps level_volume_model.fit_level_only_model/predict_level_only, which stay
the free functions used during fit-time window-selection scoring; this class
is the frozen, single, final model produced once a window is chosen.
"""
import pandas as pd

from .PriceModel import PriceModel
from .TrendCorrection import TrendCorrection
from .level_volume_model import fit_level_only_model, predict_level_only


class LevelOnlyModel(PriceModel):
    def __init__(self, baseline_price: pd.Series, alpha: float, trend: TrendCorrection):
        self.baseline_price = baseline_price
        self.alpha = alpha
        self.beta = 0.0
        self.baseline_volume = None
        self.trend = trend

    @classmethod
    def fit(
        cls, monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp,
        trend: TrendCorrection,
    ) -> "LevelOnlyModel":
        params = fit_level_only_model(monthly, train_start, train_end)
        return cls(baseline_price=params["baseline_price"], alpha=params["alpha"], trend=trend)

    def predict(self, date: pd.Timestamp, volume: float) -> float | None:
        monthly = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
        predicted = predict_level_only(monthly, self._as_params()) * self.trend.factor(date)
        value = predicted.iloc[0]
        return float(value) if pd.notna(value) else None

    def predict_series(self, monthly: pd.DataFrame) -> pd.Series:
        predicted = predict_level_only(monthly, self._as_params())
        return predicted * self.trend.factor_series(monthly["date"])

    def _as_params(self) -> dict:
        return {"baseline_price": self.baseline_price, "alpha": self.alpha}
