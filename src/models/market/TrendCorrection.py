"""
Multiplicative trend correction, composed into LevelOnlyModel/LevelVolumeModel.
The math here is transcribed unchanged from level_volume_model.predict_trend_correction
-- see that module's docstring for the full reasoning behind the block-based
rate measurement and the asymmetric [0, years_trained] clamp.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrendCorrection:
    confidence: float
    rate: float
    years_trained: float
    train_end: pd.Timestamp

    @classmethod
    def identity(cls, train_end: pd.Timestamp) -> "TrendCorrection":
        """A trend correction that's always a 1.0 no-op multiplier --
        LocalMarket's own-history models never fit a trend (see
        ARCHITECTURE.md section 6's deferred validation item); this lets
        them share LevelVolumeModel/LevelOnlyModel with CombinedMarket
        without a having-trend-or-not branch anywhere."""
        return cls(confidence=0.0, rate=0.0, years_trained=0.0, train_end=train_end)

    def factor(self, date: pd.Timestamp) -> float:
        trend_years = (date - self.train_end).days / 365.25
        trend_years = min(max(trend_years, 0.0), self.years_trained)
        return float(np.exp(self.confidence * self.rate * trend_years))

    def factor_series(self, dates: pd.Series) -> pd.Series:
        trend_years = (dates - self.train_end).dt.days / 365.25
        trend_years = trend_years.clip(lower=0.0, upper=self.years_trained)
        correction = np.exp(self.confidence * self.rate * trend_years)
        correction.index = dates.index
        return correction
