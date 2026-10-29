"""
The frozen base-model hierarchy: an interface (`PriceModel`) and its two
implementations (`LevelOnlyModel`, the no-volume-term fallback;
`LevelVolumeModel`, the seasonal-baseline + volume-response model), plus
the trend-correction term they both compose (`TrendCorrection`). Kept in
one file -- deliberately diverging from the usual one-class-per-file
convention -- because these are always read together, always changed
together (they landed as a single atomic phase), and never substituted
independently outside this cluster: tracing "how is one frozen price
model evaluated" used to mean opening four separate files for ~180 lines
of code.

`LevelOnlyModel`/`LevelVolumeModel` wrap level_volume_model.py's
fit_level_only_model/predict_level_only and
fit_level_volume_model/predict_level_volume, which stay the free
functions used during fit-time window-selection scoring; each class here
is the frozen, single, final model produced once a window is chosen.
`TrendCorrection`'s math is transcribed unchanged from
level_volume_model.predict_trend_correction -- see that module's
docstring for the full reasoning behind the block-based rate measurement
and the asymmetric [0, years_trained] clamp.

`from __future__ import annotations` is required here: LevelOnlyModel and
LevelVolumeModel reference TrendCorrection in their signatures, but
TrendCorrection is defined last in this file (preserving the cluster's
mandated PriceModel/LevelOnlyModel/LevelVolumeModel/TrendCorrection
order) -- without it, those forward references would raise a NameError
at class-definition time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .level_volume_model import (
    fit_level_only_model,
    predict_level_only,
    fit_level_volume_model,
    predict_level_volume,
)
from .scalar_predict import predict_scalar


class PriceModel(ABC):
    @abstractmethod
    def predict(self, date: pd.Timestamp, volume: float) -> float | None:
        """Predicted price for one (date, volume) pair, trend-corrected."""

    @abstractmethod
    def predict_series(self, monthly: pd.DataFrame) -> pd.Series:
        """Predicted price for a whole monthly series, trend-corrected --
        used by analysis-only callers (e.g. the fidelity backtest harness's
        live-anchor resolver, test/fidelity_backtest_harness.py's
        _live_anchor_resolver_from) that need every row at once. Fit-time
        rolling-origin scoring calls the underlying free functions directly
        instead, since no frozen model exists yet at that point."""

    def update_baseline_volume(self, month: int, new_mean: float) -> None:
        """No-op by default; LevelVolumeModel overrides this to patch its
        frozen baseline_volume in place as real landings data arrives.
        LevelOnlyModel has no volume term, so inherits this no-op."""


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
        return predict_scalar(
            predict_level_only, date, volume, self._as_params(), trend_factor=self.trend.factor(date)
        )

    def predict_series(self, monthly: pd.DataFrame) -> pd.Series:
        predicted = predict_level_only(monthly, self._as_params())
        return predicted * self.trend.factor_series(monthly["date"])

    def _as_params(self) -> dict:
        return {"baseline_price": self.baseline_price, "alpha": self.alpha}


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
        return predict_scalar(
            predict_level_volume, date, volume, self._as_params(), trend_factor=self.trend.factor(date)
        )

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
