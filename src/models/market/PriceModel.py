"""
Interface every frozen per-key price model implements -- CombinedMarket's
and LocalMarket's own-history models (LevelOnlyModel, LevelVolumeModel)
alike. Replaces dict-key-presence dispatch (params.get("baseline_volume")
is None) with real polymorphism.
"""
from abc import ABC, abstractmethod

import pandas as pd


class PriceModel(ABC):
    @abstractmethod
    def predict(self, date: pd.Timestamp, volume: float) -> float | None:
        """Predicted price for one (date, volume) pair, trend-corrected."""

    @abstractmethod
    def predict_series(self, monthly: pd.DataFrame) -> pd.Series:
        """Predicted price for a whole monthly series, trend-corrected --
        used by analysis-only callers (e.g. CombinedMarket.get_live_anchor_series)
        that need every row at once. Fit-time rolling-origin scoring calls the
        underlying free functions directly instead, since no frozen model
        exists yet at that point."""

    def update_baseline_volume(self, month: int, new_mean: float) -> None:
        """No-op by default; LevelVolumeModel overrides this to patch its
        frozen baseline_volume in place as real landings data arrives.
        LevelOnlyModel has no volume term, so inherits this no-op."""
