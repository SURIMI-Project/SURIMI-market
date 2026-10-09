"""
Neighbor-informed port price model: picks between a volume-aware (Stacked)
and volume-agnostic (DistanceOnly) sub-model each tick, based on how far
this tick's landings are from what's normal for the port (the same
dynamic switch Phase B unified into stacked_model.should_use_distance_only).
Composes the two sub-models plus a reference to the SAME VolumeBaselineTracker
LocalMarket's base model already maintains for this key -- not a separate,
Step-2-only baseline.
"""
import pandas as pd

from .stacked_model import should_use_distance_only
from .StackedModel import StackedModel
from .DistanceOnlyModel import DistanceOnlyModel
from .VolumeBaselineTracker import VolumeBaselineTracker


class NeighborInformedModel:
    def __init__(
        self, stacked: StackedModel, distance_only: DistanceOnlyModel,
        volume_baseline: VolumeBaselineTracker | None,
    ):
        self.stacked = stacked
        self.distance_only = distance_only
        self.volume_baseline = volume_baseline

    def predict(
        self, date: pd.Timestamp, volume: float, anchor_value: float, neighbor_value: float
    ) -> float | None:
        volume_ratio = self.volume_baseline.ratio(date, volume) if self.volume_baseline is not None else float("nan")
        if should_use_distance_only(volume_ratio):
            return self.distance_only.predict(date, anchor_value, neighbor_value)
        return self.stacked.predict(date, volume, anchor_value, neighbor_value)
