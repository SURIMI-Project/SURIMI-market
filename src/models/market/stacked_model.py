"""
High-tier neighbor determination and distance-weighted neighbor signal
construction (spec section 4's species-keying split, section 9.3). Step 2
Stacked/DistanceOnly fit/predict functions are added in Task 8.
"""
import pandas as pd

from . import consts


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
    series (never their real recorded prices -- plan header decision 7).
    Returns None if there are no valid source ports (no distance known,
    or the only candidate is the target itself).
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
