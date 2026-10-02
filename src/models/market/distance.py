"""
Hand-rolled haversine great-circle distance. Spec section 4: confirmed
empirically to match .cache/test/port_distances.csv (R's
geosphere::distHaversine() output) to 1e-10m precision using
EARTH_RADIUS_M = 6,378,137 (WGS84 equatorial radius) -- no new dependency
needed (pyproj/geopy/haversine were considered and rejected as unnecessary
weight for a single, stable, already-precisely-specified formula).
"""
from itertools import combinations

import numpy as np
import pandas as pd

from . import consts


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(consts.EARTH_RADIUS_M * c)


def build_distance_lookup(markets_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """
    Compute pairwise distances for every local market with valid lon/lat.
    Rows with missing coordinates (the 2-letter CombinedMarket codes, which
    have NA lon/lat in markets.csv) are excluded entirely -- they never
    participate in distance weighting (spec section 4).

    Returns a dict keyed by BOTH (a, b) and (b, a) for convenient
    undirected lookup, computed once and shared across every species.
    """
    valid = markets_df.dropna(subset=["lon", "lat"])

    lookup: dict[tuple[str, str], float] = {}
    for (_, row_a), (_, row_b) in combinations(valid.iterrows(), 2):
        dist = haversine_distance_m(row_a["lon"], row_a["lat"], row_b["lon"], row_b["lat"])
        lookup[(row_a["market_code"], row_b["market_code"])] = dist
        lookup[(row_b["market_code"], row_a["market_code"])] = dist

    return lookup
