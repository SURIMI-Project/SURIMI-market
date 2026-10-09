import math

import pandas as pd
import pytest

from src.models.market.utility import haversine_distance_m, build_distance_lookup


def test_distance_to_self_is_zero():
    assert haversine_distance_m(2.0, 41.0, 2.0, 41.0) == pytest.approx(0.0, abs=1e-6)


def test_distance_is_symmetric():
    d1 = haversine_distance_m(-0.5, 38.3, 2.2, 41.4)
    d2 = haversine_distance_m(2.2, 41.4, -0.5, 38.3)
    assert d1 == pytest.approx(d2, rel=1e-12)


def test_distance_quarter_great_circle():
    # Equator to North Pole is a quarter of the great circle: pi/2 * R.
    expected = (math.pi / 2) * 6_378_137
    actual = haversine_distance_m(0.0, 0.0, 0.0, 90.0)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_distance_one_degree_longitude_at_equator():
    # At the equator, 1 degree of longitude is R * (pi/180) meters.
    expected = 6_378_137 * (math.pi / 180)
    actual = haversine_distance_m(0.0, 0.0, 1.0, 0.0)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_build_distance_lookup_is_symmetric_and_excludes_self_pairs():
    markets_df = pd.DataFrame({
        "market_code": ["ESALC", "ESAQA", "ESARN"],
        "lon": [-0.501706, -0.051994, 2.563677],
        "lat": [38.332112, 38.590245, 41.579699],
    })

    lookup = build_distance_lookup(markets_df)

    assert ("ESALC", "ESAQA") in lookup
    assert ("ESAQA", "ESALC") in lookup
    assert lookup[("ESALC", "ESAQA")] == pytest.approx(lookup[("ESAQA", "ESALC")])
    assert ("ESALC", "ESALC") not in lookup


def test_build_distance_lookup_excludes_rows_with_missing_coordinates():
    markets_df = pd.DataFrame({
        "market_code": ["ES", "ESALC", "ESAQA"],
        "lon": [None, -0.501706, -0.051994],
        "lat": [None, 38.332112, 38.590245],
    })

    lookup = build_distance_lookup(markets_df)

    assert ("ES", "ESALC") not in lookup
    assert ("ESALC", "ES") not in lookup
    assert ("ESALC", "ESAQA") in lookup
