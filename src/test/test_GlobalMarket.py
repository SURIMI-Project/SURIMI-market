from itertools import product
import collections # Counter

import pytest

from src.GlobalMarket import GlobalMarket
from src.LocalMarket import LocalMarket



@pytest.fixture
def local_markets():
    aaa = LocalMarket("aaa")
    aaa.local_prices = {
        ("species_a", "gear_code_a"): 5.0,
        ("species_a", "gear_code_b"): 2,
        ("species_b", "gear_code_a"): 0.0,
        ("species_b", "gear_code_b"): 0.0
    }

    bbb = LocalMarket("bbb")
    bbb.local_prices = {
        ("species_a", "gear_code_a"): 7.5,
        ("species_a", "gear_code_b"): 3,
        ("species_b", "gear_code_a"): 3.2,
        ("species_b", "gear_code_b"): 0.0
    }

    ccc = LocalMarket("ccc")
    ccc.local_prices = {
        ("species_a", "gear_code_a"): 12.8,
        ("species_a", "gear_code_b"): 5,
        ("species_b", "gear_code_a"): 0.1,
        ("species_b", "gear_code_b"): 0.0
    }

    return {
        "AAA": aaa,
        "BBB": bbb,
        "CCC": ccc
    }


@pytest.fixture
def species_codes():
    """
    This emulates the species codes from the contract.
    """
    return ["species_a",
            "species_b",
            "species_c"]


@pytest.fixture
def gear_codes():
    """
    This emulates the gear codes from the contract.
    """
    return ["gear_code_a",
            "gear_code_b",
            "gear_code_c"]

# TODO Test negative values
def test_compute_mean_global_prices_from_local_markets(
        local_markets,
        species_codes,
        gear_codes
):
    """
    (species_a, gear_code_a) tests floating point precision.
    (species_a, gear_code_b) tests whether integers are converted to floats.
    (species_b, gear_code_a) tests whether a price of 0 is included in computing
                             the mean.
    (species_b, gear_code_b) tests when all prices are 0

    Anything with species_c or gear_code c must be None.
    """
    global_prices, _ = GlobalMarket().compute_prices(local_markets,
                                                     species_codes,
                                                     gear_codes)

    # Test whether the Cartesian product of all species and gear_codes are
    # present.
    pairs = product(species_codes, gear_codes)
    assert collections.Counter(pairs) == collections.Counter(global_prices.keys())

    assert global_prices[("species_a", "gear_code_a")] == pytest.approx(8.43, 0.1)
    assert global_prices[("species_a", "gear_code_b")] == pytest.approx(3.33, 0.1)
    assert global_prices[("species_b", "gear_code_a")] == pytest.approx(1.65, 0.1)
    assert global_prices[("species_b", "gear_code_b")] is None

    # Test that the global price does not use local prices that dont exist in the means computation.

