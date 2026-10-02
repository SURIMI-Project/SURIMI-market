import datetime
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.market.MarketHandler import MarketHandler


class _FakeMarket:
    def __init__(self, market_code):
        self.market_code = market_code


class _FakeSpecies:
    def __init__(self, species_code):
        self.species_code = species_code


class _FakePriceCategory:
    def __init__(self, category_code):
        self.category_code = category_code


class _FakeItems:
    def __init__(self, markets, species, price_categories):
        self.markets = markets
        self.species = species
        self.price_categories = price_categories


class _FakeSimulation:
    def __init__(self, items):
        self.items = items


def _write_two_port_config(tmp_path, scenario_id="scn"):
    scenario_dir = tmp_path / scenario_id
    scenario_dir.mkdir()

    pd.DataFrame({
        "market_code": ["ES", "ESALC", "ESBCN"],
        "market_name": ["Spain", "Alicante", "Barcelona"],
        "lon": [None, -0.5, 2.2],
        "lat": [None, 38.3, 41.4],
    }).to_csv(scenario_dir / "markets.csv", index=False)

    pd.DataFrame({
        "species_code": ["ANE", "THIN"], "species_name": ["Anchovy", "Rare species"],
    }).to_csv(scenario_dir / "species.csv", index=False)

    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    rows = []
    for port, base_volume in [("ESALC", 5000.0), ("ESBCN", 5200.0)]:
        volume = base_volume + rng.normal(0, 200, len(dates))
        price = 2.0 - 0.0005 * (volume - base_volume) + rng.normal(0, 0.01, len(dates))
        for d, p, v in zip(dates, price, volume):
            rows.append({
                "date": d, "market_code": port, "species_code": "ANE", "category_code": "Fresh - Whole",
                "landings_kg": v, "price": p, "currency": "EUR", "measurement_unit": "kg", "size": None,
            })
    # THIN: zero price history anywhere -- no frozen model can exist for it at
    # any port (section 9 point 4: silently excluded, not an error). Note a
    # single stray data point is NOT enough to exercise this path: per section
    # 9 point 2, any history at all -- however sparse -- is enough for the
    # fallback cascade (level-offset-only fit, then CombinedMarket-only) to
    # still produce a degraded-but-present price; only genuinely zero rows
    # anywhere leave no model to fall back to.

    pd.DataFrame(rows).to_csv(scenario_dir / "prices.csv", index=False)
    return tmp_path


def _fake_simulation():
    return _FakeSimulation(_FakeItems(
        markets=[_FakeMarket("ESALC"), _FakeMarket("ESBCN")],
        species=[_FakeSpecies("ANE"), _FakeSpecies("THIN")],
        price_categories=[_FakePriceCategory("Fresh - Whole")],
    ))


def test_full_flow_handles_zero_volume_tick_gracefully():
    """A tick with genuinely zero landings must still produce a price (via the
    DistanceOnly fallback, VolumeRatio=0 < 10%), never a crash or a None
    silently treated as valid (spec section 9 point 1)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_config_dir = _write_two_port_config(Path(tmp_dir))
        handler = MarketHandler("sim-1", "scn", datetime.datetime(2024, 1, 1), 1)
        response, err = handler.init_simulation(_fake_simulation(), str(root_config_dir))
        assert err is None

        # No catches reported this tick at all -> zero volume for every key.
        step_response, step_err = handler.simulate_step(datetime.datetime(2024, 1, 1))
        assert step_err is None

        prices = handler.get_species_prices()
        ane_prices = [
            sp for sp in prices.species_price_summary.species_prices if sp.species.species_code == "ANE"
        ]
        assert len(ane_prices) > 0
        for sp in ane_prices:
            assert sp.price is not None


def test_full_flow_excludes_species_with_insufficient_history_without_crashing():
    """THIN has only 1 month of data anywhere -- no CombinedMarket/LocalMarket
    model can be fit for it. get_species_prices must simply omit it, not error
    (spec section 9 point 4)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_config_dir = _write_two_port_config(Path(tmp_dir))
        handler = MarketHandler("sim-1", "scn", datetime.datetime(2024, 1, 1), 1)
        response, err = handler.init_simulation(_fake_simulation(), str(root_config_dir))
        assert err is None

        handler.simulate_step(datetime.datetime(2024, 1, 1))
        prices = handler.get_species_prices()

        thin_prices = [
            sp for sp in prices.species_price_summary.species_prices if sp.species.species_code == "THIN"
        ]
        assert thin_prices == []


def test_full_flow_real_price_overrides_prediction_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_config_dir = _write_two_port_config(Path(tmp_dir))
        handler = MarketHandler("sim-1", "scn", datetime.datetime(2024, 1, 1), 1)
        handler.init_simulation(_fake_simulation(), str(root_config_dir))

        handler.market_system.local_markets["ESALC"].price_time_series[
            ("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"))
        ] = 123.45

        handler.simulate_step(datetime.datetime(2024, 1, 1))
        prices = handler.get_species_prices()

        matching = [
            sp for sp in prices.species_price_summary.species_prices
            if sp.species.species_code == "ANE" and sp.market_code == "ESALC"
        ]
        assert len(matching) == 1
        assert matching[0].price == pytest.approx(123.45)
