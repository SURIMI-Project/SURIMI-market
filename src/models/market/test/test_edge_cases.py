import datetime
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.market.MarketHandler import MarketHandler
from src.models.market.test.fake_simulation import (
    FakeMarket as _FakeMarket,
    FakeSpecies as _FakeSpecies,
    FakePriceCategory as _FakePriceCategory,
    FakeItems as _FakeItems,
    FakeSimulation as _FakeSimulation,
)


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
    # THIN has zero price rows anywhere, so no model can be fit at any port and
    # it is silently excluded. Any history at all, however sparse, is enough
    # for the fallback cascade to still produce a degraded-but-present price.

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
    DistanceOnly fallback, VolumeRatio=0 < 10%), never a crash or a silently
    accepted None."""
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
    """THIN has zero rows anywhere, so no CombinedMarket/LocalMarket model can
    be attempted for it; get_species_prices must omit it, not error."""
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
