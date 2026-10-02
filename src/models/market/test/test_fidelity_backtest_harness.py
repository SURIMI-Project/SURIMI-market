"""
Regression test for the fidelity backtest harness itself (Task 19 fix):
real config/northwestern_med/prices.csv has no `size` column at all (unlike
MarketHandler.init_simulation's own internal read, which already defends
against this), so run_backtest_split must tolerate that shape rather than
raising KeyError when it builds `blended`.
"""
import pandas as pd

from src.models.market.test.fidelity_backtest_harness import run_backtest_split


def _write_scenario_with_no_size_column(tmp_path, scenario_id="scn"):
    """Mirrors the real config/northwestern_med/prices.csv's actual current
    shape: no `size` column present at all (not even an empty one)."""
    scenario_dir = tmp_path / scenario_id
    scenario_dir.mkdir()

    pd.DataFrame({
        "market_code": ["ES", "ESALC"],
        "market_name": ["Spain", "Alicante"],
        "lon": [None, -0.5],
        "lat": [None, 38.3],
    }).to_csv(scenario_dir / "markets.csv", index=False)

    pd.DataFrame({"species_code": ["ANE"], "species_name": ["Anchovy"]}).to_csv(
        scenario_dir / "species.csv", index=False
    )

    dates = pd.date_range("2018-01-01", periods=72, freq="MS")
    prices_df = pd.DataFrame({
        "date": dates,
        "market_code": "ESALC",
        "species_code": "ANE",
        "category_code": "Fresh - Whole",
        "landings_kg": 500.0,
        "price": 2.0,
        "currency": "EUR",
        "measurement_unit": "kg",
    })
    assert "size" not in prices_df.columns  # the whole point of this fixture
    prices_df.to_csv(scenario_dir / "prices.csv", index=False)

    return tmp_path


def test_run_backtest_split_tolerates_prices_csv_with_no_size_column(tmp_path):
    root_config_dir = _write_scenario_with_no_size_column(tmp_path)

    results_df = run_backtest_split(
        root_config_dir=str(root_config_dir),
        scenario_id="scn",
        as_of_date=pd.Timestamp("2022-01-01"),
        test_horizon_years=1,
        species_codes=["ANE"],
    )

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) > 0, "expected at least one (species, port) comparison row"
    assert list(results_df.columns) == ["species_code", "market_code", "n_test_months", "mape"]
    assert (results_df["n_test_months"] > 0).all()


def test_run_backtest_split_can_also_return_the_stepped_handler(tmp_path):
    root_config_dir = _write_scenario_with_no_size_column(tmp_path)

    results_df, handler = run_backtest_split(
        root_config_dir=str(root_config_dir),
        scenario_id="scn",
        as_of_date=pd.Timestamp("2022-01-01"),
        test_horizon_years=1,
        species_codes=["ANE"],
        return_handler=True,
    )

    assert isinstance(results_df, pd.DataFrame)
    assert handler.market_system.tick.get_tick() == 12  # 1 year, stepped monthly
