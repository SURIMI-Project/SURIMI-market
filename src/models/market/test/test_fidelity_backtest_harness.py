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


def _write_scenario_with_anomalous_held_out_price(tmp_path, as_of_date, scenario_id="scn"):
    """Pre-as_of_date training data is a stable, learnable flat price
    (2.0 EUR/kg, constant volume) -- a model fit on it has every reason to
    predict ~2.0 for any future month. The held-out window from as_of_date
    onward has a deliberately large, unpredictable real price (50.0 EUR/kg)
    injected instead, for the same single port/species pair, with nothing in
    the training data that could have hinted at it.

    This is the proof-of-fix regression test for the real-price-first data
    leakage bug: run_backtest_split used to hand MarketHandler the FULL
    prices_df (train and held-out window alike), so init_simulation's
    latest_date = prices_df["date"].max() resolved to the file's true end
    date regardless of as_of_date, and load_price_time_series loaded real
    recorded prices for every date in the file -- including the "held out"
    window. Real-price-first (spec section 8) then found and returned the
    real recorded price for every held-out tick directly, bypassing Step 1/2
    prediction entirely and producing a perfect, meaningless 0.0 MAPE
    (comparing the real price against itself). With the fix,
    MarketHandler only ever sees the pre-as_of_date truncated view, so it
    has no way to know about the 50.0 anomaly and must genuinely predict
    from the flat 2.0 baseline -- which should score a large MAPE here.
    """
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

    # 7 years of flat, stable pre-as_of_date training data, followed by a
    # year of deliberately anomalous held-out data (test_horizon_years=1).
    dates = pd.date_range(as_of_date - pd.DateOffset(years=7), as_of_date + pd.DateOffset(months=11), freq="MS")
    price = [2.0 if d < as_of_date else 50.0 for d in dates]
    prices_df = pd.DataFrame({
        "date": dates,
        "market_code": "ESALC",
        "species_code": "ANE",
        "category_code": "Fresh - Whole",
        "landings_kg": 500.0,
        "price": price,
        "currency": "EUR",
        "measurement_unit": "kg",
    })
    prices_df.to_csv(scenario_dir / "prices.csv", index=False)

    return tmp_path


def test_run_backtest_split_scores_genuine_out_of_sample_error_not_an_echo(tmp_path):
    """Proof-of-fix for the real-price-first leakage bug -- see
    _write_scenario_with_anomalous_held_out_price's docstring for the full
    mechanism. A near-zero MAPE here would mean the harness is comparing the
    real held-out price against itself again."""
    as_of_date = pd.Timestamp("2022-01-01")
    root_config_dir = _write_scenario_with_anomalous_held_out_price(tmp_path, as_of_date)

    results_df = run_backtest_split(
        root_config_dir=str(root_config_dir),
        scenario_id="scn",
        as_of_date=as_of_date,
        test_horizon_years=1,
        species_codes=["ANE"],
    )

    assert len(results_df) == 1, f"expected exactly one (species, port) comparison row, got:\n{results_df}"
    row = results_df.iloc[0]
    assert row["n_test_months"] == 12
    assert row["mape"] > 50.0, (
        f"expected a LARGE MAPE: a model fit only on flat 2.0 EUR/kg pre-as_of_date data has no "
        f"way to predict the deliberately injected 50.0 EUR/kg held-out price -- got "
        f"{row['mape']}. A small MAPE here means the harness is leaking the held-out window's "
        f"real prices into MarketHandler at init time (the bug this test guards against)."
    )
