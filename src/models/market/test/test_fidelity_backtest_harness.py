"""
Regression test for the fidelity backtest harness itself: real
config/northwestern_med/prices.csv has no `size` column at all, so
run_backtest_split must tolerate that shape rather than raising KeyError
when it builds `blended`.
"""
import numpy as np
import pandas as pd

from src.models.market.CombinedMarket import CombinedMarket
from src.models.market.test.fidelity_backtest_harness import run_backtest_split, _live_anchor_resolver_from


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
    """Builds a scenario with a stable flat 2.0 EUR/kg price before
    as_of_date and a deliberately anomalous 50.0 EUR/kg price in the
    held-out window, with nothing in training data to hint at it. Proves
    run_backtest_split scores genuine out-of-sample error rather than
    leaking the held-out window's real prices back into MarketHandler
    (which would score a false near-zero MAPE)."""
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

    # 7 years of flat training data, then a year of anomalous held-out data.
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


def _synthetic_monthly(n_years=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    volume = 1000.0 + rng.normal(0, 50, len(dates))
    price = 2.0 - 0.001 * (volume - 1000.0) + rng.normal(0, 0.01, len(dates))
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})


def _trending_synthetic_monthly(n_years=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_years * 12, freq="MS")
    volume = 1000.0 + rng.normal(0, 50, len(dates))
    years = dates.year.to_numpy(dtype=float)
    price = (
        2.0 - 0.001 * (volume - 1000.0)
        + 0.1 * (years - years.min())
        + rng.normal(0, 0.01, len(dates))
    )
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})


# _live_anchor_resolver_from is the relocated home of the now-deleted
# CombinedMarket.get_live_anchor_series (identical math, moved here since
# nothing in production ever used it -- see this module's own docstring
# on the function). These four tests moved from test_CombinedMarket.py
# along with it, adapted to call the resolver instead of the deleted
# method directly.

def test_live_anchor_resolver_varies_month_to_month_unlike_static_baseline():
    market = CombinedMarket("ES")
    key = ("ANE", "Fresh - Whole")
    own_monthly = _synthetic_monthly()
    market.fit({key: own_monthly}, latest_date=pd.Timestamp("2024-01-01"))

    resolver = _live_anchor_resolver_from({"ES": {key: own_monthly}})
    port_monthly = pd.DataFrame({"date": own_monthly["date"]})
    live_series = resolver(market, "ANE", "Fresh - Whole", port_monthly)

    assert live_series is not None
    assert len(live_series) == len(own_monthly)
    # Unlike the static table (one value per calendar month, repeating
    # every year), the live series reflects real volume at each date, so
    # two different Januaries should generally differ.
    by_year = live_series.groupby(live_series.index.year).first()
    assert by_year.nunique() > 1


def test_live_anchor_resolver_reindexes_to_target_dates_with_nan_outside_own_range():
    market = CombinedMarket("ES")
    key = ("ANE", "Fresh - Whole")
    own_monthly = _synthetic_monthly(n_years=2)  # 2018-01 .. 2019-12
    market.fit({key: own_monthly}, latest_date=pd.Timestamp("2020-01-01"))

    target_dates = pd.date_range("2019-06-01", periods=12, freq="MS")  # extends past own_monthly's range
    resolver = _live_anchor_resolver_from({"ES": {key: own_monthly}})
    port_monthly = pd.DataFrame({"date": target_dates})
    live_series = resolver(market, "ANE", "Fresh - Whole", port_monthly)

    assert len(live_series) == len(target_dates)
    assert pd.notna(live_series[pd.Timestamp("2019-06-01")])
    assert pd.isna(live_series[pd.Timestamp("2020-05-01")])  # past own_monthly's last date


def test_live_anchor_resolver_returns_none_when_no_frozen_model():
    market = CombinedMarket("ES")
    own_monthly = _synthetic_monthly()
    resolver = _live_anchor_resolver_from({"ES": {("GHOST", "Fresh - Whole"): own_monthly}})
    port_monthly = pd.DataFrame({"date": own_monthly["date"]})
    result = resolver(market, "GHOST", "Fresh - Whole", port_monthly)
    assert result is None


def test_live_anchor_resolver_unaffected_by_trend_term():
    # Analysis-only path -- confirm it still returns a result shaped the
    # same way for a trend-bearing key. predict_series does multiply by
    # self.trend.factor_series(...), which does read confidence/rate, but
    # that factor is an exact 1.0 no-op for every in-sample date here
    # (trend_years clips to 0 whenever date <= train_end), so the series
    # is unaffected by the trend term regardless of its fitted value.
    market = CombinedMarket("ES")
    key = ("ANE", "Fresh - Whole")
    monthly = _trending_synthetic_monthly()
    market.fit({key: monthly}, latest_date=pd.Timestamp("2025-01-01"))

    resolver = _live_anchor_resolver_from({"ES": {key: monthly}})
    port_monthly = pd.DataFrame({"date": monthly["date"]})
    live_series = resolver(market, "ANE", "Fresh - Whole", port_monthly)
    assert live_series is not None
    assert len(live_series) == len(monthly)


def test_run_backtest_split_scores_genuine_out_of_sample_error_not_an_echo(tmp_path):
    """Guards against the real-price-first data leakage bug: a near-zero
    MAPE here would mean the harness is comparing the real held-out price
    against itself instead of a genuine model prediction."""
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
