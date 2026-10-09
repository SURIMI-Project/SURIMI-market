import logging

import numpy as np
import pandas as pd
import pytest

from src.models.market.CombinedMarket import CombinedMarket


def _synthetic_monthly(n_years=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    volume = 1000.0 + rng.normal(0, 50, len(dates))
    price = 2.0 - 0.001 * (volume - 1000.0) + rng.normal(0, 0.01, len(dates))
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})


def test_fit_selects_and_freezes_a_window_per_key():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}

    selected = market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    assert selected[("ANE", "Fresh - Whole")] is not None
    assert ("ANE", "Fresh - Whole") in market.models


def test_get_price_uses_real_data_before_frozen_model(caplog):
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    market.load_price_time_series(pd.DataFrame({
        "date": pd.to_datetime(["2024-02-01"]),
        "market_code": ["ES"],
        "species_code": ["ANE"],
        "category_code": ["Fresh - Whole"],
        "price": [9.99],
        "size": [None],
    }))

    price = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=1000.0)
    assert price == 9.99


def test_get_price_falls_through_to_frozen_model_when_real_price_is_nan():
    # size is NaN here but price can still be NaN too; get_price must not
    # treat a stored NaN as real data and must fall through to the frozen model.
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    market.load_price_time_series(pd.DataFrame({
        "date": pd.to_datetime(["2024-02-01"]),
        "market_code": ["ES"],
        "species_code": ["ANE"],
        "category_code": ["Fresh - Whole"],
        "price": [float("nan")],
        "size": [None],
    }))

    price = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=1000.0)
    assert price is not None
    assert not pd.isna(price)
    assert price == pytest.approx(2.0, abs=0.5)


def test_get_price_falls_back_to_frozen_model_when_no_real_price():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    price = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2024-03-01"), volume=1000.0)
    assert price is not None
    assert price == pytest.approx(2.0, abs=0.5)


def test_load_price_time_series_excludes_per_size_rows():
    market = CombinedMarket("ES")
    market.load_price_time_series(pd.DataFrame({
        "date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
        "market_code": ["ES", "ES"],
        "species_code": ["ANE", "ANE"],
        "category_code": ["Fresh - Whole", "Fresh - Whole"],
        "price": [2.0, 99.0],
        "size": [None, "1"],
    }))

    assert market.real_prices.get("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01")) == 2.0


def test_get_price_returns_none_for_unknown_species():
    market = CombinedMarket("ES")
    price = market.get_price("ZZZ", "Fresh - Whole", pd.Timestamp("2024-03-01"), volume=1000.0)
    assert price is None


def test_get_baseline_price_table_returns_frozen_static_table():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    table = market.get_baseline_price_table("ANE", "Fresh - Whole")
    assert table is not None
    assert set(table.index) == set(range(1, 13))


def test_fit_falls_back_to_level_only_then_excludes_on_total_failure(caplog):
    market = CombinedMarket("ES")
    # Only 2 months of data -- not enough for level-volume, but enough for
    # level-only (price-only) to still succeed.
    dates = pd.date_range("2023-01-01", periods=2, freq="MS")
    monthly = pd.DataFrame({"date": dates, "price": [2.0, 2.1], "landings_kg": [1000.0, 1000.0]})

    with caplog.at_level(logging.WARNING):
        selected = market.fit({("THIN", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2023-03-01"))

    assert ("THIN", "Fresh - Whole") in market.models
    assert any("level-only" in r.message or "fallback" in r.message for r in caplog.records)


def test_get_baseline_price_table_reindexes_to_all_12_months_on_fallback():
    # The level-only fallback's baseline_price only covers months present
    # in the training window (here, Jan/Feb) -- get_baseline_price_table
    # must still return all 12 months, with missing ones as NaN.
    market = CombinedMarket("ES")
    dates = pd.date_range("2023-01-01", periods=2, freq="MS")
    monthly = pd.DataFrame({"date": dates, "price": [2.0, 2.1], "landings_kg": [1000.0, 1000.0]})

    market.fit({("THIN", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2023-03-01"))

    table = market.get_baseline_price_table("THIN", "Fresh - Whole")
    assert table is not None
    assert set(table.index) == set(range(1, 13))
    missing_months = set(range(1, 13)) - {1, 2}
    for month in missing_months:
        assert pd.isna(table[month])
    assert pd.notna(table[1])
    assert pd.notna(table[2])


def test_get_live_anchor_series_varies_month_to_month_unlike_static_baseline():
    market = CombinedMarket("ES")
    own_monthly = _synthetic_monthly()
    market.fit({("ANE", "Fresh - Whole"): own_monthly}, latest_date=pd.Timestamp("2024-01-01"))

    live_series = market.get_live_anchor_series("ANE", "Fresh - Whole", own_monthly, own_monthly["date"])

    assert live_series is not None
    assert len(live_series) == len(own_monthly)
    # Unlike the static table (one value per calendar month, repeating
    # every year), the live series reflects real volume at each date, so
    # two different Januaries should generally differ.
    by_year = live_series.groupby(live_series.index.year).first()
    assert by_year.nunique() > 1


def test_get_live_anchor_series_reindexes_to_target_dates_with_nan_outside_own_range():
    market = CombinedMarket("ES")
    own_monthly = _synthetic_monthly(n_years=2)  # 2018-01 .. 2019-12
    market.fit({("ANE", "Fresh - Whole"): own_monthly}, latest_date=pd.Timestamp("2020-01-01"))

    target_dates = pd.date_range("2019-06-01", periods=12, freq="MS")  # extends past own_monthly's range
    live_series = market.get_live_anchor_series("ANE", "Fresh - Whole", own_monthly, target_dates)

    assert len(live_series) == len(target_dates)
    assert pd.notna(live_series[pd.Timestamp("2019-06-01")])
    assert pd.isna(live_series[pd.Timestamp("2020-05-01")])  # past own_monthly's last date


def test_get_live_anchor_series_returns_none_when_no_frozen_model():
    market = CombinedMarket("ES")
    own_monthly = _synthetic_monthly()
    result = market.get_live_anchor_series("GHOST", "Fresh - Whole", own_monthly, own_monthly["date"])
    assert result is None


def test_fit_excludes_species_with_zero_usable_data(caplog):
    market = CombinedMarket("ES")
    monthly = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2, freq="MS"),
        "price": [np.nan, np.nan],
        "landings_kg": [np.nan, np.nan],
    })

    with caplog.at_level(logging.WARNING):
        selected = market.fit({("GHOST", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2023-03-01"))

    assert selected[("GHOST", "Fresh - Whole")] is None
    assert ("GHOST", "Fresh - Whole") not in market.models


def test_update_volume_baseline_is_incremental_running_mean():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    before = market.models[("ANE", "Fresh - Whole")].baseline_volume[1]  # January baseline
    market.update_volume_baseline("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=5000.0)
    after = market.models[("ANE", "Fresh - Whole")].baseline_volume[1]

    assert after != before
    assert after > before  # one large new observation should pull the mean up


def test_record_history_appends_a_snapshot():
    market = CombinedMarket("ES")
    market.record_history(tick=0, prices={("ANE", "Fresh - Whole"): 2.0}, volumes={("ANE", "Fresh - Whole"): 1000.0})
    assert len(market.price_history) == 1
    assert market.price_history[0][0] == 0


def test_direct_catch_accumulator_for_sales_routed_straight_to_the_prefix():
    market = CombinedMarket("FR")
    market.add_direct_catch("ANE", "Fresh - Whole", 100.0)
    market.add_direct_catch("ANE", "Fresh - Whole", 25.0)

    assert market.get_direct_catch_snapshot()[("ANE", "Fresh - Whole")] == 125.0

    market.clear_direct_catch()
    assert market.get_direct_catch_snapshot() == {}


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


def test_fit_attaches_trend_params_to_every_key():
    market = CombinedMarket("ES")
    monthly_by_key = {
        ("ANE", "Fresh - Whole"): _trending_synthetic_monthly(),
        ("MUT", "Fresh - Whole"): _synthetic_monthly(),
    }
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2025-01-01"))

    for key in monthly_by_key:
        model = market.models[key]
        for field in ("confidence", "rate", "years_trained", "train_end"):
            assert hasattr(model.trend, field)
        assert 0.0 <= model.trend.confidence <= 1.0

    # The trending series must measure meaningfully higher confidence than
    # the flat one -- confirms each key's own monthly data is actually
    # threaded through, not a shared or stale value.
    trending_confidence = market.models[("ANE", "Fresh - Whole")].trend.confidence
    flat_confidence = market.models[("MUT", "Fresh - Whole")].trend.confidence
    assert trending_confidence == pytest.approx(1.0)
    assert flat_confidence == pytest.approx(1 / 12)


def test_window_and_baseline_identical_regardless_of_trend_confidence(monkeypatch):
    # Confirms the invariant that trend confidence can never affect window
    # selection or the baseline fit. Patches compute_trend_confidence
    # directly (rather than using a low/high-confidence dataset pair) since
    # confidence is a pure side-computation attached after fit() has
    # already chosen the window and baseline.
    monthly = _trending_synthetic_monthly()
    key = ("ANE", "Fresh - Whole")
    latest_date = pd.Timestamp("2025-01-01")

    monkeypatch.setattr("src.models.market.CombinedMarket.compute_trend_confidence", lambda m: 0.0)
    market_low = CombinedMarket("ES")
    market_low.fit({key: monthly}, latest_date=latest_date)

    monkeypatch.setattr("src.models.market.CombinedMarket.compute_trend_confidence", lambda m: 1.0)
    market_high = CombinedMarket("ES")
    market_high.fit({key: monthly}, latest_date=latest_date)

    assert market_low.models[key].trend.confidence == 0.0
    assert market_high.models[key].trend.confidence == 1.0
    assert market_low.window_labels[key] == market_high.window_labels[key]
    model_low = market_low.models[key]
    model_high = market_high.models[key]
    pd.testing.assert_series_equal(model_low.baseline_price, model_high.baseline_price)
    pd.testing.assert_series_equal(model_low.baseline_log_price, model_high.baseline_log_price)
    assert model_low.alpha == model_high.alpha
    assert model_low.beta == model_high.beta


def test_get_price_price_increases_with_positive_trend_correction():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _trending_synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2025-01-01"))
    model = market.models[("ANE", "Fresh - Whole")]
    assert model.trend.confidence == pytest.approx(1.0)  # precondition for this test

    price_near_fit_end = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2025-02-01"), volume=1000.0)
    price_far_future = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2040-02-01"), volume=1000.0)

    assert price_near_fit_end is not None
    assert price_far_future is not None
    # Constructed with a positive price trend and full confidence -- a
    # later date should predict a higher price than a date right after
    # the fit's own cutoff.
    assert price_far_future > price_near_fit_end


def test_get_price_is_unaffected_by_trend_for_a_flat_series():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))
    model = market.models[("ANE", "Fresh - Whole")]
    assert model.trend.confidence == pytest.approx(1 / 12)  # precondition: low confidence

    # Same calendar month both times, so baseline/volume-anomaly terms
    # match and any difference comes only from the trend correction. This
    # fixture's rate is ~0.0, so this mainly confirms a near-zero-rate
    # series doesn't acquire a fabricated trend (low-confidence damping
    # itself is covered by predict_trend_correction's own tests).
    price_near_fit_end = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=1000.0)
    price_far_future = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2039-02-01"), volume=1000.0)

    assert price_near_fit_end is not None
    assert price_far_future is not None
    assert price_far_future == pytest.approx(price_near_fit_end, rel=0.05)


def test_get_baseline_price_table_unaffected_by_trend_term():
    # The static anchor must only ever read baseline_price, never the
    # trend coefficient -- confirm a trend-augmented model's baseline
    # table is identical to what a plain model would produce from the
    # same seasonal means.
    market = CombinedMarket("ES")
    monthly = _trending_synthetic_monthly()
    market.fit({("ANE", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2025-01-01"))
    model = market.models[("ANE", "Fresh - Whole")]
    assert model.trend.confidence == pytest.approx(1.0)  # precondition: this is the high-confidence case

    table = market.get_baseline_price_table("ANE", "Fresh - Whole")
    assert table is not None
    assert set(table.index) == set(range(1, 13))
    # baseline_price is the plain per-month arithmetic mean, computed
    # identically regardless of whether a trend term was also fit.
    pd.testing.assert_series_equal(
        table, model.baseline_price.reindex(range(1, 13)), check_names=False
    )


def test_get_live_anchor_series_unaffected_by_trend_term():
    # Analysis-only path -- confirm it still returns a result shaped the
    # same way for a trend-bearing key. It dispatches only on
    # baseline_volume is None and never reads trend_confidence/trend_rate,
    # so it always takes the plain predict_level_volume branch with no
    # trend correction.
    market = CombinedMarket("ES")
    monthly = _trending_synthetic_monthly()
    market.fit({("ANE", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2025-01-01"))

    live_series = market.get_live_anchor_series("ANE", "Fresh - Whole", monthly, monthly["date"])
    assert live_series is not None
    assert len(live_series) == len(monthly)


import os

VALIDATED_SPECIES = ["ANE", "PIL", "HKE", "SAA", "MUT"]
CATEGORY = "Fresh - Whole"
CONFIG_DIR = "config/northwestern_med"


def _build_monthly_from_real_prices(market_code, species_code):
    from src.models.market.level_volume_model import build_complete_monthly_series

    prices = pd.read_csv(f"{CONFIG_DIR}/prices.csv", low_memory=False)
    prices["date"] = pd.to_datetime(prices["date"])
    if "size" not in prices.columns:
        prices["size"] = None
    blended = prices[prices["size"].isna()]
    rows = blended[
        (blended["market_code"] == market_code)
        & (blended["category_code"] == CATEGORY)
        & (blended["species_code"] == species_code)
    ]
    return build_complete_monthly_series(rows)


# Measured against real production data. Window label must match exactly
# (deterministic, from fit_level_volume_model/select_best_window).
# confidence and rate use generous tolerances since small fencepost
# differences in window boundaries are expected, not a bug.
EXPECTED_TREND_PARAMS = {
    ("ES", "ANE"): {"window": "3yr", "confidence": 0.92, "rate": 0.0625},
    ("ES", "PIL"): {"window": "2yr", "confidence": 0.00, "rate": 0.0060},
    ("ES", "HKE"): {"window": "Full", "confidence": 0.00, "rate": 0.0157},
    ("ES", "SAA"): {"window": "6yr", "confidence": 0.08, "rate": 0.0203},
    ("ES", "MUT"): {"window": "3yr", "confidence": 0.42, "rate": 0.0150},
    ("FR", "ANE"): {"window": "3yr", "confidence": 0.08, "rate": 0.0337},
    ("FR", "PIL"): {"window": "Full", "confidence": 0.08, "rate": -0.0027},
    ("FR", "HKE"): {"window": "4yr", "confidence": 0.17, "rate": 0.0294},
    ("FR", "SAA"): {"window": "Full", "confidence": 0.25, "rate": 0.0226},
    ("FR", "MUT"): {"window": "3yr", "confidence": 0.42, "rate": -0.0252},
}


@pytest.mark.skipif(
    not os.path.exists(f"{CONFIG_DIR}/prices.csv"),
    reason="requires config/northwestern_med/prices.csv",
)
def test_real_data_trend_params_match_measured_values():
    for market_code in ["ES", "FR"]:
        market = CombinedMarket(market_code)
        monthly_by_key = {
            (species, CATEGORY): _build_monthly_from_real_prices(market_code, species)
            for species in VALIDATED_SPECIES
        }
        monthly_by_key = {k: v for k, v in monthly_by_key.items() if not v.empty}
        latest_date = pd.Timestamp(
            max(m["date"].max() for m in monthly_by_key.values())
        )

        market.fit(monthly_by_key, latest_date=latest_date)

        for species in VALIDATED_SPECIES:
            key = (species, CATEGORY)
            expected = EXPECTED_TREND_PARAMS.get((market_code, species))
            if expected is None or key not in market.models:
                continue
            model = market.models[key]
            assert market.window_labels[key] == expected["window"], (
                f"{market_code}/{species}: expected window {expected['window']!r}, "
                f"got {market.window_labels[key]!r}"
            )
            assert model.trend.confidence == pytest.approx(expected["confidence"], abs=0.1), (
                f"{market_code}/{species}: expected confidence ~{expected['confidence']}, "
                f"got {model.trend.confidence}"
            )
            assert model.trend.rate == pytest.approx(expected["rate"], abs=0.015), (
                f"{market_code}/{species}: expected rate ~{expected['rate']}, got {model.trend.rate}"
            )
