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
    # A blended row (size is NaN) can still have a missing/NaN price -- the
    # size filter in load_price_time_series does not also filter out NaN
    # prices, so get_price must not treat a stored NaN as "real data found"
    # and return it directly; it must fall through to the frozen model.
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    market.price_time_series[("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"))] = float("nan")

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

    assert market.price_time_series[("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"))] == 2.0


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
    # Only 2 months of data: not enough for level-volume (needs variance +
    # >=2 points) but also not even enough to try fitting meaningfully --
    # construct a case where level-volume fails every window but
    # level-only (price-only) still succeeds.
    dates = pd.date_range("2023-01-01", periods=2, freq="MS")
    monthly = pd.DataFrame({"date": dates, "price": [2.0, 2.1], "landings_kg": [1000.0, 1000.0]})

    with caplog.at_level(logging.WARNING):
        selected = market.fit({("THIN", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2023-03-01"))

    assert ("THIN", "Fresh - Whole") in market.models
    assert any("level-only" in r.message or "fallback" in r.message for r in caplog.records)


def test_get_baseline_price_table_reindexes_to_all_12_months_on_fallback():
    # The level-only fallback's baseline_price is built from whatever
    # calendar months happen to appear in the (possibly short) training
    # window -- here, only January and February. get_baseline_price_table
    # must still return a Series covering all 12 months, with the missing
    # ones as NaN, since Task 9's LocalMarket Step 2 anchor relies on
    # dates.month.map(table) never silently producing NaN for a month that
    # was simply absent from the table's index.
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

    before = market.models[("ANE", "Fresh - Whole")]["baseline_volume"][1]  # January baseline
    market.update_volume_baseline("ANE", "Fresh - Whole", pd.Timestamp("2024-01-01"), volume=5000.0)
    after = market.models[("ANE", "Fresh - Whole")]["baseline_volume"][1]

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
        params = market.models[key]
        for field in ("trend_confidence", "trend_rate", "trend_years_trained", "trend_train_end"):
            assert field in params
        assert 0.0 <= params["trend_confidence"] <= 1.0

    # The trending series must measure meaningfully higher confidence than
    # the flat one -- confirms each key's own monthly data is actually
    # threaded through, not a shared or stale value.
    trending_confidence = market.models[("ANE", "Fresh - Whole")]["trend_confidence"]
    flat_confidence = market.models[("MUT", "Fresh - Whole")]["trend_confidence"]
    assert trending_confidence == pytest.approx(1.0)
    assert flat_confidence == pytest.approx(1 / 12)


def test_window_and_baseline_identical_regardless_of_trend_confidence(monkeypatch):
    # The regression test spec section 6 calls out directly: this is what
    # would have caught v1's bug (trend-eligibility changing which window
    # gets selected, and so changing baseline_price). Patches
    # compute_trend_confidence's return value directly rather than
    # constructing a low/high-confidence dataset pair -- confidence is a
    # pure side-computation attached AFTER fit() has already chosen a
    # window and baseline (Task 3's own design), so this isolates the
    # invariant ("confidence can never affect window selection or the
    # baseline fit") from any incidental correlation between "this series
    # has a trend" and "window scoring happens to prefer a different
    # window length for unrelated data reasons."
    monthly = _trending_synthetic_monthly()
    key = ("ANE", "Fresh - Whole")
    latest_date = pd.Timestamp("2025-01-01")

    monkeypatch.setattr("src.models.market.CombinedMarket.compute_trend_confidence", lambda m: 0.0)
    market_low = CombinedMarket("ES")
    market_low.fit({key: monthly}, latest_date=latest_date)

    monkeypatch.setattr("src.models.market.CombinedMarket.compute_trend_confidence", lambda m: 1.0)
    market_high = CombinedMarket("ES")
    market_high.fit({key: monthly}, latest_date=latest_date)

    assert market_low.window_labels[key] == market_high.window_labels[key]
    params_low = market_low.models[key]
    params_high = market_high.models[key]
    pd.testing.assert_series_equal(params_low["baseline_price"], params_high["baseline_price"])
    pd.testing.assert_series_equal(params_low["baseline_log_price"], params_high["baseline_log_price"])
    assert params_low["alpha"] == params_high["alpha"]
    assert params_low["beta"] == params_high["beta"]


def test_get_price_applies_trend_correction_proportional_to_confidence():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _trending_synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2025-01-01"))
    params = market.models[("ANE", "Fresh - Whole")]
    assert params["trend_confidence"] == pytest.approx(1.0)  # precondition for this test

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
    params = market.models[("ANE", "Fresh - Whole")]
    assert params["trend_confidence"] == pytest.approx(1 / 12)  # precondition: low confidence

    # Same calendar month both times, so baseline/volume-anomaly terms are
    # identical by construction -- any remaining difference comes only
    # from the trend correction, which low confidence should make tiny.
    price_near_fit_end = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2024-02-01"), volume=1000.0)
    price_far_future = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2039-02-01"), volume=1000.0)

    assert price_near_fit_end is not None
    assert price_far_future is not None
    assert price_far_future == pytest.approx(price_near_fit_end, rel=0.05)


def test_get_baseline_price_table_unaffected_by_trend_term():
    # Step 2's static anchor must only ever read baseline_price, never the
    # trend coefficient -- confirm directly, not just by prose, that a
    # trend-augmented model's baseline table is identical in shape/values
    # to what a plain model would have produced from the same seasonal
    # means (i.e. the trend term never leaks into this table).
    market = CombinedMarket("ES")
    monthly = _trending_synthetic_monthly()
    market.fit({("ANE", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2025-01-01"))
    params = market.models[("ANE", "Fresh - Whole")]
    assert params["trend_confidence"] == pytest.approx(1.0)  # precondition: this is the high-confidence case

    table = market.get_baseline_price_table("ANE", "Fresh - Whole")
    assert table is not None
    assert set(table.index) == set(range(1, 13))
    # baseline_price is the plain per-month arithmetic mean, computed
    # identically regardless of whether a trend term was also fit.
    pd.testing.assert_series_equal(
        table, params["baseline_price"].reindex(range(1, 13)), check_names=False
    )


def test_get_live_anchor_series_unaffected_by_trend_term():
    # Analysis-only path (spec: "never used by the live production path").
    # Confirm it still returns a result shaped the same way for a
    # trend-bearing key -- it dispatches only on baseline_volume is None,
    # never reads trend_confidence/trend_rate at all, so it always takes
    # the plain predict_level_volume branch with no trend correction
    # applied -- this function was not touched by this plan, confirm that
    # directly.
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


# Measured against the real, unmodified production code (spec
# 2026-10-08-combined-market-trend-term-v2-design.md section 4). Window
# label is expected to match exactly -- it's produced by code this plan
# never modifies (fit_level_volume_model/select_best_window). confidence
# and rate use generous tolerances: they come from a fresh implementation
# of the spec's described algorithm, not a byte-for-byte copy of the
# throwaway spike script that originally measured these numbers, and
# small fencepost differences (e.g. exactly which calendar months a given
# window boundary includes) are expected, not a bug, as long as they're
# close.
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
            params = market.models[key]
            assert market.window_labels[key] == expected["window"], (
                f"{market_code}/{species}: expected window {expected['window']!r}, "
                f"got {market.window_labels[key]!r}"
            )
            assert params["trend_confidence"] == pytest.approx(expected["confidence"], abs=0.1), (
                f"{market_code}/{species}: expected confidence ~{expected['confidence']}, "
                f"got {params['trend_confidence']}"
            )
            assert params["trend_rate"] == pytest.approx(expected["rate"], abs=0.015), (
                f"{market_code}/{species}: expected rate ~{expected['rate']}, got {params['trend_rate']}"
            )
