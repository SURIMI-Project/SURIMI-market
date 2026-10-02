import pandas as pd
import pytest

from src.models.market.window_selection import build_rolling_splits, select_best_window


def test_build_rolling_splits_anchors_at_latest_date():
    latest = pd.Timestamp("2025-01-01")
    splits = build_rolling_splits(latest)

    assert splits == [
        (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01")),
        (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")),
        (pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")),
    ]


def test_select_best_window_picks_lowest_scoring_candidate():
    candidates = {"2yr": 2, "4yr": 4, "Full": None}
    rolling_splits = [
        (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01")),
        (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")),
    ]

    # "4yr" always scores best (lowest MAPE); the other two are worse.
    scores_by_label = {"2yr": 10.0, "4yr": 2.0, "Full": 8.0}

    def fit_fn(train_start, train_end):
        return {"train_start": train_start, "train_end": train_end}

    def score_fn(params, test_start, test_end):
        # Reverse-engineer which candidate this is from the gap between
        # train_end and test_start having no bearing here -- instead,
        # thread the label through via train_start's distance from train_end.
        if params["train_start"] is None:
            return scores_by_label["Full"]
        years_back = round((params["train_end"] - params["train_start"]).days / 365.25)
        label = {2: "2yr", 4: "4yr"}.get(years_back, "Full")
        return scores_by_label[label]

    result = select_best_window(candidates, rolling_splits, fit_fn, score_fn)

    assert result == ("4yr", 4)


def test_select_best_window_returns_none_when_every_candidate_fails():
    candidates = {"2yr": 2, "4yr": 4}
    rolling_splits = [(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01"))]

    def fit_fn(train_start, train_end):
        raise ValueError("not enough usable training rows")

    def score_fn(params, test_start, test_end):
        raise AssertionError("score_fn should never be called if fit_fn always raises")

    result = select_best_window(candidates, rolling_splits, fit_fn, score_fn)

    assert result is None


def test_select_best_window_respects_sanity_check_fn():
    candidates = {"2yr": 2, "4yr": 4}
    rolling_splits = [(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01"))]

    def fit_fn(train_start, train_end):
        return {"train_start": train_start}

    def score_fn(params, test_start, test_end):
        return 1.0

    def sanity_check_fn(params):
        # Reject every candidate as implausible -- the safeguard must
        # trigger even though fit_fn and score_fn both "succeed".
        return False

    result = select_best_window(candidates, rolling_splits, fit_fn, score_fn, sanity_check_fn)

    assert result is None


def test_select_best_window_skips_splits_with_no_valid_test_data():
    candidates = {"2yr": 2}
    rolling_splits = [
        (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01")),
        (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")),
    ]

    def fit_fn(train_start, train_end):
        return {}

    call_count = {"n": 0}

    def score_fn(params, test_start, test_end):
        call_count["n"] += 1
        # First split has no test data (e.g. species not yet landed then);
        # second split does.
        return None if call_count["n"] == 1 else 5.0

    result = select_best_window(candidates, rolling_splits, fit_fn, score_fn)

    assert result == ("2yr", 2)
