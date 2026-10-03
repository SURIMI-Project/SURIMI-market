"""
Generic, model-agnostic adaptive training-window selection (spec section 7).
Callers supply fit/score closures; this module knows nothing about what
kind of model is being fit -- it only knows how to generate rolling-origin
splits and pick whichever candidate window length generalizes best.
"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from . import consts


def build_rolling_splits(latest_date: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Generate N_ROLLING_SPLITS rolling-origin splits anchored at the latest
    available real data point (never at start_time -- see plan header
    decision 5). Each split's test window is ROLLING_TEST_HORIZON_YEARS
    long; consecutive splits step back by ROLLING_SPLIT_STEP_YEARS.
    """
    splits = []
    for i in reversed(range(consts.N_ROLLING_SPLITS)):
        k_years = consts.ROLLING_TEST_HORIZON_YEARS + i * consts.ROLLING_SPLIT_STEP_YEARS
        train_end = latest_date - relativedelta(years=k_years)
        test_end = latest_date - relativedelta(years=i * consts.ROLLING_SPLIT_STEP_YEARS)
        splits.append((train_end, test_end))
    return splits


def rank_windows_by_score(
    candidates: dict[str, int | None],
    rolling_splits: list[tuple[pd.Timestamp, pd.Timestamp]],
    fit_fn,
    score_fn,
    sanity_check_fn=None,
) -> list[tuple[str, int | None]]:
    """
    Score every candidate window length across every rolling split and
    return every candidate that produced at least one usable score,
    ordered best (lowest mean score) first.

    An empty list is the sanity-bound safeguard signal (spec section 7):
    callers MUST treat this as "fall back to the simpler fit cascade",
    never as "proceed with no window selected". A caller that only needs
    the single best candidate should use select_best_window instead; this
    function exists for callers that need to retry the NEXT-best candidate
    if the best one's final re-fit (on the literal final window, which
    this function never attempts) turns out to fail -- rolling-origin
    validation success does not guarantee the final re-fit succeeds, since
    the final window can differ in exactly which months it includes.
    """
    if sanity_check_fn is None:
        sanity_check_fn = lambda params: True

    mean_scores: dict[str, float] = {}

    for label, years in candidates.items():
        split_scores = []
        for train_end, test_end in rolling_splits:
            train_start = (train_end - relativedelta(years=years)) if years is not None else None
            try:
                params = fit_fn(train_start, train_end)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not sanity_check_fn(params):
                continue
            score = score_fn(params, train_end, test_end)
            if score is not None:
                split_scores.append(score)

        if split_scores:
            mean_scores[label] = float(np.mean(split_scores))

    return [(label, candidates[label]) for label in sorted(mean_scores, key=mean_scores.get)]


def select_best_window(
    candidates: dict[str, int | None],
    rolling_splits: list[tuple[pd.Timestamp, pd.Timestamp]],
    fit_fn,
    score_fn,
    sanity_check_fn=None,
) -> tuple[str, int | None] | None:
    """
    Score every candidate window length across every rolling split and
    return the (label, years) of whichever scores lowest on average.

    Returns None if every candidate fails to produce any usable score on
    any split -- this is the sanity-bound safeguard signal (spec section 7):
    callers MUST treat this as "fall back to the simpler fit cascade",
    never as "proceed with no window selected".
    """
    ranked = rank_windows_by_score(candidates, rolling_splits, fit_fn, score_fn, sanity_check_fn)
    return ranked[0] if ranked else None
