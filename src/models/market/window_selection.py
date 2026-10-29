"""
Generic, model-agnostic adaptive training-window selection. Callers supply
fit/score closures; this module knows nothing about what kind of model is
being fit -- it only knows how to generate rolling-origin splits and pick
whichever candidate window length generalizes best.

Also holds fit_price_model_with_window_selection, the shared
window-selection-then-fallback cascade used identically by
CombinedMarket.fit and LocalMarket.fit_base_models (both fit the same
level-volume/level-only base model, just against combined-market-level or
per-port-level monthly data).
"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from . import consts
from .price_models import LevelOnlyModel, LevelVolumeModel
from .market_trackers import seed_tracker
from .level_volume_model import fit_level_volume_model, predict_level_volume, score_mape


def build_rolling_splits(latest_date: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Generate N_ROLLING_SPLITS rolling-origin splits anchored at the latest
    available real data point, never at start_time. Each split's test
    window is ROLLING_TEST_HORIZON_YEARS long; consecutive splits step
    back by ROLLING_SPLIT_STEP_YEARS.
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
    Score every candidate across every rolling split and return the viable
    ones, ordered best (lowest mean score) first. An empty result means
    none scored -- callers must fall back to the simpler fit cascade
    rather than proceed with no window selected.

    Unlike select_best_window, returns every viable candidate so callers
    can retry the next-best one if the top candidate's final re-fit fails
    -- rolling-origin success doesn't guarantee that, since the final
    window's exact months can differ from any split's.
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

    Returns None if no candidate produced any usable score -- callers
    must fall back to the simpler fit cascade, not proceed with no
    window selected.
    """
    ranked = rank_windows_by_score(candidates, rolling_splits, fit_fn, score_fn, sanity_check_fn)
    return ranked[0] if ranked else None


def fit_price_model_with_window_selection(
    monthly_by_key: dict[tuple[str, str], pd.DataFrame],
    latest_date: pd.Timestamp,
    *,
    build_trend,
    models: dict,
    window_labels: dict,
    volume_baseline: dict,
    logger,
    class_label: str,
    market_code: str,
    window_noun: str,
    exclusion_suffix: str = "",
    on_model_fitted=None,
) -> dict[tuple[str, str], str | None]:
    """
    Shared by CombinedMarket.fit and LocalMarket.fit_base_models (identical
    window-selection-then-fallback cascade in both, previously duplicated):
    select the best rolling-origin-validated training window for the
    level-volume model, falling back to a level-only fit if no window is
    stable or the chosen window's final re-fit fails; exclude the key
    entirely if even the level-only fallback fails.

    models/window_labels/volume_baseline are the caller's own dicts,
    mutated in place. volume_baseline is seeded (via
    VolumeBaselineTracker.seed_tracker) only on the full level-volume
    success path, never on the level-only fallback -- matching both
    classes' pre-extraction behavior exactly.

    build_trend(monthly, latest_date) -> TrendCorrection lets each caller
    supply its own trend construction (CombinedMarket fits a real trend
    per call; LocalMarket passes a closure returning one pre-built
    TrendCorrection.identity(latest_date) instance, computed once outside
    this function and reused for every key/branch).

    logger/class_label/market_code/window_noun/exclusion_suffix reproduce
    each caller's exact historical log wording -- CombinedMarket and
    LocalMarket differ in a few words beyond just the class name, so this
    isn't reduced to one generic message.

    on_model_fitted(key, model, monthly), if given, runs after EVERY
    successful fit (both the level-volume success path and the level-only
    fallback) -- used by LocalMarket to additionally stash
    _base_predicted_series; CombinedMarket passes None.
    """
    selected: dict[tuple[str, str], str | None] = {}

    for key, monthly in monthly_by_key.items():
        rolling_splits = build_rolling_splits(latest_date)

        def fit_fn(train_start, train_end, _monthly=monthly):
            return fit_level_volume_model(_monthly, train_start, train_end)

        def score_fn(params, test_start, test_end, _monthly=monthly):
            predicted = predict_level_volume(_monthly, params)
            mask = (_monthly["date"] >= test_start) & (_monthly["date"] < test_end)
            return score_mape(predicted[mask], _monthly["price"][mask])

        result = select_best_window(consts.WINDOW_CANDIDATE_YEARS, rolling_splits, fit_fn, score_fn)

        model = None
        if result is not None:
            window_label, years = result
            train_start = latest_date - relativedelta(years=years) if years is not None else None
            try:
                model = LevelVolumeModel.fit(monthly, train_start, latest_date, build_trend(monthly, latest_date))
            except (ValueError, np.linalg.LinAlgError):
                # Rolling-origin validation scored this window acceptably, but the
                # final window (train_start..latest_date) can still fail to fit if
                # sparse real data makes the volume-anomaly series degenerate (zero
                # variance) -- fall through to the same fallback used when no window
                # was selected at all.
                logger.warning(
                    f"{class_label} {market_code}: selected window {window_label} for {key} "
                    f"scored acceptably in rolling-origin validation but failed to re-fit on "
                    f"the final window ending at {latest_date}; falling back to level-only fit"
                )
                model = None

            if model is not None:
                models[key] = model
                window_labels[key] = window_label
                seed_tracker(volume_baseline, key, monthly, train_start, latest_date)
                if on_model_fitted is not None:
                    on_model_fitted(key, model, monthly)
                selected[key] = window_label
                continue

        # Fallback: try a level-only fit.
        else:
            logger.warning(
                f"{class_label} {market_code}: no stable {window_noun} found "
                f"for {key}, falling back to level-only fit"
            )
        try:
            model = LevelOnlyModel.fit(monthly, None, latest_date, build_trend(monthly, latest_date))
            models[key] = model
            window_labels[key] = "level-only-fallback"
            if on_model_fitted is not None:
                on_model_fitted(key, model, monthly)
            selected[key] = "level-only-fallback"
        except ValueError:
            logger.warning(
                f"{class_label} {market_code}: no usable price history at all "
                f"for {key}, excluding from output{exclusion_suffix}"
            )
            selected[key] = None

    return selected
