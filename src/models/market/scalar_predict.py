"""
Shared "build the minimal single-row pandas structures a vectorized
predict_* function needs, call it, unwrap its one-row result to a plain
scalar" helper. LevelOnlyModel.predict, LevelVolumeModel.predict, and
StackedModel.predict each independently built this throwaway
DataFrame/Series machinery for one tick's prediction -- this runs every
tick, for every port/species key, so it's real, avoidable pandas
construction/teardown overhead on the hot path, not just a readability
cost. Now implemented once, used by all three.
"""
import pandas as pd


def predict_scalar(
    predict_fn,
    date: pd.Timestamp,
    landings_kg,
    params: dict,
    *series_values: float,
    trend_factor: float = 1.0,
) -> float | None:
    """
    predict_fn is one of the vectorized predict_* free functions (meant for
    a whole historical series); this builds the single-row inputs it needs
    for one (date, ...) tick, calls it, and unwraps the result.

    landings_kg becomes the single row's "landings_kg" column. series_values
    are extra per-tick scalars the neighbor-informed StackedModel needs
    (anchor_value, neighbor_value) -- each becomes its own single-value
    pd.Series indexed on date, in the same order predict_fn expects them,
    positioned between the single-row DataFrame and params; omitted
    entirely for the base models (LevelOnlyModel/LevelVolumeModel), whose
    predict_fn takes just (monthly, params).

    trend_factor multiplies the raw prediction before unwrapping -- the
    base models pass their own TrendCorrection.factor(date); StackedModel
    has no trend term, so it relies on the default 1.0 (an exact no-op:
    multiplying by precisely 1.0 changes no bit of a finite float, and
    leaves NaN/None-derived NaN as NaN).

    Returns None (never NaN) if the unwrapped value is missing, matching
    every one of the three callers' previous behavior.
    """
    single_row = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [landings_kg]})
    series_args = tuple(pd.Series([value], index=[date]) for value in series_values)
    predicted = predict_fn(single_row, *series_args, params) * trend_factor
    value = predicted.iloc[0]
    return float(value) if pd.notna(value) else None
