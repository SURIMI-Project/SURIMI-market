# CombinedMarket Trend Term v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v1's trend-term mechanism (merged `4558d5a..32a3f02`) with the v2 design: window selection and the frozen baseline become completely unconditional again (the direct fix for a real regression v1 caused in `LocalMarket` Step 2), and the trend itself becomes a separately-measured, continuous, additive correction layered on top.

**Architecture:** `CombinedMarket.fit()`'s window-length selection and final re-fit call `fit_level_volume_model`/`predict_level_volume` unconditionally, with zero branching on any trend-related property, for every key. After that fit succeeds (main path or the level-only fallback alike), a separate step computes a continuous trend confidence (same per-calendar-month OLS significance math as v1's gate, no threshold) and a historical rate (sequential 3-year block averages, not a regression), and attaches both to the frozen params. `get_price()` always multiplies the unchanged baseline prediction by `exp(confidence * rate * trend_years)` — a pure no-op (×1.0) for a key with near-zero confidence, so no "does this key have a trend" branch is needed anywhere in the prediction path either.

**Tech Stack:** Python, numpy (`polyfit`), pandas. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-08-combined-market-trend-term-v2-design.md` — supersedes `docs/superpowers/specs/2026-10-07-combined-market-trend-term-design.md` (v1) in full. Read both the v2 spec and this plan; the spec is the authority this plan argues from.

## Global Constraints

- **Scope: `CombinedMarket` only.** `LocalMarket`, `stacked_model.py`, and `window_selection.py`'s shared `build_rolling_splits`/`select_best_window`/`rank_windows_by_score` are never modified by any task in this plan.
- **Window selection and `baseline_price`/`alpha`/`beta` must be IDENTICAL for every species, with no trend-eligibility branching anywhere in that path.** This is the single most important constraint — every task that touches `CombinedMarket.fit()` must preserve it. Concretely: `fit_fn`/`score_fn` inside `fit()`'s per-key loop always wrap the plain `fit_level_volume_model`/`predict_level_volume`, never a trend-augmented variant, for every key, unconditionally.
- **No pass/fail gate anywhere.** Confidence is a continuous value in `[0, 1]`, always computed, always applied as a multiplier (possibly ~0) — never a branch that fully excludes a species from getting a trend correction computed.
- **One uniform procedure, every species.** Same formula, same constants (`TREND_CONFIDENCE_T_THRESHOLD = 2.0`, `TREND_RATE_BLOCK_YEARS = 3`, damping exponent = 1, i.e. linear/no exponent applied at all), applied identically. No species is named in code.
- **Trend params are attached to every key that gets a model at all** — the main level-volume path and the level-only fallback path alike — so `get_price()` never has to branch on which path produced a given key's params (unlike v1's `"gamma" in params` check).

---

### Task 1: Continuous trend confidence (`trend_confidence.py`)

**Files:**
- Create: `src/models/market/trend_confidence.py`
- Modify: `src/models/market/consts.py:52-61` (replace the trend-gate constants block)
- Create: `src/models/market/test/test_trend_confidence.py`
- Delete: `src/models/market/trend_gate.py`
- Delete: `src/models/market/test/test_trend_gate.py`

**Interfaces:**
- Consumes: nothing new — same per-calendar-month OLS math as v1's `trend_gate.py` (read it before starting; it's being replaced, not re-derived).
- Produces: `compute_trend_confidence(monthly: pd.DataFrame) -> float`, a continuous value in `[0.0, 1.0]`. Consumed by Task 3 (`CombinedMarket._attach_trend_params`).

- [ ] **Step 1: Replace the trend-gate constants in `consts.py`**

In `src/models/market/consts.py`, replace lines 52-61:

```python
# Trend-term gate (spec 2026-10-07-combined-market-trend-term-design.md,
# section 3): per-calendar-month OLS on the FULL real historical record
# (no train/test split), used to decide whether a (species_code,
# category_code) key's CombinedMarket fit should include a trend term at
# all. Deliberately strict -- chosen to cleanly separate the one
# validated real case (ANE@ES: 12/12 months agree in sign, 11/12
# individually significant) from the nearest false positive (HKE@FR:
# 12/12 sign agreement, but only 2/12 significant).
TREND_GATE_T_THRESHOLD = 2.0
TREND_GATE_MIN_SIGNIFICANT_MONTHS = 9
```

with:

```python
# Trend confidence (spec 2026-10-08-combined-market-trend-term-v2-design.md
# section 3, Step 2): per-calendar-month OLS on the FULL real historical
# record (no train/test split), continuous -- NOT a pass/fail gate (v1's
# TREND_GATE_MIN_SIGNIFICANT_MONTHS threshold is gone; see
# trend_confidence.py). confidence = (months individually significant in
# the majority sign direction) / 12.
TREND_CONFIDENCE_T_THRESHOLD = 2.0
```

(Task 2 appends a second constant, `TREND_RATE_BLOCK_YEARS`, right after this block.)

- [ ] **Step 2: Write the failing tests**

Create `src/models/market/test/test_trend_confidence.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.models.market.trend_confidence import compute_trend_confidence


def _make_monthly(n_years, slope, noise_std, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_years * 12, freq="MS")
    years = dates.year.to_numpy(dtype=float)
    seasonal = np.array([2.0 + 0.1 * d.month for d in dates])
    trend = slope * (years - years.min())
    noise = rng.normal(0, noise_std, len(dates))
    price = seasonal + trend + noise
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": [1000.0] * len(dates)})


def test_confidence_is_1_for_a_clear_consistent_trend():
    monthly = _make_monthly(n_years=10, slope=0.1, noise_std=0.05, seed=0)
    assert compute_trend_confidence(monthly) == pytest.approx(1.0)


def test_confidence_is_low_for_flat_noise_with_no_real_trend():
    monthly = _make_monthly(n_years=10, slope=0.0, noise_std=0.1, seed=0)
    assert compute_trend_confidence(monthly) == pytest.approx(1 / 12)


def test_confidence_is_partial_when_sign_agrees_but_not_all_months_significant():
    # Shaped after the real HKE@FR near-miss found during the v1
    # investigation: all 12 months drift the same direction, but the slope
    # is small relative to its own noise, so only some months are
    # individually significant -- confidence must land strictly between 0
    # and 1, not snap to either extreme the way v1's pass/fail gate did.
    monthly = _make_monthly(n_years=10, slope=0.03, noise_std=0.12, seed=6)
    assert compute_trend_confidence(monthly) == pytest.approx(0.25)


def test_confidence_is_zero_with_insufficient_history():
    monthly = _make_monthly(n_years=2, slope=0.1, noise_std=0.05, seed=0)
    assert compute_trend_confidence(monthly) == 0.0


def test_confidence_handles_a_month_with_zero_variance_in_years_without_crashing():
    # Degenerate: January has multiple observations (n >= 3) but all from
    # the same calendar year, so sxx == 0 -- no year-over-year slope is
    # estimable for that month at all; it must simply not count toward the
    # numerator, not raise. Other months have 3 years of data (2015-2017)
    # with an exactly linear embedded trend and zero noise.
    dates = []
    prices = []

    dates.extend([
        pd.Timestamp(2015, 1, 1),
        pd.Timestamp(2015, 1, 15),
        pd.Timestamp(2015, 1, 28),
    ])
    prices.extend([2.0, 2.05, 2.02])

    for month in range(2, 13):
        for year in [2015, 2016, 2017]:
            dates.append(pd.Timestamp(year=year, month=month, day=15))
            prices.append(2.0 + (year - 2015) * 0.15)

    monthly = pd.DataFrame({
        "date": dates,
        "price": prices,
        "landings_kg": [1000.0] * len(dates),
    })
    # January is skipped (sxx == 0); the other 11 months have a perfectly
    # linear embedded trend, which in floating point still produces a
    # tiny-but-nonzero residual, giving an enormous (not undefined) t-stat
    # -- so all 11 evaluated months count as significant.
    assert compute_trend_confidence(monthly) == pytest.approx(11 / 12)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest src/models/market/test/test_trend_confidence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.market.trend_confidence'`

- [ ] **Step 4: Write `trend_confidence.py`**

Create `src/models/market/trend_confidence.py`:

```python
"""
Continuous trend confidence (spec
2026-10-08-combined-market-trend-term-v2-design.md, section 3, Step 2) for
a (species_code, category_code) key's CombinedMarket fit. Uses ONLY the
complete real historical record -- no train/test split, no dependency on
level_volume_model.py's own fit. Per-calendar-month OLS (year -> price),
same slope/standard-error/t-statistic math as v1's trend_gate.py, but
returned as a continuous fraction in [0, 1] instead of a boolean -- there
is no sign-agreement requirement and no significant-months threshold. A
species with no real trend naturally measures near 0; a species with a
clear, sustained trend measures near 1. Never a hard exclusion.
"""
import numpy as np
import pandas as pd

from . import consts


def compute_trend_confidence(monthly: pd.DataFrame) -> float:
    """
    Fraction (0.0-1.0) of the 12 calendar months whose own year-over-year
    OLS slope is individually significant (|t| > TREND_CONFIDENCE_T_THRESHOLD)
    in the majority sign direction across all 12 months. A month with
    fewer than 3 years of observations, or whose observations all fall in
    a single year (zero year-variance, so no slope is estimable), simply
    does not count toward the numerator -- it is treated as "not
    significant" for that month rather than raising or forcing the whole
    key to a fixed value, since this is a continuous measure, not a gate.
    """
    monthly = monthly.dropna(subset=["price"]).copy()
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    slopes: dict[int, float] = {}
    tstats: dict[int, float] = {}
    for month in range(1, 13):
        sub = monthly[monthly["month"] == month]
        years = sub["year"].to_numpy(dtype=float)
        prices = sub["price"].to_numpy(dtype=float)
        n = len(years)
        if n < 3:
            continue  # not enough evidence to assess this month

        x_bar, y_bar = years.mean(), prices.mean()
        sxx = ((years - x_bar) ** 2).sum()
        if sxx == 0:
            continue  # every observation of this month falls in the same year

        slope = ((years - x_bar) * (prices - y_bar)).sum() / sxx
        intercept = y_bar - slope * x_bar
        residuals = prices - (intercept + slope * years)
        s2 = (residuals ** 2).sum() / (n - 2)
        se_slope = np.sqrt(s2 / sxx)
        t_stat = slope / se_slope if se_slope > 0 else 0.0

        slopes[month] = slope
        tstats[month] = t_stat

    if not slopes:
        return 0.0

    n_positive = sum(1 for s in slopes.values() if s > 0)
    n_negative = sum(1 for s in slopes.values() if s < 0)
    if n_positive == 0 and n_negative == 0:
        return 0.0
    majority_sign = 1 if n_positive >= n_negative else -1

    n_significant = sum(
        1 for month, t in tstats.items()
        if abs(t) > consts.TREND_CONFIDENCE_T_THRESHOLD and np.sign(slopes[month]) == majority_sign
    )
    return n_significant / 12
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest src/models/market/test/test_trend_confidence.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 6: Delete v1's gate file and its test**

```bash
git rm src/models/market/trend_gate.py src/models/market/test/test_trend_gate.py
```

- [ ] **Step 7: Run the full test suite to confirm nothing else references the deleted module**

Run: `python -m pytest src/models/market/test/ -v -k "not fidelity and not real_data"`
Expected: FAIL only on collection errors in `test_CombinedMarket.py`/`test_level_volume_model.py` if they still import `trend_gate` or the old trend functions (expected at this point — Tasks 2-4 fix those). If the only failures are those pre-existing imports, this step is confirming the right thing; do not try to fix them here.

- [ ] **Step 8: Commit**

```bash
git add src/models/market/trend_confidence.py src/models/market/test/test_trend_confidence.py src/models/market/consts.py
git rm src/models/market/trend_gate.py src/models/market/test/test_trend_gate.py
git commit -m "feat: replace the trend pass/fail gate with continuous confidence"
```

(If Step 6 already staged the deletions, the second `git rm` is a no-op confirmation — fine either way.)

---

### Task 2: Block-based trend rate (`level_volume_model.py`)

**Files:**
- Modify: `src/models/market/level_volume_model.py:143-229` (replace `fit_level_volume_trend_model`/`predict_level_volume_trend` entirely)
- Modify: `src/models/market/consts.py` (append `TREND_RATE_BLOCK_YEARS` after Task 1's block)
- Modify: `src/models/market/test/test_level_volume_model.py:1-20,132-218` (imports; replace the trend-specific tests)

**Interfaces:**
- Consumes: nothing new.
- Produces: `fit_trend_rate_model(monthly: pd.DataFrame, train_end: pd.Timestamp) -> dict` returning `{"rate": float, "years_trained": float}`; `predict_trend_correction(monthly: pd.DataFrame, params: dict) -> pd.Series`, reading `params["trend_rate"]`, `params["trend_years_trained"]`, `params["trend_confidence"]`, `params["trend_train_end"]` and returning a multiplicative correction factor (not a final price). Both consumed by Task 3/4 (`CombinedMarket.py`).

- [ ] **Step 1: Append the block-size constant to `consts.py`**

Immediately after the block Task 1 wrote (the `TREND_CONFIDENCE_T_THRESHOLD = 2.0` line), append:

```python

# Trend rate (spec v2 section 3, Step 3): sequential block size, in years,
# used to measure the historical rate from block-average log-prices
# (numpy.polyfit across all blocks that fit in the complete real record
# before train_end, built forward from the oldest available date). 3yr
# was empirically found to balance a long-enough fair test horizon
# against enough blocks to fit a meaningful slope -- a larger block size
# needs more lead-time history before its first usable cutoff, which
# shortens how far into the future that configuration can be fairly
# tested (confirmed directly: 4yr blocks could only be tested out to 1-7
# years vs. 4-11 years for 2-3yr blocks) -- see the v2 spec section 2.
TREND_RATE_BLOCK_YEARS = 3
```

- [ ] **Step 2: Write the failing tests**

In `src/models/market/test/test_level_volume_model.py`, replace lines 5-14 (the `from src.models.market.level_volume_model import (...)` block — lines 1-3 are plain `numpy`/`pandas`/`pytest` imports and stay untouched):

```python
from src.models.market.level_volume_model import (
    fit_level_volume_model,
    predict_level_volume,
    score_mape,
    build_complete_monthly_series,
    fit_level_only_model,
    predict_level_only,
    fit_trend_rate_model,
    predict_trend_correction,
)
```

(This is the current import list with `fit_level_volume_trend_model, predict_level_volume_trend` swapped out for `fit_trend_rate_model, predict_trend_correction` — nothing else changes.)

Then replace lines 132-218 (everything from `def _make_trending_monthly` through the end of `test_trend_fit_raises_when_volume_has_zero_variance`) with:

```python
def _make_trending_monthly(n_years=10, slope=0.1, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_years * 12, freq="MS")
    years = dates.year.to_numpy(dtype=float)
    price = 2.0 + slope * (years - years.min()) + rng.normal(0, 0.01, len(dates))
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": [1000.0] * len(dates)})


def test_fit_trend_rate_model_measures_a_positive_rate_from_rising_blocks():
    monthly = _make_trending_monthly(n_years=10, slope=0.1, seed=0)
    train_end = pd.Timestamp("2025-01-01")

    params = fit_trend_rate_model(monthly, train_end)

    assert params["rate"] == pytest.approx(0.0423, abs=0.001)
    assert params["years_trained"] == pytest.approx(10.0, abs=0.01)


def test_fit_trend_rate_model_returns_zero_rate_with_fewer_than_two_blocks():
    # 5 years of history and 3-year blocks gives exactly one full block
    # (years 0-3) plus a 2-year remainder too short for a second block --
    # a least-squares line needs at least 2 points, so this measures a
    # harmless rate of 0.0 rather than raising or crashing.
    monthly = _make_trending_monthly(n_years=5, slope=0.1, seed=0)
    train_end = pd.Timestamp("2020-01-01")

    params = fit_trend_rate_model(monthly, train_end)

    assert params["rate"] == 0.0
    assert params["years_trained"] == pytest.approx(5.0, abs=0.01)


def test_fit_trend_rate_model_handles_empty_history():
    monthly = pd.DataFrame({"date": pd.to_datetime([]), "price": [], "landings_kg": []})
    params = fit_trend_rate_model(monthly, pd.Timestamp("2020-01-01"))
    assert params == {"rate": 0.0, "years_trained": 0.0}


def test_predict_trend_correction_is_identity_at_and_before_train_end():
    monthly = _make_trending_monthly(n_years=10, slope=0.1, seed=0)
    train_end = pd.Timestamp("2025-01-01")
    rate_params = fit_trend_rate_model(monthly, train_end)
    params = {
        "trend_rate": rate_params["rate"],
        "trend_years_trained": rate_params["years_trained"],
        "trend_confidence": 1.0,
        "trend_train_end": train_end,
    }

    at_or_before = pd.DataFrame({
        "date": [train_end, train_end - pd.DateOffset(years=3)],
        "price": [None, None],
        "landings_kg": [1000.0, 1000.0],
    })
    correction = predict_trend_correction(at_or_before, params)
    assert correction.iloc[0] == pytest.approx(1.0)
    assert correction.iloc[1] == pytest.approx(1.0)


def test_predict_trend_correction_grows_forward_and_clamps_at_years_trained():
    monthly = _make_trending_monthly(n_years=10, slope=0.1, seed=0)
    train_end = pd.Timestamp("2025-01-01")
    rate_params = fit_trend_rate_model(monthly, train_end)
    params = {
        "trend_rate": rate_params["rate"],
        "trend_years_trained": rate_params["years_trained"],
        "trend_confidence": 1.0,
        "trend_train_end": train_end,
    }

    one_year_out = pd.DataFrame(
        {"date": [train_end + pd.DateOffset(years=1)], "price": [None], "landings_kg": [1000.0]}
    )
    far_future = pd.DataFrame(
        {"date": [train_end + pd.DateOffset(years=20)], "price": [None], "landings_kg": [1000.0]}
    )
    at_years_trained_bound = pd.DataFrame(
        {
            "date": [train_end + pd.DateOffset(years=int(rate_params["years_trained"]))],
            "price": [None],
            "landings_kg": [1000.0],
        }
    )

    corr_1yr = predict_trend_correction(one_year_out, params).iloc[0]
    corr_far = predict_trend_correction(far_future, params).iloc[0]
    corr_bound = predict_trend_correction(at_years_trained_bound, params).iloc[0]

    assert corr_1yr > 1.0  # positive rate -> correction grows forward
    assert corr_far == pytest.approx(corr_bound, rel=1e-6)  # clamped, doesn't keep extrapolating


def test_predict_trend_correction_is_a_no_op_at_zero_confidence():
    monthly = _make_trending_monthly(n_years=10, slope=0.1, seed=0)
    train_end = pd.Timestamp("2025-01-01")
    rate_params = fit_trend_rate_model(monthly, train_end)
    params = {
        "trend_rate": rate_params["rate"],
        "trend_years_trained": rate_params["years_trained"],
        "trend_confidence": 0.0,
        "trend_train_end": train_end,
    }

    far_future = pd.DataFrame(
        {"date": [train_end + pd.DateOffset(years=5)], "price": [None], "landings_kg": [1000.0]}
    )
    assert predict_trend_correction(far_future, params).iloc[0] == pytest.approx(1.0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest src/models/market/test/test_level_volume_model.py -v`
Expected: FAIL at collection (`ImportError: cannot import name 'fit_trend_rate_model'`)

- [ ] **Step 4: Replace the old trend fit/predict functions**

In `src/models/market/level_volume_model.py`, replace lines 143-229 (`fit_level_volume_trend_model` and `predict_level_volume_trend` in their entirety) with:

```python
def fit_trend_rate_model(monthly: pd.DataFrame, train_end: pd.Timestamp) -> dict:
    """
    Trend rate (spec 2026-10-08-combined-market-trend-term-v2-design.md
    section 3, Step 3). Builds sequential TREND_RATE_BLOCK_YEARS-year
    blocks forward from the oldest available real price observation,
    stopping once the next block would extend past train_end (any
    partial remainder nearest train_end is simply unused -- this matches
    the spec's own worked example: ES's ~10yr history yields exactly 3
    blocks, not 3.33). Each block's mean log-price is one data point; rate
    is the slope of a straight line through those points
    (numpy.polyfit), with the x-axis in years relative to train_end
    (negative for historical blocks, matching predict_trend_correction's
    trend_years convention exactly). Completely independent of window
    selection -- this function never reads or influences which window
    CombinedMarket.fit() picks.

    Returns {"rate": 0.0, ...} (a harmless no-op, not a raised error) when
    fewer than 2 blocks fit -- a straight line needs at least 2 points,
    and real history shorter than 2*TREND_RATE_BLOCK_YEARS years is a
    normal, expected case (e.g. a newly-onboarded species/market), not a
    fitting failure.
    """
    history = monthly.dropna(subset=["price"]).copy()
    history = history[history["date"] < train_end].sort_values("date")
    if history.empty:
        return {"rate": 0.0, "years_trained": 0.0}

    history["log_price"] = np.log(history["price"])
    block_days = pd.Timedelta(days=consts.TREND_RATE_BLOCK_YEARS * 365.25)

    oldest = history["date"].min()
    years_trained = (train_end - oldest).days / 365.25

    block_centers = []
    block_averages = []
    block_start = oldest
    while block_start + block_days <= train_end:
        block_end = block_start + block_days
        block = history[(history["date"] >= block_start) & (history["date"] < block_end)]
        if len(block) > 0:
            center_date = block_start + (block_end - block_start) / 2
            block_centers.append((center_date - train_end).days / 365.25)
            block_averages.append(block["log_price"].mean())
        block_start = block_end

    if len(block_centers) < 2:
        return {"rate": 0.0, "years_trained": max(years_trained, 0.0)}

    rate, _ = np.polyfit(block_centers, block_averages, 1)
    return {"rate": float(rate), "years_trained": float(years_trained)}


def predict_trend_correction(monthly: pd.DataFrame, params: dict) -> pd.Series:
    """
    Multiplicative trend correction (spec v2 section 3, Step 4):
    exp(confidence * rate * clip(trend_years, 0, years_trained)). Returns
    a correction FACTOR, not a price -- callers multiply this onto
    predict_level_volume's (or predict_level_only's) own output.
    trend_years is computed from params["trend_train_end"], mirroring
    v1's predict_level_volume_trend, but clamped to [0, years_trained],
    NOT symmetric -- the v2 spec's section 3 Step 4 explains why this
    differs from v1's symmetric clamp fix: v1's gamma was fit via
    regression directly against unclamped, negative trend_years values
    for training rows, so an asymmetric predict-time clamp would have
    been inconsistent with what that fit assumed. Here, the rate is
    measured purely from block averages with no per-row regression
    against trend_years at all, so there is no equivalent fitting-time
    assumption to be inconsistent with -- any date at or before
    train_end gets a correction of exactly 1.0 (no-op), matching that
    every historical date is covered by get_price's real-price-first path
    before this model is ever reached anyway.
    """
    trend_years = (monthly["date"] - params["trend_train_end"]).dt.days / 365.25
    trend_years = trend_years.clip(lower=0.0, upper=params["trend_years_trained"])
    correction = np.exp(params["trend_confidence"] * params["trend_rate"] * trend_years)
    correction.index = monthly.index
    return correction
```

Also update the module docstring's sentence that currently reads "Adapted from the validated fit_level_volume_model/predict_level_volume functions..." — no change needed there, but double check no other comment in this file still refers to `gamma`, `fit_level_volume_trend_model`, or `predict_level_volume_trend` by name after this edit (a stale comment pointing at a deleted function is worse than no comment).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest src/models/market/test/test_level_volume_model.py -v`
Expected: PASS (all tests, old and new — `fit_level_only_model`/`predict_level_only`/`fit_level_volume_model`/`predict_level_volume`'s own existing tests are untouched by this task and must keep passing)

- [ ] **Step 6: Commit**

```bash
git add src/models/market/level_volume_model.py src/models/market/consts.py src/models/market/test/test_level_volume_model.py
git commit -m "feat: replace the lstsq trend regression with a block-based rate"
```

---

### Task 3: Wire trend params into `CombinedMarket.fit()` unconditionally

**Files:**
- Modify: `src/models/market/CombinedMarket.py:1-150` (imports; `fit()` method; new `_attach_trend_params` helper)
- Test: `src/models/market/test/test_CombinedMarket.py` (replace `test_fit_includes_trend_for_a_key_that_passes_the_gate`/`test_fit_excludes_trend_for_a_key_that_fails_the_gate`; add the new baseline-identity regression test)

**Interfaces:**
- Consumes: `compute_trend_confidence` (Task 1), `fit_trend_rate_model` (Task 2).
- Produces: every key in `self.models` (both the main level-volume path and the level-only fallback) now carries `"trend_confidence"`, `"trend_rate"`, `"trend_years_trained"`, `"trend_train_end"` — read by Task 4's `get_price()`. Window selection and `baseline_price`/`alpha`/`beta` are byte-for-byte what `fit_level_volume_model`/`predict_level_volume` alone would have produced, for every key, with no exceptions.

- [ ] **Step 1: Write the failing tests**

Append to `src/models/market/test/test_CombinedMarket.py` (it already has `_synthetic_monthly` and `_trending_synthetic_monthly` defined near the top of the file and near line 225 respectively — reuse both, don't redefine):

```python
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
```

(`test_CombinedMarket.py` already imports `pytest` at the top of the file — no import changes needed here.)

Now replace the existing `test_fit_includes_trend_for_a_key_that_passes_the_gate` and `test_fit_excludes_trend_for_a_key_that_fails_the_gate` (they assert `"gamma" in market.models[...]`, which no longer exists) — delete both; `test_fit_attaches_trend_params_to_every_key` above is their replacement.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest src/models/market/test/test_CombinedMarket.py -v -k "attaches_trend_params or identical_regardless_of_trend"`
Expected: FAIL — `test_fit_attaches_trend_params_to_every_key` fails with a `KeyError`/assertion failure (`fit()` doesn't attach these fields yet); `test_window_and_baseline_identical_regardless_of_trend_confidence` fails at the `monkeypatch.setattr` call with `AttributeError: <module> does not have the attribute 'compute_trend_confidence'` (not imported into `CombinedMarket.py` yet).

- [ ] **Step 3: Update imports**

In `src/models/market/CombinedMarket.py`, replace lines 16-27:

```python
from . import consts
from .level_volume_model import (
    fit_level_volume_model,
    predict_level_volume,
    fit_level_volume_trend_model,
    predict_level_volume_trend,
    fit_level_only_model,
    predict_level_only,
    score_mape,
)
from .trend_gate import passes_trend_gate
from .window_selection import build_rolling_splits, select_best_window
```

with:

```python
from . import consts
from .level_volume_model import (
    fit_level_volume_model,
    predict_level_volume,
    fit_trend_rate_model,
    predict_trend_correction,
    fit_level_only_model,
    predict_level_only,
    score_mape,
)
from .trend_confidence import compute_trend_confidence
from .window_selection import build_rolling_splits, select_best_window
```

(`predict_trend_correction` is imported here already even though Task 4 is what actually calls it in `get_price()` — this avoids a second import-editing step in Task 4.)

- [ ] **Step 4: Rewrite `fit()`'s per-key loop**

Replace the entire `fit()` method body (currently lines 54-136, from `def fit(` through its closing `return selected`) with:

```python
    def fit(
        self, monthly_by_key: dict[tuple[str, str], pd.DataFrame], latest_date: pd.Timestamp
    ) -> dict[tuple[str, str], str | None]:
        """
        Fit and freeze Model 2 for every (species_code, category_code),
        selecting the best training-window length independently per key
        (spec section 7). Falls back to a level-only fit (spec section 9
        point 2) if no window produces a stable level-volume model; excludes
        the key entirely (spec section 9 point 4) if even that fails.

        Window selection and the frozen baseline (baseline_price/alpha/beta)
        are completely unconditional -- identical for every key, with no
        trend-related branching anywhere in this path (spec
        2026-10-08-combined-market-trend-term-v2-design.md section 3, Step
        1 / Global Constraints). Trend confidence and rate (Steps 2-3) are
        computed separately from the complete monthly series and attached
        to the frozen params afterward via _attach_trend_params -- for
        every key that gets a model at all, main path or fallback alike --
        so get_price() never needs to branch on whether a key "has" a
        trend; the correction it applies is just ~1.0 (a no-op) for a
        non-trending key.
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

            if result is not None:
                label, years = result
                train_start = latest_date - relativedelta(years=years) if years is not None else None
                try:
                    params = fit_level_volume_model(monthly, train_start, latest_date)
                    self._attach_trend_params(params, monthly, latest_date)
                    self.models[key] = params
                    self.window_labels[key] = label
                    self._seed_volume_baseline_stats(key, monthly, train_start, latest_date)
                    selected[key] = label
                    continue
                except (ValueError, np.linalg.LinAlgError):
                    # The window length scored acceptably across historical rolling-origin
                    # splits (each with its own, earlier train_end), but real volume data is
                    # sparse enough that the literal final window (train_start..latest_date)
                    # can still fail to fit -- e.g. every non-null landings_kg reading in this
                    # exact window happens to be the sole representative of its calendar
                    # month, making the volume-anomaly series degenerate (zero variance).
                    # Cross-validation generalizing to historical splits doesn't guarantee the
                    # production window itself is fittable, so fall through to the same
                    # lower-tier fallback used when no window was selected at all.
                    self.logger.warning(
                        f"CombinedMarket {self.market_code}: selected window {label} for {key} "
                        f"scored acceptably in rolling-origin validation but failed to re-fit on "
                        f"the final window ending at {latest_date}; falling back to level-only fit"
                    )

            # Fallback cascade (spec section 9 point 2): try level-only.
            else:
                self.logger.warning(
                    f"CombinedMarket {self.market_code}: no stable level-volume window found "
                    f"for {key}, falling back to level-only fit"
                )
            try:
                params = fit_level_only_model(monthly, None, latest_date)
                self._attach_trend_params(params, monthly, latest_date)
                self.models[key] = params
                self.window_labels[key] = "level-only-fallback"
                selected[key] = "level-only-fallback"
            except ValueError:
                self.logger.warning(
                    f"CombinedMarket {self.market_code}: no usable price history at all "
                    f"for {key}, excluding from output"
                )
                selected[key] = None

        return selected

    def _attach_trend_params(self, params: dict, monthly: pd.DataFrame, train_end: pd.Timestamp) -> None:
        """
        spec 2026-10-08-combined-market-trend-term-v2-design.md section 3,
        Steps 2-4: continuous confidence + block-based rate, computed from
        the complete real historical record and attached unconditionally
        -- same procedure for every key, main level-volume path and the
        level-only fallback alike, so get_price() can apply the correction
        without ever checking which path produced a given key's params.
        """
        trend_rate_params = fit_trend_rate_model(monthly, train_end)
        params["trend_confidence"] = compute_trend_confidence(monthly)
        params["trend_rate"] = trend_rate_params["rate"]
        params["trend_years_trained"] = trend_rate_params["years_trained"]
        params["trend_train_end"] = train_end
```

Note what did NOT change: the `except (ValueError, np.linalg.LinAlgError)` fallback-to-level-only cascade, the final `fit_level_only_model` call's own `except ValueError`, `_seed_volume_baseline_stats`, and every log message are identical to today's code — only the `trend_eligible` branch is gone and `_attach_trend_params` is called in the two places a `params` dict is produced.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest src/models/market/test/test_CombinedMarket.py -v`
Expected: Most tests PASS. `test_get_price_uses_trend_augmented_prediction_when_model_has_trend` and `test_get_baseline_price_table_unaffected_by_trend_term` are expected to FAIL here (`"gamma" in params` is never true anymore) — that's Task 4's job to fix; confirm no OTHER test regresses.

- [ ] **Step 6: Commit**

```bash
git add src/models/market/CombinedMarket.py src/models/market/test/test_CombinedMarket.py
git commit -m "feat: attach continuous trend confidence/rate unconditionally in CombinedMarket.fit()"
```

---

### Task 4: Apply the trend correction unconditionally in `get_price()`

**Files:**
- Modify: `src/models/market/CombinedMarket.py` (the `get_price` method)
- Test: `src/models/market/test/test_CombinedMarket.py` (replace the two get_price-related trend tests; update two preconditions)

**Interfaces:**
- Consumes: `predict_trend_correction` (Task 2, already imported in Task 3's Step 3), the `trend_*` fields `_attach_trend_params` (Task 3) attaches to every key.
- Produces: `get_price()` always returns `predict_level_only(...)` or `predict_level_volume(...)` multiplied by `predict_trend_correction(...)` — no branch on whether a key "has" a trend.

- [ ] **Step 1: Write the failing tests**

In `src/models/market/test/test_CombinedMarket.py`, replace `test_get_price_uses_trend_augmented_prediction_when_model_has_trend` with:

```python
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
```

Then update the precondition assertions in the two tests immediately below (`test_get_baseline_price_table_unaffected_by_trend_term` and `test_get_live_anchor_series_unaffected_by_trend_term`): both currently contain `assert "gamma" in params  # precondition: this is the trend-augmented path`. Replace that line in `test_get_baseline_price_table_unaffected_by_trend_term` with:

```python
    assert params["trend_confidence"] == pytest.approx(1.0)  # precondition: this is the high-confidence case
```

`test_get_live_anchor_series_unaffected_by_trend_term`'s assertions (`live_series is not None`, `len(live_series) == len(monthly)`) never change — only its comment references stale v1 terminology. Replace that test's comment (currently):

```python
    # Analysis-only path (spec: "never used by the live production path").
    # Confirm it still returns a result shaped the same way for a
    # trend-augmented model -- it dispatches on baseline_volume is None,
    # not on "gamma", so a trend-augmented model (which always has
    # baseline_volume set) must still take the predict_level_volume branch
    # here, not predict_level_volume_trend -- this function was not
    # touched by this plan, confirm that directly.
```

with:

```python
    # Analysis-only path (spec: "never used by the live production path").
    # Confirm it still returns a result shaped the same way for a
    # trend-bearing key -- it dispatches only on baseline_volume is None,
    # never reads trend_confidence/trend_rate at all, so it always takes
    # the plain predict_level_volume branch with no trend correction
    # applied -- this function was not touched by this plan, confirm that
    # directly.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest src/models/market/test/test_CombinedMarket.py -v -k "trend_correction or unaffected_by_trend or get_price_is_unaffected"`
Expected: `test_get_price_applies_trend_correction_proportional_to_confidence` FAILS (`get_price` doesn't apply any correction yet, so `price_far_future == price_near_fit_end` exactly). `test_get_baseline_price_table_unaffected_by_trend_term` FAILS at the updated precondition line only if `trend_confidence` isn't attached — it should already pass given Task 3; if it does fail here, Task 3 wasn't fully applied, stop and check. `test_get_price_is_unaffected_by_trend_for_a_flat_series` should already PASS (no correction applied yet at all, so trivially unaffected) — that's fine, it becomes a meaningful test once Step 3 below is done.

- [ ] **Step 3: Update `get_price` to apply the correction unconditionally**

The current `get_price` method reads:

```python
        params = self.models[key]
        monthly = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
        if params.get("baseline_volume") is None:
            predicted = predict_level_only(monthly, params)
        elif "gamma" in params:
            predicted = predict_level_volume_trend(monthly, params)
        else:
            predicted = predict_level_volume(monthly, params)
        value = predicted.iloc[0]
        return float(value) if pd.notna(value) else None
```

Change it to:

```python
        params = self.models[key]
        monthly = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
        if params.get("baseline_volume") is None:
            predicted = predict_level_only(monthly, params)
        else:
            predicted = predict_level_volume(monthly, params)
        predicted = predicted * predict_trend_correction(monthly, params)
        value = predicted.iloc[0]
        return float(value) if pd.notna(value) else None
```

Also update this method's docstring (currently just `"""Real-price-first (spec section 8): check price_time_series, else predict."""`) to:

```python
        """
        Real-price-first (spec section 8): check price_time_series, else
        predict. The trend correction (spec v2 section 3, Step 4) is
        applied unconditionally -- it is a pure multiplier of ~1.0 for a
        key with near-zero trend confidence, so there is no branch here on
        whether a key "has" a trend, unlike v1's "gamma" in params check.
        """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest src/models/market/test/test_CombinedMarket.py -v`
Expected: PASS (all tests in this file)

- [ ] **Step 5: Commit**

```bash
git add src/models/market/CombinedMarket.py src/models/market/test/test_CombinedMarket.py
git commit -m "feat: apply the trend correction unconditionally in get_price()"
```

---

### Task 5: Real-data integration test pinning spec section 4's measured values

**Files:**
- Modify: `src/models/market/test/test_CombinedMarket.py:315-365` (replace `test_real_data_selects_trend_only_for_ane_at_es`)

**Interfaces:**
- Consumes: the real `config/northwestern_med/prices.csv` data, via the existing `_build_monthly_from_real_prices` helper (already defined in this file, unchanged).
- Produces: no new production code — a pinned regression test against spec section 4's table.

- [ ] **Step 1: Replace the test**

Replace `test_real_data_selects_trend_only_for_ane_at_es` (and the `expected_trend = {("ES", "ANE")}` line immediately above it) with:

```python
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
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest src/models/market/test/test_CombinedMarket.py -v -k test_real_data_trend_params_match_measured_values`
Expected: PASS. If `trend_confidence` or `trend_rate` falls outside tolerance for any pair, do not loosen the tolerance reflexively — first double check `fit_trend_rate_model`'s block-boundary logic against Task 2's spec (forward from oldest, stop before exceeding `train_end`) and `compute_trend_confidence` against Task 1's spec; a real logic bug (e.g. off-by-one block boundary, wrong majority-sign tie-break) is more likely than the original spike's numbers being wrong, since those were independently verified against this same real dataset earlier in this project. Only widen a tolerance if you've confirmed the logic matches the spec exactly and the discrepancy is a legitimate, explainable implementation-detail difference (document which, and why, in the test's comment if you do).

If `config/northwestern_med/prices.csv` doesn't exist in this environment, this test is already wrapped in the file's existing `@pytest.mark.skipif(not os.path.exists(...))` decorator — confirm that decorator still sits directly above the new test (copy it from directly above the old `test_real_data_selects_trend_only_for_ane_at_es`, it isn't shown in the replacement text above).

- [ ] **Step 3: Commit**

```bash
git add src/models/market/test/test_CombinedMarket.py
git commit -m "test: pin real-data trend confidence/rate to spec section 4's measured table"
```

---

### Task 6: Update the CombinedMarket fidelity backtest

**Files:**
- Modify: `src/models/market/test/test_combined_market_fidelity.py`

**Interfaces:**
- Consumes: `run_backtest_split` (unchanged, from `fidelity_backtest_harness.py`).
- Produces: no new production code — confirms the v2 mechanism is a real net accuracy improvement (same `MAX_ACCEPTABLE_MAPE` ceiling as before) and replaces the v1 boolean `"gamma" in params` trend-bearing check with its continuous equivalent.

- [ ] **Step 1: Update the stale comment naming v1's deleted gate**

This file's `test_combined_market_accuracy_within_acceptable_mape` function opens with a comment block (currently):

```python
    # as_of_date/test_horizon_years deliberately differ from
    # test_fidelity_backtest.py's own 2022-01-01/2yr split: the trend gate
    # (passes_trend_gate) needs ~9+ years of real history to reach unanimous
    # month-sign agreement for ANE@ES -- confirmed directly that a
    # 2022-01-01 cutoff (7yr training history) leaves the gate False (one
    # dissenting calendar month) and so never exercises the trend term at
    # all anywhere in the backtest. Cutting over at 2024-01-01 instead (9yr
    # training history, the gate's own empirically-confirmed activation
    # point) leaves a full 12 months of real, still-genuinely-held-out data
    # (2024-01 through 2024-12, the real dataset's final year) for
    # comparison, while actually letting ANE@ES pick up its trend term --
    # see the assertion below, which confirms this directly rather than
    # assuming it.
```

This names `passes_trend_gate`, a function Task 1 already deleted, and describes pass/fail gate mechanics (unanimous sign agreement, a boolean "gate False") that no longer exist. Replace it with:

```python
    # as_of_date/test_horizon_years deliberately differ from
    # test_fidelity_backtest.py's own 2022-01-01/2yr split: ANE@ES's trend
    # confidence (compute_trend_confidence) needs ~9+ years of real history
    # before individual calendar months start reaching statistical
    # significance -- confirmed directly that a 2022-01-01 cutoff (7yr
    # training history) leaves confidence too low to exercise the trend
    # correction meaningfully anywhere in the backtest. Cutting over at
    # 2024-01-01 instead (9yr training history) leaves a full 12 months of
    # real, still-genuinely-held-out data (2024-01 through 2024-12, the
    # real dataset's final year) for comparison, while actually letting
    # ANE@ES's trend correction activate -- see the assertion below, which
    # confirms this directly rather than assuming it.
```

- [ ] **Step 2: Update the trend-bearing assertion**

Replace the `trend_bearing` block (currently):

```python
    # Confirm the trend term is actually active for ANE@ES in this backtest
    # (not merely assumed) -- and nowhere else, matching Task 5's real-data
    # pin (test_CombinedMarket.py::test_real_data_selects_trend_only_for_ane_at_es).
    # Without this, a future change to the gate or the data could silently
    # revert this test to measuring ANE@ES's no-trend baseline again (as
    # happened with the original 2022-01-01/2yr split) without anyone
    # noticing, defeating this test's whole purpose.
    trend_bearing = {
        (market_code, species)
        for market_code, combined_market in handler.market_system.combined_markets.items()
        for species, category in combined_market.models
        if category == CATEGORY and "gamma" in combined_market.models[(species, category)]
    }
    assert trend_bearing == {("ES", "ANE")}, (
        f"expected the trend term to be active for ES/ANE only, got: {trend_bearing}"
    )
```

with:

```python
    # Confirm ES/ANE's trend confidence is active and clearly the
    # standout case in this backtest (not merely assumed) -- matching Task
    # 5's real-data pin (test_CombinedMarket.py::test_real_data_trend_params_match_measured_values).
    # Confidence is now continuous (never a hard "has/doesn't have"
    # boolean), so this checks the shape that matters: ES/ANE measures
    # meaningfully higher confidence than every other validated pair.
    # Without this, a future change to the confidence math or the data
    # could silently stop exercising the trend correction at all in this
    # backtest without anyone noticing, defeating this test's whole purpose.
    confidences = {
        (market_code, species): combined_market.models[(species, category)]["trend_confidence"]
        for market_code, combined_market in handler.market_system.combined_markets.items()
        for species, category in combined_market.models
        if category == CATEGORY
    }
    assert confidences[("ES", "ANE")] > 0.5, (
        f"expected ES/ANE's trend confidence to be clearly active (>0.5) in this backtest, "
        f"got {confidences[('ES', 'ANE')]}"
    )
    for pair, confidence in confidences.items():
        if pair != ("ES", "ANE"):
            assert confidence < confidences[("ES", "ANE")], (
                f"expected ES/ANE to have the highest trend confidence of any validated pair "
                f"in this backtest, but {pair} measured {confidence} >= ES/ANE's {confidences[('ES', 'ANE')]}"
            )
```

Do not change anything else in this file — `MAX_ACCEPTABLE_MAPE`, the `as_of_date`/`test_horizon_years` split, and the rest of the MAPE-computation logic all stay exactly as they are; this task is only replacing the boolean trend-bearing check (and the stale comment from Step 1) with their continuous equivalents.

- [ ] **Step 3: Run the test**

Run: `python -m pytest src/models/market/test/test_combined_market_fidelity.py -v -s`
Expected: PASS, including the `MAX_ACCEPTABLE_MAPE` ceiling (50.0) for every (species, market) pair — printed MAPE table should look similar in shape to before (ES/ANE meaningfully improved vs. a no-trend baseline, nothing else regressed). If any pair exceeds the ceiling, this is a real accuracy regression from the rewrite (not just a test-wiring issue) — treat it with the same seriousness systematic-debugging would: find the root cause in `fit_trend_rate_model`/`predict_trend_correction`/`compute_trend_confidence` before touching the ceiling itself.

- [ ] **Step 4: Commit**

```bash
git add src/models/market/test/test_combined_market_fidelity.py
git commit -m "test: update CombinedMarket fidelity backtest for continuous trend confidence"
```

---

### Task 7: Visual confirmation via the 20-year baseline scenario

**Files:**
- Create (throwaway, delete after use): `src/models/market/test/_verify_trend_v2_baseline.py`

**Interfaces:**
- Consumes: the real `MarketHandler`/`CombinedMarket` with Tasks 1-4 applied.
- Produces: no new production code — visual/numeric confirmation that the 20-year baseline-scenario forecast matches the shape already verified during this feature's own design investigation (spec section 3's closing paragraph): ANE@ES climbing smoothly from ~3.09 (2025) to ~3.88 (2029), and SAA@FR rising gently from ~0.733 to ~0.750 over the same period, rather than either flatlining (the original complaint that motivated this whole feature) or blowing up (v1's full-strength-regression failure mode this design replaced).

- [ ] **Step 1: Write and run a throwaway 20-year baseline simulation**

Create `src/models/market/test/_verify_trend_v2_baseline.py`:

```python
"""
THROWAWAY -- visual/numeric confirmation (spec
2026-10-08-combined-market-trend-term-v2-design.md section 6) that the
20-year baseline forecast matches the shape verified during this
feature's design investigation: ANE@ES climbs smoothly post-2024 instead
of flatlining or blowing up, SAA@FR rises gently instead of exploding
(v1's failure mode), and every other validated species/market scales
proportional to its own measured confidence. Delete after use.
"""
import pandas as pd

from src.models.market.MarketHandler import MarketHandler

CONFIG_DIR = "config"
SCENARIO_ID = "northwestern_med"
VALIDATED_SPECIES = ["ANE", "PIL", "HKE", "SAA", "MUT"]
CATEGORY = "Fresh - Whole"
N_MONTHS = 240


class _FakeMarket:
    def __init__(self, market_code):
        self.market_code = market_code


class _FakeSpecies:
    def __init__(self, species_code):
        self.species_code = species_code


class _FakePriceCategory:
    def __init__(self, category_code):
        self.category_code = category_code


class _FakeItems:
    def __init__(self, markets, species, price_categories):
        self.markets = markets
        self.species = species
        self.price_categories = price_categories


class _FakeSimulation:
    def __init__(self, items):
        self.items = items


def test_verify_trend_v2_in_baseline_forecast():
    markets_df = pd.read_csv(f"{CONFIG_DIR}/{SCENARIO_ID}/markets.csv")
    local_codes = markets_df[markets_df["market_code"].str.len() == 5]["market_code"].tolist()

    prices = pd.read_csv(f"{CONFIG_DIR}/{SCENARIO_ID}/prices.csv", low_memory=False)
    prices["date"] = pd.to_datetime(prices["date"])
    if "size" not in prices.columns:
        prices["size"] = None
    blended = prices[prices["size"].isna()]
    start_date = blended["date"].min()
    blended = blended[blended["category_code"] == CATEGORY]
    monthly_mean = (
        blended[blended["species_code"].isin(VALIDATED_SPECIES)]
        .assign(month=lambda d: d["date"].dt.month)
        .groupby(["market_code", "species_code", "month"])["landings_kg"]
        .mean()
    )

    simulation = _FakeSimulation(_FakeItems(
        markets=[_FakeMarket(c) for c in local_codes],
        species=[_FakeSpecies(s) for s in VALIDATED_SPECIES],
        price_categories=[_FakePriceCategory(CATEGORY)],
    ))
    handler = MarketHandler("verify-trend-v2", SCENARIO_ID, start_date.to_pydatetime(), 1)
    response, err = handler.init_simulation(simulation, CONFIG_DIR)
    assert err is None
    market_system = handler.market_system

    print(f"\n{'='*90}\nConfidence/rate per species/market (production fit)\n{'='*90}")
    for market_code, combined_market in market_system.combined_markets.items():
        for species in VALIDATED_SPECIES:
            key = (species, CATEGORY)
            if key not in combined_market.models:
                continue
            params = combined_market.models[key]
            print(
                f"{market_code}/{species}: window={combined_market.window_labels[key]} "
                f"confidence={params['trend_confidence']:.2f} rate={params['trend_rate']:+.4f}"
            )

    print(f"\n{'='*90}\n20-year baseline forecast, January of each year\n{'='*90}")
    for month_index in range(N_MONTHS):
        current_date = market_system.tick.get_datetime()
        cal_month = current_date.month
        for port in local_codes:
            local_market = market_system.local_markets.get(port)
            if local_market is None:
                continue
            for species in VALIDATED_SPECIES:
                base_volume = monthly_mean.get((port, species, cal_month))
                if base_volume is None or pd.isna(base_volume):
                    continue
                local_market.add_monthly_catch(species, CATEGORY, base_volume)
        market_system.step()

    for market_code, combined_market in market_system.combined_markets.items():
        for species in VALIDATED_SPECIES:
            key = (species, CATEGORY)
            if key not in combined_market.models:
                continue
            print(f"\n{market_code}/{species}:")
            for tick_id, snapshot in combined_market.price_history:
                tick_date = start_date + pd.DateOffset(months=tick_id)
                if tick_date.month == 1:
                    price = snapshot["prices"].get(key)
                    print(f"  {tick_date.strftime('%Y-%m')}: {price}")
```

This reuses v1's own validated Task 7 verification harness verbatim (same `MarketHandler`/fake-simulation setup, same `local_market.add_monthly_catch(...)` + `market_system.step()` per-port-per-month driving loop, confirmed against the current `LocalMarket`/`MarketSystem` API: `add_monthly_catch` exists on `LocalMarket`, and `market_system.tick.get_datetime()` exists via `MarketSystem`'s own `Tick` instance) — only the printed summary (confidence/rate table, per-species January prices) is new, to match this task's own pass/fail criteria below.

Run: `python -m pytest src/models/market/test/_verify_trend_v2_baseline.py -v -s`

Confirm from the printed output:
- ES/ANE's January price climbs from roughly 3.09 (2025) up toward roughly 3.9 by 2029, continuing to climb (not flatlining) through 2044.
- FR/SAA's (or ES/SAA's) January price rises gently year over year — nothing resembling v1's 99%+ MAPE blowup (i.e., no explosive, order-of-magnitude price swings).
- ES/PIL and ES/HKE (near-zero confidence) stay essentially flat across the full 20 years.

If any of these don't hold, this is a signal of a real problem in Tasks 1-4's logic (most likely `fit_trend_rate_model`'s block construction or `predict_trend_correction`'s clamp), not something to patch here — stop and investigate per `systematic-debugging` before moving on.

- [ ] **Step 2: Delete the throwaway script**

```bash
rm src/models/market/test/_verify_trend_v2_baseline.py
git status --porcelain
```

Confirm `git status --porcelain` shows nothing for this file (it was never staged/committed) — this project's established discipline for throwaway verification scripts (never committed, deleted immediately after use).

---

## Final check

After all 7 tasks: run the full suite once (`python -m pytest -q`) to confirm nothing elsewhere in the project (in particular `LocalMarket`'s own tests and the existing `test_fidelity_backtest.py`) regressed. Per the spec's Global Constraints, `LocalMarket`/Step 2/`stacked_model.py` should show zero diff in behavior — this run confirms that empirically rather than by inspection alone.

---

## Self-Review Notes (for the plan author, not a task)

- Spec coverage: Step 1 (Task 3/4), Step 2 (Task 1), Step 3 (Task 2), Step 4 (Task 2/4), section 4's table (Task 5), section 5's "every key, not gated" requirement (Task 3), section 6's new regression test (Task 3), section 7's out-of-scope items (untouched by every task — no task here modifies `LocalMarket`, `stacked_model.py`, or `window_selection.py`).
- `trend_rate`/`trend_confidence`/`trend_years_trained`/`trend_train_end` are the four field names used consistently from Task 2 through Task 6 — no renaming partway through.
- `get_live_anchor_series` is deliberately never modified by any task (matches v1's own precedent — analysis-only, never used by the live production path) — Task 4 only updates that function's existing test's comment wording, not its assertions or the function itself.
