# CombinedMarket Trend Term Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `CombinedMarket`'s frozen Model 2 represent a genuine, sustained secular trend for whichever `(species_code, category_code)` key demonstrably has one, gated by direct evidence from the full real historical record, with zero effect on every key that doesn't pass the gate.

**Architecture:** A new, standalone gate function (`trend_gate.py`) decides per-key eligibility from raw historical data alone (no model fitting involved). Eligible keys get fit with a new trend-augmented sibling of `fit_level_volume_model`/`predict_level_volume` (added to `level_volume_model.py`), selected via the *same, unmodified* window-length selection machinery already in production. `CombinedMarket.fit()`/`get_price()` dispatch between the plain and trend-augmented paths; every non-eligible key's code path is untouched, byte-for-byte.

**Tech Stack:** Python, numpy (`polyfit`, `linalg.lstsq`), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-combined-market-trend-term-design.md`

## Global Constraints

- **Scope: `CombinedMarket` only.** Do not touch `LocalMarket.fit_step1`/`fit_step2` or `stacked_model.py` in any task below.
- **`LocalMarket` Step 2's static anchor is unaffected by design** — `get_baseline_price_table` only ever reads `baseline_price`, never the regression coefficients. Task 4 verifies this with a test rather than trusting it.
- **Frozen-forever still holds** — the trend term is fit once at `init_simulation` time, from whatever real history precedes it, exactly like every other coefficient. Nothing in this plan refits anything mid-simulation.
- **One uniform procedure, every species** — the gate in Task 1 is a single formula with fixed thresholds (`consts.TREND_GATE_T_THRESHOLD = 2.0`, `consts.TREND_GATE_MIN_SIGNIFICANT_MONTHS = 9`), applied identically to every key. Do not special-case any species/market code anywhere in this plan.
- **A key that fails the gate must produce byte-for-byte the same `CombinedMarket.fit()`/`get_price()` behavior as today** — this is the single most important regression constraint; Task 5's integration test exists specifically to pin it down against real data.

---

### Task 1: Trend gate (direct evidence from full history)

**Files:**
- Create: `src/models/market/trend_gate.py`
- Modify: `src/models/market/consts.py` (append new constants)
- Test: `src/models/market/test/test_trend_gate.py`

**Interfaces:**
- Produces: `passes_trend_gate(monthly: pd.DataFrame) -> bool` — `monthly` is a `(date, price, landings_kg)` DataFrame shaped like `level_volume_model.build_complete_monthly_series`'s output (one row per calendar month, `price`/`landings_kg` may be `NaN` for gaps). Only `date` and `price` are read. Returns `True` iff, computed independently per calendar month across the complete real history: (a) all 12 months' own year-over-year price slopes share the same, non-zero sign, and (b) at least `consts.TREND_GATE_MIN_SIGNIFICANT_MONTHS` of those 12 are individually significant (`|t| > consts.TREND_GATE_T_THRESHOLD`) in that same direction. Returns `False` for any month with fewer than 3 years of data, or if all years in some month are identical (no slope estimable).
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd

from src.models.market.trend_gate import passes_trend_gate


def _make_monthly(n_years, slope, noise_std, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_years * 12, freq="MS")
    years = dates.year.to_numpy(dtype=float)
    seasonal = np.array([2.0 + 0.1 * d.month for d in dates])
    trend = slope * (years - years.min())
    noise = rng.normal(0, noise_std, len(dates))
    price = seasonal + trend + noise
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": [1000.0] * len(dates)})


def test_passes_for_a_clear_consistent_trend():
    monthly = _make_monthly(n_years=10, slope=0.1, noise_std=0.05, seed=0)
    assert passes_trend_gate(monthly) is True


def test_fails_for_flat_noise_with_no_real_trend():
    monthly = _make_monthly(n_years=10, slope=0.0, noise_std=0.1, seed=0)
    assert passes_trend_gate(monthly) is False


def test_fails_when_sign_agrees_but_not_significant():
    # Shaped after the real HKE@FR near-miss found during investigation:
    # all 12 months drift the same direction, but the slope is small
    # relative to its own noise, so only a few months are individually
    # significant -- sign agreement alone must not be enough to pass.
    monthly = _make_monthly(n_years=10, slope=0.03, noise_std=0.12, seed=6)
    assert passes_trend_gate(monthly) is False


def test_fails_with_insufficient_history():
    monthly = _make_monthly(n_years=2, slope=0.1, noise_std=0.05, seed=0)
    assert passes_trend_gate(monthly) is False


def test_fails_when_a_month_has_zero_variance_in_years():
    # Degenerate: every observation of January falls in the same year --
    # no year-over-year slope is estimable for that month at all.
    dates = pd.date_range("2015-01-01", periods=11, freq="MS")  # Jan 2015 .. Nov 2015, one year only
    monthly = pd.DataFrame({
        "date": dates, "price": np.linspace(2.0, 2.5, 11), "landings_kg": [1000.0] * 11,
    })
    assert passes_trend_gate(monthly) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest src/models/market/test/test_trend_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.market.trend_gate'`

- [ ] **Step 3: Add the new constants**

Append to `src/models/market/consts.py` (after the existing `ANCHOR_SCHEME_*` constants at the end of the file):

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

- [ ] **Step 4: Write the implementation**

Create `src/models/market/trend_gate.py`:

```python
"""
Direct-evidence gate (spec 2026-10-07-combined-market-trend-term-design.md,
section 3) for whether a (species_code, category_code) key's CombinedMarket
fit should include a trend term. Uses ONLY the complete real historical
record -- no train/test split, no dependency on level_volume_model.py's own
fit. Per-calendar-month OLS (year -> price), requiring unanimous sign
agreement across all 12 months AND a strong majority individually
significant in that same direction.
"""
import numpy as np
import pandas as pd

from . import consts


def passes_trend_gate(monthly: pd.DataFrame) -> bool:
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
            return False  # not enough evidence to assess every month

        x_bar, y_bar = years.mean(), prices.mean()
        sxx = ((years - x_bar) ** 2).sum()
        if sxx == 0:
            return False  # every observation of this month falls in the same year

        slope = ((years - x_bar) * (prices - y_bar)).sum() / sxx
        intercept = y_bar - slope * x_bar
        residuals = prices - (intercept + slope * years)
        s2 = (residuals ** 2).sum() / (n - 2)
        se_slope = np.sqrt(s2 / sxx)
        t_stat = slope / se_slope if se_slope > 0 else 0.0

        slopes[month] = slope
        tstats[month] = t_stat

    signs = {np.sign(s) for s in slopes.values()}
    if len(signs) != 1:
        return False  # not unanimous

    majority_sign = next(iter(signs))
    if majority_sign == 0:
        return False  # exactly-flat consensus is not a trend

    n_significant = sum(
        1 for month, t in tstats.items()
        if abs(t) > consts.TREND_GATE_T_THRESHOLD and np.sign(slopes[month]) == majority_sign
    )
    return n_significant >= consts.TREND_GATE_MIN_SIGNIFICANT_MONTHS
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest src/models/market/test/test_trend_gate.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add src/models/market/trend_gate.py src/models/market/consts.py src/models/market/test/test_trend_gate.py
git commit -m "feat: add direct-evidence trend gate for CombinedMarket"
```

---

### Task 2: Trend-augmented fit/predict functions

**Files:**
- Modify: `src/models/market/level_volume_model.py` (append two new functions after the existing `predict_level_volume`, i.e. after line 140)
- Test: `src/models/market/test/test_level_volume_model.py` (append new test functions)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `fit_level_volume_trend_model(monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp) -> dict` and `predict_level_volume_trend(monthly: pd.DataFrame, params: dict) -> pd.Series`. `params` has the same keys as `fit_level_volume_model`'s output (`baseline_price`, `baseline_log_price`, `baseline_volume`, `alpha`, `beta`, `vol_anom_min`, `vol_anom_max`) plus three new ones: `gamma` (float, the trend coefficient), `trend_years_max` (float, the training window's own span in years — the forward clamp bound), and `train_end` (`pd.Timestamp`, the fit's own reference date, needed by `predict_level_volume_trend` to compute `trend_years` without a third argument). `"gamma" in params` is the dispatch marker later tasks use to distinguish a trend-augmented fit from a plain one.

- [ ] **Step 1: Write the failing tests**

Append to `src/models/market/test/test_level_volume_model.py`:

```python
from src.models.market.level_volume_model import (
    fit_level_volume_trend_model,
    predict_level_volume_trend,
)


def _make_trending_monthly(n_years=6, base_price=2.0, vol_response=-0.001, price_trend_per_year=0.1, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_years * 12, freq="MS")
    volume = 1000.0 + rng.normal(0, 50, len(dates))
    years = dates.year.to_numpy(dtype=float)
    price = (
        base_price + vol_response * (volume - 1000.0)
        + price_trend_per_year * (years - years.min())
        + rng.normal(0, 0.01, len(dates))
    )
    return pd.DataFrame({"date": dates, "price": price, "landings_kg": volume})


def test_trend_fit_recovers_a_positive_trend_and_predicts_forward():
    monthly = _make_trending_monthly(price_trend_per_year=0.1)
    train_end = pd.Timestamp("2023-01-01")
    params = fit_level_volume_trend_model(monthly, train_start=None, train_end=train_end)

    assert params["gamma"] > 0  # constructed with a positive price trend
    assert params["train_end"] == train_end
    assert params["trend_years_max"] == pytest.approx(5.0, abs=0.01)

    predicted = predict_level_volume_trend(monthly, params)
    assert len(predicted) == len(monthly)
    train_mask = monthly["date"] < train_end
    mape = score_mape(predicted[train_mask], monthly["price"][train_mask])
    assert mape < 5.0  # well-fit synthetic data on its own training window


def test_trend_prediction_clamps_beyond_the_training_window_span():
    monthly = _make_trending_monthly(price_trend_per_year=0.1)
    train_end = pd.Timestamp("2023-01-01")
    params = fit_level_volume_trend_model(monthly, train_start=None, train_end=train_end)

    at_cutoff = pd.DataFrame({"date": [train_end], "price": [None], "landings_kg": [1000.0]})
    far_future = pd.DataFrame({"date": [pd.Timestamp("2050-01-01")], "price": [None], "landings_kg": [1000.0]})
    way_past_max = pd.DataFrame(
        {"date": [train_end + pd.DateOffset(years=int(params["trend_years_max"]) + 1)],
         "price": [None], "landings_kg": [1000.0]}
    )

    pred_cutoff = predict_level_volume_trend(at_cutoff, params)
    pred_far = predict_level_volume_trend(far_future, params)
    pred_beyond_max = predict_level_volume_trend(way_past_max, params)

    assert pred_far.iloc[0] > pred_cutoff.iloc[0]  # trend continues forward...
    assert pred_far.iloc[0] == pytest.approx(pred_beyond_max.iloc[0], rel=1e-9)  # ...but clamps, doesn't keep climbing


def test_trend_fit_raises_on_insufficient_data():
    monthly = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "price": [2.0, 2.1],
        "landings_kg": [1000.0, 1000.0],
    })
    with pytest.raises(ValueError):
        fit_level_volume_trend_model(monthly, train_start=None, train_end=pd.Timestamp("2020-03-01"))


def test_trend_fit_raises_when_volume_has_zero_variance():
    dates = pd.date_range("2018-01-01", periods=24, freq="MS")
    monthly = pd.DataFrame({
        "date": dates,
        "price": np.linspace(1.8, 2.2, 24),
        "landings_kg": [1000.0] * 24,  # zero variance
    })
    with pytest.raises(ValueError):
        fit_level_volume_trend_model(monthly, train_start=None, train_end=pd.Timestamp("2020-01-01"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest src/models/market/test/test_level_volume_model.py -v -k trend`
Expected: FAIL with `ImportError: cannot import name 'fit_level_volume_trend_model'`

- [ ] **Step 3: Write the implementation**

Append to `src/models/market/level_volume_model.py`, directly after the existing `predict_level_volume` function (after line 140):

```python


def fit_level_volume_trend_model(monthly: pd.DataFrame, train_start: pd.Timestamp | None, train_end: pd.Timestamp) -> dict:
    """
    Trend-augmented sibling of fit_level_volume_model (spec
    2026-10-07-combined-market-trend-term-design.md, section 4). Extends
    the existing 1-D vol_anom -> price_anom regression to a 2-D
    least-squares fit adding trend_years = (date - train_end) / 365.25 as
    a second predictor (zero at train_end, negative across the training
    window). Only ever called for a key that has already passed
    trend_gate.passes_trend_gate -- this function itself does not re-check
    eligibility.
    """
    monthly = monthly.copy()
    monthly["month"] = monthly["date"].dt.month
    monthly["log_price"] = np.log(monthly["price"])
    monthly["trend_years"] = (monthly["date"] - train_end).dt.days / 365.25

    train = monthly[monthly["date"] < train_end]
    if train_start is not None:
        train = train[train["date"] >= train_start]

    baseline_price = train.groupby("month")["price"].mean()
    baseline_log_price = train.groupby("month")["log_price"].mean()
    baseline_volume = train.groupby("month")["landings_kg"].mean()

    baseline_volume_mapped = train["month"].map(baseline_volume)
    train_price_anom = train["log_price"] - train["month"].map(baseline_log_price)
    train_vol_anom = (train["landings_kg"] - baseline_volume_mapped) / baseline_volume_mapped
    fit_mask = train_price_anom.notna() & train_vol_anom.notna() & (baseline_volume_mapped > 0)

    if fit_mask.sum() < 3 or train_vol_anom[fit_mask].var() == 0:
        raise ValueError("not enough usable training rows to fit level-volume-trend model")

    X = np.column_stack([
        np.ones(int(fit_mask.sum())),
        train_vol_anom[fit_mask].to_numpy(),
        train["trend_years"][fit_mask].to_numpy(),
    ])
    y = train_price_anom[fit_mask].to_numpy()
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta, gamma = coeffs

    years_trained = (
        (train_end - train["date"].min()).days / 365.25 if train_start is None
        else (train_end - train_start).days / 365.25
    )

    return {
        "baseline_price": baseline_price,
        "baseline_log_price": baseline_log_price,
        "baseline_volume": baseline_volume,
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "vol_anom_min": float(train_vol_anom[fit_mask].min()),
        "vol_anom_max": float(train_vol_anom[fit_mask].max()),
        "trend_years_max": float(years_trained),
        "train_end": train_end,
    }


def predict_level_volume_trend(monthly: pd.DataFrame, params: dict) -> pd.Series:
    """
    Trend-augmented sibling of predict_level_volume. trend_years is
    computed from params["train_end"] (so this function, like
    predict_level_volume, only ever needs (monthly, params)) and clamped
    to [0, params["trend_years_max"]] -- the trend cannot be extrapolated
    forward further than the years of real evidence it was estimated
    from, mirroring the existing vol_anom_min/max clamp exactly.
    """
    monthly = monthly.copy()
    monthly["month"] = monthly["date"].dt.month
    monthly["trend_years"] = (monthly["date"] - params["train_end"]).dt.days / 365.25
    monthly["trend_years"] = monthly["trend_years"].clip(lower=0, upper=params["trend_years_max"])

    baseline_volume_mapped = monthly["month"].map(params["baseline_volume"])
    vol_anom = (monthly["landings_kg"] - baseline_volume_mapped) / baseline_volume_mapped
    vol_anom = vol_anom.clip(lower=params["vol_anom_min"], upper=params["vol_anom_max"])

    predicted_log = (
        monthly["month"].map(params["baseline_log_price"])
        + params["alpha"]
        + params["beta"] * vol_anom
        + params["gamma"] * monthly["trend_years"]
    )
    predicted = np.exp(predicted_log)
    predicted.index = monthly.index
    return predicted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest src/models/market/test/test_level_volume_model.py -v -k trend`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full file to confirm no regression on the existing (non-trend) tests**

Run: `python -m pytest src/models/market/test/test_level_volume_model.py -v`
Expected: PASS (all tests, old and new)

- [ ] **Step 6: Commit**

```bash
git add src/models/market/level_volume_model.py src/models/market/test/test_level_volume_model.py
git commit -m "feat: add trend-augmented level-volume fit/predict functions"
```

---

### Task 3: Wire the gate and trend-aware selection into `CombinedMarket.fit()`

**Files:**
- Modify: `src/models/market/CombinedMarket.py:1-120` (imports at top; `fit()` method body)
- Test: `src/models/market/test/test_CombinedMarket.py` (append new test functions)

**Interfaces:**
- Consumes: `passes_trend_gate` (Task 1), `fit_level_volume_trend_model`/`predict_level_volume_trend` (Task 2).
- Produces: `CombinedMarket.fit()`'s behavior change — for a key where `passes_trend_gate(monthly)` is `True`, the frozen `self.models[key]` dict will contain `"gamma"` (the dispatch marker Task 4 reads); for every other key, `self.models[key]` is produced by the exact same code path as before this task, unchanged.

- [ ] **Step 1: Write the failing test**

Append to `src/models/market/test/test_CombinedMarket.py`:

```python
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


def test_fit_includes_trend_for_a_key_that_passes_the_gate():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _trending_synthetic_monthly()}

    market.fit(monthly_by_key, latest_date=pd.Timestamp("2025-01-01"))

    assert "gamma" in market.models[("ANE", "Fresh - Whole")]


def test_fit_excludes_trend_for_a_key_that_fails_the_gate():
    # _synthetic_monthly (already defined above in this file) has no
    # embedded trend -- flat price around 2.0 plus noise -- so it must not
    # pass the gate, and the frozen model must be the plain (non-trend)
    # shape, exactly as before this feature existed.
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _synthetic_monthly()}

    market.fit(monthly_by_key, latest_date=pd.Timestamp("2024-01-01"))

    assert "gamma" not in market.models[("ANE", "Fresh - Whole")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest src/models/market/test/test_CombinedMarket.py -v -k "includes_trend or excludes_trend"`
Expected: FAIL — `test_fit_includes_trend_for_a_key_that_passes_the_gate` fails (`"gamma" in market.models[...]` is `False`, since `fit()` doesn't call the gate yet); `test_fit_excludes_trend_for_a_key_that_fails_the_gate` passes vacuously already (no behavior change needed for that path) — that's expected and fine.

- [ ] **Step 3: Update imports**

In `src/models/market/CombinedMarket.py`, replace lines 16-24:

```python
from . import consts
from .level_volume_model import (
    fit_level_volume_model,
    predict_level_volume,
    fit_level_only_model,
    predict_level_only,
    score_mape,
)
from .window_selection import build_rolling_splits, select_best_window
```

with:

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

- [ ] **Step 4: Rewrite `fit()`'s per-key fitting/scoring closures**

In `src/models/market/CombinedMarket.py`, inside the `fit()` method, replace the body of the `for key, monthly in monthly_by_key.items():` loop from the `rolling_splits = build_rolling_splits(latest_date)` line through the `result = select_best_window(...)` line (i.e. replace the existing lines defining `rolling_splits`, `fit_fn`, `score_fn`, and `result`) with:

```python
        for key, monthly in monthly_by_key.items():
            rolling_splits = build_rolling_splits(latest_date)
            trend_eligible = passes_trend_gate(monthly)

            if trend_eligible:
                def fit_fn(train_start, train_end, _monthly=monthly):
                    return fit_level_volume_trend_model(_monthly, train_start, train_end)

                def score_fn(params, test_start, test_end, _monthly=monthly):
                    predicted = predict_level_volume_trend(_monthly, params)
                    mask = (_monthly["date"] >= test_start) & (_monthly["date"] < test_end)
                    return score_mape(predicted[mask], _monthly["price"][mask])
            else:
                def fit_fn(train_start, train_end, _monthly=monthly):
                    return fit_level_volume_model(_monthly, train_start, train_end)

                def score_fn(params, test_start, test_end, _monthly=monthly):
                    predicted = predict_level_volume(_monthly, params)
                    mask = (_monthly["date"] >= test_start) & (_monthly["date"] < test_end)
                    return score_mape(predicted[mask], _monthly["price"][mask])

            result = select_best_window(consts.WINDOW_CANDIDATE_YEARS, rolling_splits, fit_fn, score_fn)
```

This is the only place `trend_eligible` needs to be threaded through for selection — `fit_fn`/`score_fn` are already passed opaquely into `select_best_window`, which doesn't care which concrete model they wrap (per `window_selection.py`'s own stated design: "knows nothing about what kind of model is being fit").

- [ ] **Step 5: Update the final re-fit to also dispatch on `trend_eligible`**

Immediately below (still inside the same `for key, monthly in ...` loop), the existing code reads:

```python
            if result is not None:
                label, years = result
                train_start = latest_date - relativedelta(years=years) if years is not None else None
                try:
                    params = fit_level_volume_model(monthly, train_start, latest_date)
```

Change the `fit_level_volume_model(monthly, train_start, latest_date)` call on that line to dispatch the same way:

```python
            if result is not None:
                label, years = result
                train_start = latest_date - relativedelta(years=years) if years is not None else None
                try:
                    if trend_eligible:
                        params = fit_level_volume_trend_model(monthly, train_start, latest_date)
                    else:
                        params = fit_level_volume_model(monthly, train_start, latest_date)
```

No other line in `fit()` needs to change — the surrounding `self.models[key] = params`, `self.window_labels[key] = label`, `self._seed_volume_baseline_stats(...)`, the `except (ValueError, np.linalg.LinAlgError)` fallback-to-level-only cascade, and the final `fit_level_only_model` fallback are all already generic over `params`'s shape and apply identically regardless of which function produced it.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest src/models/market/test/test_CombinedMarket.py -v`
Expected: PASS (all tests, old and new — in particular every pre-existing test in this file uses `_synthetic_monthly`, which has no embedded trend and so must continue selecting the plain, non-trend path exactly as before)

- [ ] **Step 7: Commit**

```bash
git add src/models/market/CombinedMarket.py src/models/market/test/test_CombinedMarket.py
git commit -m "feat: wire trend gate into CombinedMarket.fit()'s candidate selection"
```

---

### Task 4: Dispatch in `get_price()`; confirm Step 2's anchor and the live-anchor path are unaffected

**Files:**
- Modify: `src/models/market/CombinedMarket.py:163-182` (the `get_price` method)
- Test: `src/models/market/test/test_CombinedMarket.py` (append new test functions)

**Interfaces:**
- Consumes: the `"gamma" in params` marker produced by Task 3.
- Produces: `get_price()` now returns a trend-aware prediction for any key whose frozen model includes a trend term, with no change to its existing real-price-first behavior or its None/no-model behavior.

- [ ] **Step 1: Write the failing tests**

Append to `src/models/market/test/test_CombinedMarket.py`:

```python
def test_get_price_uses_trend_augmented_prediction_when_model_has_trend():
    market = CombinedMarket("ES")
    monthly_by_key = {("ANE", "Fresh - Whole"): _trending_synthetic_monthly()}
    market.fit(monthly_by_key, latest_date=pd.Timestamp("2025-01-01"))
    assert "gamma" in market.models[("ANE", "Fresh - Whole")]  # precondition for this test

    price_near_fit_end = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2025-02-01"), volume=1000.0)
    price_far_future = market.get_price("ANE", "Fresh - Whole", pd.Timestamp("2040-02-01"), volume=1000.0)

    assert price_near_fit_end is not None
    assert price_far_future is not None
    # Constructed with a positive price trend -- a later date should
    # predict a higher price than a date right after the fit's own cutoff.
    assert price_far_future > price_near_fit_end


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
    assert "gamma" in params  # precondition: this is the trend-augmented path

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
    # trend-augmented model -- it dispatches on baseline_volume is None,
    # not on "gamma", so a trend-augmented model (which always has
    # baseline_volume set) must still take the predict_level_volume branch
    # here, not predict_level_volume_trend -- this function was not
    # touched by this plan, confirm that directly.
    market = CombinedMarket("ES")
    monthly = _trending_synthetic_monthly()
    market.fit({("ANE", "Fresh - Whole"): monthly}, latest_date=pd.Timestamp("2025-01-01"))

    live_series = market.get_live_anchor_series("ANE", "Fresh - Whole", monthly, monthly["date"])
    assert live_series is not None
    assert len(live_series) == len(monthly)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest src/models/market/test/test_CombinedMarket.py -v -k "trend_augmented_prediction or unaffected_by_trend"`
Expected: `test_get_price_uses_trend_augmented_prediction_when_model_has_trend` FAILS (`get_price` doesn't dispatch to the trend predictor yet, so `price_far_future` won't exceed `price_near_fit_end` the way a trend-aware prediction would — or may raise a `KeyError` on `params["train_end"]`/`gamma` never being read, depending on exact current code path). The other two tests PASS already (nothing to change for them) — confirming your starting assumption before Step 3 changes anything.

- [ ] **Step 3: Update `get_price` to dispatch on the trend marker**

In `src/models/market/CombinedMarket.py`, the current `get_price` method reads:

```python
        params = self.models[key]
        monthly = pd.DataFrame({"date": [date], "price": [None], "landings_kg": [volume]})
        if params.get("baseline_volume") is None:
            predicted = predict_level_only(monthly, params)
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
        elif "gamma" in params:
            predicted = predict_level_volume_trend(monthly, params)
        else:
            predicted = predict_level_volume(monthly, params)
        value = predicted.iloc[0]
        return float(value) if pd.notna(value) else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest src/models/market/test/test_CombinedMarket.py -v`
Expected: PASS (all tests, old and new)

- [ ] **Step 5: Commit**

```bash
git add src/models/market/CombinedMarket.py src/models/market/test/test_CombinedMarket.py
git commit -m "feat: dispatch get_price to the trend-augmented predictor when present"
```

---

### Task 5: Integration test against real production config data

**Files:**
- Test: `src/models/market/test/test_CombinedMarket.py` (append one new test function)

**Interfaces:**
- Consumes: everything from Tasks 1-4, exercised against real data instead of synthetic fixtures.
- Produces: a regression-pinning test asserting the now-empirically-validated real-data outcome, so any future change to the gate's thresholds or the model code is immediately caught if it changes which keys get a trend.

- [ ] **Step 1: Write the failing test**

Append to `src/models/market/test/test_CombinedMarket.py`:

```python
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


@pytest.mark.skipif(
    not os.path.exists(f"{CONFIG_DIR}/prices.csv"),
    reason="requires config/northwestern_med/prices.csv",
)
def test_real_data_selects_trend_only_for_ane_at_es():
    expected_trend = {("ES", "ANE")}  # the one key validated during investigation

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
            if key not in market.models:
                continue
            has_trend = "gamma" in market.models[key]
            should_have_trend = (market_code, species) in expected_trend
            assert has_trend == should_have_trend, (
                f"{market_code}/{species}: expected trend={should_have_trend}, got {has_trend}"
            )
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest src/models/market/test/test_CombinedMarket.py -v -k test_real_data_selects_trend_only_for_ane_at_es`
Expected: PASS. If it fails, do not weaken the assertion or widen `expected_trend` to make it pass — this test is pinning the exact, already-empirically-validated outcome from the design investigation (section 3 of the spec); a failure here means something in Tasks 1-4's implementation doesn't match the verified math, and that discrepancy must be found and fixed, not papered over.

- [ ] **Step 3: Commit**

```bash
git add src/models/market/test/test_CombinedMarket.py
git commit -m "test: pin real-data trend selection to ANE@ES only"
```

---

### Task 6: CombinedMarket-scoped accuracy validation

**Files:**
- Create: `src/models/market/test/test_combined_market_fidelity.py`

**Interfaces:**
- Consumes: `fidelity_backtest_harness.run_backtest_split` (existing, unmodified — called with `return_handler=True` to get a real, fitted `MarketHandler` back without needing any changes to that shared harness).
- Produces: a new, standalone accuracy check. Note: the existing `test_fidelity_backtest.py`/`fidelity_backtest_harness.py` only ever inspect `handler.market_system.local_markets` price history (`LocalMarket`) — there is currently no `CombinedMarket`-level accuracy coverage anywhere in the test suite at all. This task adds it directly, rather than retrofitting the shared harness (which stays untouched, per this plan's scope).

- [ ] **Step 1: Write the test**

Create `src/models/market/test/test_combined_market_fidelity.py`:

```python
"""
CombinedMarket-scoped accuracy check (this plan's Task 6) -- confirms the
trend term is a real net improvement for ANE@ES specifically and a
non-regression for every other validated species/market, against the real
MarketHandler/CombinedMarket price history (not a reimplementation).
Complements test_fidelity_backtest.py, which only covers LocalMarket.
"""
import os

import pandas as pd
import pytest

from src.models.market.test.fidelity_backtest_harness import run_backtest_split
from src.models.market.level_volume_model import score_mape

VALIDATED_SPECIES = ["ANE", "PIL", "HKE", "SAA", "MUT"]
CATEGORY = "Fresh - Whole"
CONFIG_DIR = "config/northwestern_med"

# Generous ceiling, matching test_fidelity_backtest.py's own convention --
# catches a real regression without requiring exact reproduction of a
# specific historical number.
MAX_ACCEPTABLE_MAPE = 50.0


@pytest.mark.skipif(
    not os.path.exists(f"{CONFIG_DIR}/prices.csv"),
    reason="requires config/northwestern_med/prices.csv",
)
def test_combined_market_accuracy_within_acceptable_mape():
    results, handler = run_backtest_split(
        root_config_dir=CONFIG_DIR.rsplit("/", 1)[0],
        scenario_id=CONFIG_DIR.split("/")[-1],
        as_of_date=pd.Timestamp("2022-01-01"),
        test_horizon_years=2,
        species_codes=VALIDATED_SPECIES,
        return_handler=True,
    )

    prices = pd.read_csv(f"{CONFIG_DIR}/prices.csv", low_memory=False)
    prices["date"] = pd.to_datetime(prices["date"])
    if "size" not in prices.columns:
        prices["size"] = None
    blended = prices[prices["size"].isna()]
    real_lookup = {
        (r.date, r.market_code, r.species_code, r.category_code): r.price
        for r in blended.itertuples(index=False)
    }

    rows = []
    for market_code, combined_market in handler.market_system.combined_markets.items():
        for species in VALIDATED_SPECIES:
            key = (species, CATEGORY)
            pairs = []
            for tick_id, snapshot in combined_market.price_history:
                predicted_price = snapshot["prices"].get(key)
                if predicted_price is None:
                    continue
                tick_date = pd.Timestamp(handler.start_time) + pd.DateOffset(months=tick_id)
                real_price = real_lookup.get((tick_date, market_code, species, CATEGORY))
                if real_price is None:
                    continue
                pairs.append((predicted_price, real_price))
            if not pairs:
                continue
            predicted = pd.Series([p for p, _ in pairs])
            real = pd.Series([r for _, r in pairs])
            mape = score_mape(predicted, real)
            rows.append({"market_code": market_code, "species_code": species, "mape": mape, "n": len(pairs)})

    results_df = pd.DataFrame(rows)
    assert len(results_df) > 0, "backtest produced no comparable CombinedMarket (species, market) pairs at all"

    print("\nCombinedMarket fidelity backtest MAPE by species/market:")
    print(results_df.sort_values(["species_code", "market_code"]).to_string(index=False))

    failures = results_df[results_df["mape"] > MAX_ACCEPTABLE_MAPE]
    assert failures.empty, (
        f"{len(failures)} CombinedMarket (species, market) pairs exceeded the "
        f"{MAX_ACCEPTABLE_MAPE}% MAPE ceiling:\n{failures.to_string(index=False)}"
    )
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest src/models/market/test/test_combined_market_fidelity.py -v -s`
Expected: PASS. Read the printed per-species/market MAPE table — confirm `ANE`/`ES` is present and reasonable (consistent with the investigation's own measured numbers: no-trend baseline was ~18% MAPE at the 2yr horizon, trend-augmented was ~16%). If any pair exceeds the ceiling, investigate before adding it to a known-limitations list — do not silently widen `MAX_ACCEPTABLE_MAPE` to make a failure disappear.

- [ ] **Step 3: Commit**

```bash
git add src/models/market/test/test_combined_market_fidelity.py
git commit -m "test: add CombinedMarket-scoped fidelity backtest"
```

---

---

### Task 7: Visual confirmation via the 20-year baseline scenario

**Files:**
- Create (throwaway, delete after use): `src/models/market/test/_verify_trend_baseline.py`

**Interfaces:**
- Consumes: the real `MarketHandler`/`CombinedMarket` with Tasks 1-4 applied, run exactly the way the investigation's own baseline-scenario spikes did throughout this project.
- Produces: no new production code — this is the spec's own explicit testing requirement #5 (re-run the 20-year baseline scenario to visually confirm the forecast shape changed as intended), kept as a one-off script per this project's established throwaway-script discipline (delete after use, never commit it).

- [ ] **Step 1: Write and run a throwaway 20-year baseline simulation**

Create `src/models/market/test/_verify_trend_baseline.py`:

```python
"""
THROWAWAY -- visual confirmation (spec testing requirement #5) that
ANE@ES's 20-year baseline forecast now continues trending upward post-2024
instead of flatlining, while every other validated species/market is
unchanged. Delete after use.
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


def test_verify_trend_in_baseline_forecast():
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
    handler = MarketHandler("verify-trend", SCENARIO_ID, start_date.to_pydatetime(), 1)
    response, err = handler.init_simulation(simulation, CONFIG_DIR)
    assert err is None
    market_system = handler.market_system

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

    ane_es = market_system.combined_markets["ES"]
    series = ane_es.price_history
    key = ("ANE", CATEGORY)
    real_data_end = blended["date"].max()
    post_2024 = [
        (tick_id, snapshot["prices"].get(key))
        for tick_id, snapshot in series
        if start_date + pd.DateOffset(months=tick_id) > real_data_end
    ]
    post_2024_prices = [p for _, p in post_2024 if p is not None]
    print(f"\nANE@ES post-real-data forecast: first={post_2024_prices[0]:.3f}, "
          f"last={post_2024_prices[-1]:.3f}")
    # Must now be visibly climbing, not flat -- this is the whole point of
    # the feature. Print, don't just assert, so a human reviews the shape,
    # not only a single pass/fail number.
    assert post_2024_prices[-1] > post_2024_prices[0]
```

- [ ] **Step 2: Run it**

Run: `python -m pytest src/models/market/test/_verify_trend_baseline.py -v -s`
Expected: PASS, with the printed first/last post-2024 ANE@ES values showing a real, visible increase (not approximately equal).

- [ ] **Step 3: Delete the throwaway script**

```bash
rm src/models/market/test/_verify_trend_baseline.py
git status --porcelain  # confirm clean -- this script is never committed
```

---

## Final check

After all 7 tasks: run the full suite once (`python -m pytest -q`) to confirm nothing elsewhere in the project (in particular `LocalMarket`'s own tests and the existing `test_fidelity_backtest.py`) regressed. Per the spec's Global Constraints, `LocalMarket`/Step 2/`stacked_model.py` should show zero diff in behavior — this run confirms that empirically rather than by inspection alone.
