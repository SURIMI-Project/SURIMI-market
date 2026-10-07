# CombinedMarket Trend Term — Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let `CombinedMarket`'s frozen Model 2 represent a genuine, sustained secular price trend for a `(species_code, category_code)` where one demonstrably exists in the real historical record, without introducing the instability that made the same idea fail when tried before (the original notebook's "Hybrid: Trend + Volume" model, and this investigation's own rolling-origin-selected trend candidate).

**Architecture:** Add an optional linear-in-time trend term to the existing log-price/relative-volume-anomaly fit (`level_volume_model.py`), used identically to how `vol_anom` is already clamped to its own training range. Whether a given `(species_code, category_code)` gets the trend term at all is decided by a new, uniform gate computed directly from the full real historical record — not from rolling-origin backward validation, which this investigation found gives false positives (species that look good on a short or arbitrarily-placed historical holdout, but whose trend reverses or degrades badly over a genuine long forecast horizon).

**Tech Stack:** Python, numpy (`linalg.lstsq`, `polyfit`), pandas. No new dependencies.

**Spec:** Builds on `docs/superpowers/specs/2026-10-02-market-model-handler-design.md` (the original Market model handler design — sections 6/7/15 on frozen-forever coefficients, adaptive window selection, and the previously-investigated-and-rejected rolling-recalibration/hybrid-trend alternatives). This document does not re-litigate anything settled there except where explicitly noted below.

## Global Constraints

- **Scope: `CombinedMarket` only.** `LocalMarket` Step 1 shares `level_volume_model.py`'s fit/predict functions but is explicitly **not** in scope for the trend term. An empirical spike (10-port sample across the 5 validated species) showed only 12/44 units preferring a trend variant, with some catastrophic instability (one port/species pair's best trend candidate scored 208pp worse than no-trend) — consistent with the project's own documented "estimation instability" concern for sparse per-port data (spec section 15, the ANE@ESCDK Step 2 finding). Revisit LocalMarket separately if ever justified; do not fold it into this change.
- **`LocalMarket` Step 2's static anchor is unaffected.** `get_baseline_price_table` only ever reads the plain per-month arithmetic mean (`baseline_price`), never the regression coefficients — so this change has zero effect on Step 2 regardless of scope decisions above.
- **Frozen-forever still holds.** The trend term is fit once, from real history available at `init_simulation`, exactly like every other coefficient (spec section 6). It is never refit mid-simulation. "Rolling re-calibration" (refit schedule) and "trend term" (fit shape) are orthogonal — refitting a trend-blind model more often still produces a trend-blind frozen number; that's why the two were never the same question, and why the prior rolling-recalibration investigation said nothing about whether a trend term would help.
- **One uniform procedure, every species.** The gate below is a single formula with fixed thresholds, applied identically to every `(species_code, category_code)` at `CombinedMarket` level — no species is named in code. It happens to currently pass only `ANE` at `ES`, as a consequence of the data, not a branch.

---

## 1. Background: why this needed its own investigation

The published scenario-forecast chart showed a frozen, dead-flat 20-year forecast for every species after real data ends. Investigation confirmed this is a direct, documented consequence of `fit_level_volume_model`'s `baseline_log_price = train.groupby("month")["log_price"].mean()` — a plain per-calendar-month arithmetic mean has no capacity to represent a slope, regardless of training window length or how often it's refit.

Checked against real Spain CPI (cumulative +23%, 2015-2024) the flatness is not simply "missing inflation": of the 5 validated species, only `ANE` shows a real, large nominal increase (+68.8%, 2015-2024) — the other four are flat or declining in nominal terms, meaning they are getting *cheaper* in real terms. A uniform inflation-index adjustment would therefore be wrong for 4 of 5 species.

## 2. Why rolling-origin-selected trend was rejected

Three rounds of empirical testing, all using real `config/northwestern_med/prices.csv` data (ES and FR `CombinedMarket`, 5 validated species each):

1. **Naive rolling-origin selection** (today's production methodology: 3 splits, 2-year test horizon, same `select_best_window`/`score_mape` used for window-length today): 5/10 species-market pairs preferred a trend-augmented candidate. Promising-looking, but —
2. **Extended long-horizon test** (many more, denser splits; MAPE measured in 1-year buckets out to 6-11 years ahead, not just 2): revealed that several of those 5 "wins" (`PIL@ES`, `SAA@FR`, `HKE@FR`) had a trend coefficient that was *statistically significant and consistently signed in-sample*, yet produced **worse** forecasts at every long-horizon bucket, sometimes catastrophically (`SAA@FR`: 99.99% MAPE at year 6 vs. 39% for no-trend). Significance and consistent sign at fit time did not predict long-horizon generalization.
3. **Sweeping the validation horizon itself** (2 through 6 years, same unmodified selection code, no new machinery): confirmed `ANE@ES` is the only pair that survives consistently (4/5 horizons tested) — everything else that looked good at a short horizon was correctly and consistently rejected once the horizon lengthened, with one single-point noise exception (`SAA@FR` at exactly horizon=4).

All three approaches share a structural flaw: they validate against an **artificially truncated** proxy model (holding out real data to create a test set), never the actual model that gets deployed (fit on all available real history). That proxy's behavior is not guaranteed to match the real one's, and the extended test showed concretely that it sometimes doesn't.

## 3. The gate: direct evidence from the full historical record

Instead of any backward holdout, decide per-`(species_code, category_code)` trend inclusion directly from the complete real history — the same data the final frozen fit will use, nothing withheld:

For each of the 12 calendar months independently, fit `price ~ year` via OLS over that month's own real historical observations (e.g. every real January across all years of history). This yields 12 independent slope estimates and their t-statistics.

**Gate condition — both must hold:**
- **Unanimous sign:** all 12 months' slopes share the same sign.
- **Strong significance:** at least 9 of the 12 months are individually significant in that same direction (`|t| > 2`, roughly `p < 0.05`).

Measured against all 10 validated species-market pairs:

| | months agreeing in sign | months individually significant |
|---|---|---|
| **ES/ANE** | **12/12** | **11/12** — passes |
| ES/PIL | 7/12 | 0/12 |
| ES/HKE | 9/12 | 0/12 |
| ES/SAA | 8/12 | 1/12 |
| ES/MUT | 8/12 | 5/12 |
| FR/ANE | 8/12 | 1/12 |
| FR/PIL | 2/12 | 1/12 |
| FR/HKE | 12/12 | 2/12 — sign-unanimous but fails significance |
| FR/SAA | 11/12 | 3/12 |
| FR/MUT | 5/12 | 5/12 |

`FR/HKE` is the important near-miss: 12/12 sign agreement alone would have wrongly admitted it. Requiring a strong majority of months to also be individually significant is what correctly excludes it (only 2/12) while keeping `ES/ANE` (11/12). Both conditions are required; neither alone is sufficient.

This check uses only `numpy.polyfit`/standard OLS standard-error arithmetic on the real monthly series grouped by calendar month — no new statistical machinery, no dependency on `level_volume_model.py`'s own fit at all. It lives in a new, small, standalone module, `src/models/market/trend_gate.py` (one function, e.g. `passes_trend_gate(monthly: pd.DataFrame) -> bool`), consistent with the project's existing pattern of focused single-purpose files (`window_selection.py` alongside `level_volume_model.py`). It runs once per key, before the existing window-length selection, and only decides yes/no on trend eligibility.

## 4. The model extension

`level_volume_model.py` gains a trend-augmented sibling pair, used only for keys that pass the gate:

```python
def fit_level_volume_trend_model(monthly, train_start, train_end):
    """
    Same as fit_level_volume_model, plus a third regression predictor:
    trend_years = (date - train_end) / 365.25 (zero at train_end, negative
    across the training window). Extends the existing 1-D np.polyfit
    (vol_anom -> price_anom) to a 2-D least-squares fit
    (vol_anom, trend_years) -> price_anom via numpy.linalg.lstsq.
    Returns the same dict as fit_level_volume_model, plus "gamma" (the
    trend coefficient) and "trend_years_max" (the training window's own
    span in years, used as the forward clamp bound).
    """

def predict_level_volume_trend(monthly, params):
    """
    Same as predict_level_volume, plus + params["gamma"] * trend_years,
    where trend_years = (date - params["train_end"]) / 365.25 (params
    stores its own fit's train_end, so this function needs no third
    argument) clamped to [-params["trend_years_max"], params["trend_years_max"]]
    -- bounds how far the trend can be extrapolated, in either direction,
    to the number of years of real history it was estimated from.
    """
```

> **Correction (made during Task 2's implementation):** the original version of this document clamped the lower bound to `0` instead of `-trend_years_max`, and passed `train_end` as a third argument instead of storing it in `params`. The zero-lower-bound reasoning (the trend should contribute nothing for any date at or before `train_end`) doesn't survive contact with how `gamma` is actually fit: `fit_level_volume_trend_model` regresses against the raw, unclamped (negative) `trend_years` for every training row, so a `[0, ·]` predict-time clamp makes `predict` inconsistent with what `fit` assumed — confirmed empirically (training-window MAPE 11.82% with the `[0, ·]` clamp vs. 1.15% with the symmetric one, same fitted `gamma`). The symmetric clamp fixes that inconsistency and has zero effect on the feature's production-critical paths: window-length selection only ever scores test dates `>= train_end` (non-negative `trend_years` either way), and the final deployed forecast only ever queries dates after `train_end` too. The only case the lower bound actually governs is `get_price` querying a historical gap date within the training span with no real-price-first match — a rare edge case, now handled by the more internally-consistent symmetric clamp instead of an unmotivated floor at zero. The storing-`train_end`-in-`params` change is purely a signature simplification matching `predict_level_volume`'s existing 2-argument shape; no behavior change.

`CombinedMarket.fit()`, for a key that passes the gate: runs the *existing, unmodified* `select_best_window`/`rolling_splits`/`score_mape` machinery exactly as today, just fitting/scoring `fit_level_volume_trend_model`/`predict_level_volume_trend` instead of the plain functions, to choose the window length. For a key that does not pass the gate: entirely unchanged, today's code path, byte-for-byte.

`CombinedMarket.get_price()` dispatches on a marker in the frozen params (`"gamma" in params`) to call `predict_level_volume_trend` instead of `predict_level_volume`. `get_baseline_price_table()` (Step 2's anchor) and `get_live_anchor_series()` (unused in production) are unchanged — both only ever read `baseline_price`/`baseline_log_price`, never the regression coefficients.

## 5. Testing

1. **Unit tests for the gate** (`src/models/market/test/test_trend_gate.py`): synthetic monthly series with a clear embedded trend (gate passes) and without one (gate fails), plus the real `ANE@ES`/`HKE@FR`-shaped edge case (sign-unanimous but not significant — must fail).
2. **Unit tests for `fit_level_volume_trend_model`/`predict_level_volume_trend`**, mirroring the existing `level_volume_model.py` test shape (degenerate/insufficient-data `ValueError` cases, clamp behavior at the trend boundary).
3. **Integration test against real config data**: `CombinedMarket.fit()` on `config/northwestern_med` must select the trend variant for `ANE`/`Fresh - Whole`/`ES` and the plain variant for all other validated-species keys — a regression test pinning the now-validated outcome.
4. **Re-run the `CombinedMarket`-scoped slice of the fidelity backtest** (not the full ~160-port suite — out of scope) to confirm no regression for the 9 other keys and a real accuracy improvement for `ANE@ES`.
5. **Re-run the 20-year baseline scenario simulation** (as used throughout this investigation) to visually confirm `ANE@ES`'s forecast now continues trending upward post-2024 instead of flatlining, while every other species/market is pixel-for-pixel unchanged.

## 6. Explicitly out of scope / deferred

- `LocalMarket` Step 1 trend term (see Global Constraints).
- `LocalMarket` Step 2 (`stacked_model.py`) — untouched, as in every prior investigation in this project.
- Revisiting the gate's exact thresholds (12/12 sign, 9/12 significance) based on future species/regions (e.g. `aegean_sea`) — these are deliberately strict, chosen to cleanly separate the one validated real case from the nearest false positive (`FR/HKE`, 12/12 sign but 2/12 significance) in the currently-available data; revisit empirically if a future region's real data calls for it, the same way `WINDOW_CANDIDATE_YEARS` was.
