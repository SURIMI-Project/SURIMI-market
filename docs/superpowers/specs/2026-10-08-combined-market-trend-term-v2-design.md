# CombinedMarket Trend Term v2 — Snapshot Rate + Continuous Confidence

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the v1 trend-term mechanism (merged `4558d5a..32a3f02`) with a simpler, safer design that preserves the same user-facing outcome — a real, continuing price trend for species that genuinely have one (currently only ANE@ES) — while eliminating a real side effect v1 was found to have on `LocalMarket` Step 2.

**Architecture:** The trend is now a separate, additive correction computed independently of — and layered on top of — the existing, completely unmodified window-selection and baseline fit. Window selection, `baseline_price`, `alpha`, `beta` are now **byte-for-byte identical** to the pre-trend-feature code, for every species, with no branching at all. A continuous confidence score (no pass/fail gate) scales how strongly a separately-measured historical rate gets projected forward.

**Tech Stack:** Python, numpy (`polyfit`), pandas. No new dependencies.

**Spec:** Supersedes `docs/superpowers/specs/2026-10-07-combined-market-trend-term-design.md` (v1) in full. Builds on the original `docs/superpowers/specs/2026-10-02-market-model-handler-design.md` (frozen-forever coefficients, adaptive window selection) exactly as v1 did.

## Global Constraints

- **Scope: `CombinedMarket` only.** Same as v1 — `LocalMarket`, `stacked_model.py`, `window_selection.py`'s shared `build_rolling_splits`/`N_ROLLING_SPLITS` stay untouched.
- **Window selection and `baseline_price`/`alpha`/`beta` must be IDENTICAL for every species, with no trend-eligibility branching anywhere in that path.** This is the single most important constraint in this document — it is the direct fix for the v1 regression below, and every task must preserve it.
- **No pass/fail gate anywhere.** Confidence is a continuous value in `[0, 1]`, always computed, always applied as a multiplier (possibly ~0) — never a branch that fully excludes a species.
- **One uniform procedure, every species.** Same formula, same constants (`TREND_CONFIDENCE_T_THRESHOLD = 2.0`, `TREND_RATE_BLOCK_YEARS = 3`, damping exponent = 1, i.e. linear/no exponent applied at all), applied identically. No species is named in code.

---

## 1. Why v1 is being replaced

v1 (merged `4558d5a..32a3f02`) added a hard pass/fail gate (12/12 month-sign agreement, ≥9/12 individually significant) and, for keys that passed it, re-fit the trend-augmented model via `numpy.linalg.lstsq` and used it to **also drive window-length selection** (the same `select_best_window` machinery, now scoring the trend-augmented predictor instead of the plain one for eligible keys).

The final whole-branch review (full history in git log on this branch) found that this coupling has a real, previously undocumented consequence: **the window CombinedMarket selects for a trend-eligible key can differ from what the plain (non-trend) scoring would have picked**, and `baseline_price` — the exact quantity `LocalMarket` Step 2 reads as its static anchor via `get_baseline_price_table` — is computed from whichever window won. For ANE@ES specifically, trend-eligible scoring selected the `Full` (~10yr) window where plain scoring would have selected `3yr`; the two windows' `baseline_price` differ by roughly 20%. Measured directly against the real LocalMarket fidelity backtest: 33 of 36 ES ports' ANE MAPE changed (mean +0.42pp, up to +23.3pp worse at one port, −10.7pp better at another) — a real, uncontrolled side effect on a system this project explicitly scoped as untouched.

Separately, during discussion of a fix, the user's own reasoning surfaced two further findings that independently argued against keeping v1's mechanism even with that coupling fixed:
- The hard pass/fail gate itself was rejected on its own terms — a continuous alternative was wanted instead.
- A from-scratch empirical comparison (below) found v1's regression-based rate is **not actually the most robust way to measure the trend** — a simpler, direct method does better on the one case v1's own gate was built to protect against.

## 2. What was tried and rejected, with evidence

All of the following were tested against real `config/northwestern_med/prices.csv` data, using the project's established long-horizon, many-historical-cutoff methodology (fit on a truncated window, walk the held-out comparison out 1-11 years depending on how much real data remains after each cutoff) — not just the short 1-2 year horizon the project's existing `test_fidelity_backtest.py` convention uses, which earlier in this investigation was shown to give a misleading picture for several species.

- **Continuous confidence-weighted damping of v1's own regression-based rate** (`confidence^k`, k=1,2,3): a real improvement over applying the full-strength rate unconditionally, but never fully closed the gap for the two genuinely dangerous cases (PIL@ES, SAA@FR) even at k=3 — e.g. SAA@FR's MAPE at a 6-year horizon: no-trend 39.3%, full-strength regression 157.9% (catastrophic), confidence³-weighted 33.3% (meaningfully better, but the underlying mechanism was still regression-based and the window-selection coupling bug hadn't yet been found).
- **Model 1** (pure recursive, `predicted[i] = predicted[i-1] + weighted_avg(MoM changes 1-3yr back)`, ported verbatim from the original research notebook) and **the Hybrid** (Model 1's recursive trend + a volume correction fit on its residuals, also ported verbatim): both decisively worse than every other approach tested under this project's own long-horizon methodology — mean MAPE at each pair's longest tested horizon, averaged across all 10 validated species/market pairs: Model 1 = 63.49%, Hybrid = 53.35%, vs. ~20-26% for every other approach, with catastrophic worst cases (Model 1 reached 208.5% MAPE for SAA@FR at a 7-year horizon). This independently reconfirms, under a different and arguably more rigorous methodology, the original notebook's own reason for abandoning both in favor of a frozen model in the first place. Not revisited further.
- **A confidence measure computed from the snapshot blocks themselves** (fraction of consecutive block-to-block deltas agreeing with the overall direction), instead of the per-month regression significance measure: rejected. With only 2-6 historical blocks available (10-15 years of real history, 2-4yr blocks), this measure is effectively binary (0, 0.5, or 1) rather than genuinely continuous — raising an already-binary value to any power changes nothing, confirmed directly (identical results across damping exponents 1 through 5). Once tested at each block size's own fair longest horizon (see below), it was also mostly worse than the month-regression measure: better for ANE@ES and SAA@FR specifically, but substantially worse for ES/PIL, FR/ANE, FR/HKE, FR/MUT. Net: higher variance, not a net improvement, and not genuinely continuous in practice. Not adopted.
- **Block size 4yr**: structurally in tension with testing a long horizon — a larger block size needs more lead-time history before the first usable cutoff, which necessarily shortens how far into the future that configuration can be fairly tested (confirmed directly: 4yr blocks could only be tested out to 1-7 years depending on market, vs. 4-11 years for 2-3yr blocks). Given this, and no clear win over 3yr blocks at the horizons where a fair comparison was possible, 3yr blocks were kept as the default — the same block size used throughout this investigation's own validated-good result for ANE@ES.
- **Clamping the rate's raw magnitude** to the single most extreme observed block-to-block transition rate (in addition to confidence-weighting): a no-op in every test performed — the least-squares fit across 2+ blocks never exceeded the single most extreme pairwise delta in this dataset. Not adopted; the existing `vol_anom`-style clamp on `trend_years` already bounds how far forward the rate can be extrapolated, which is where the real risk was.
- **Damping exponent**: confidence¹ (linear) beat confidence² and confidence³ on both mean and worst-case MAPE, consistently, once properly isolated from the window-selection coupling bug. Squaring/cubing was this investigation's own earlier, not-fully-justified interim recommendation; it does not hold up. Confidence¹ is adopted.

## 3. The adopted design

**Step 1 — baseline (unchanged from today's pre-v1 code):** `CombinedMarket.fit()`'s window-length selection and final re-fit use `fit_level_volume_model`/`predict_level_volume` exactly as before any trend feature existed — no branching, no trend-aware scoring, for any key. This is the literal fix for section 1's regression: `baseline_price` cannot diverge by trend-eligibility because nothing about how it's computed depends on trend-eligibility at all.

**Step 2 — confidence (same math as v1's gate, now used as a continuous multiplier, not a threshold):** per calendar month 1-12, OLS slope of `price ~ year` over that month's own real historical observations (same formula as v1, see v1 spec section 3 for the exact slope/SE/t-statistic derivation — unchanged). 

```
confidence = (number of the 12 months individually significant, |t| > 2, in the majority sign direction) / 12
```

No sign-agreement requirement and no ≥9/12 threshold — those were v1's pass/fail gate and are removed. `confidence` is used directly as a multiplier (see Step 4); a species with no real trend naturally gets confidence near 0 (e.g. PIL@ES measured 0.00 in production), not a hard exclusion.

**Step 3 — rate (replaces v1's `numpy.linalg.lstsq` regression entirely):** build sequential 3-year blocks spanning the complete real historical record before `train_end` (as many as fit — for ES's ~10yr history that's 3 blocks, for FR's ~15yr history that's 4-5), compute each block's own mean log-price, and fit the rate as the slope of a straight line through the block averages (`numpy.polyfit(block_centers, block_averages, 1)`, `block_centers` in years-before-`train_end`). This measurement is completely independent of Step 1 — it doesn't touch, read, or influence the window-length selection in any way.

Measured in production for the 10 validated species/market pairs (see section 4's table for the full picture): ANE@ES confidence 0.92, rate +0.0625/yr; the next-highest confidence is MUT (ES 0.42, FR 0.42) and SAA@FR (0.25); everything else is ≤0.17.

**Step 4 — combine, as an additive correction on top of Step 1's unchanged prediction:**

```
trend_years = clip((date - train_end) / 365.25, 0, years_trained)
price_hat = predict_level_volume(date, volume)  ×  exp(confidence × rate × trend_years)
```

`years_trained` is the span of real history actually used to compute the rate (so the correction cannot be extrapolated further forward than the years of evidence it was measured from — same principle as the existing `vol_anom` clamp, applied here to time instead of volume). The lower clamp bound is `0`, not symmetric — for any date at or before `train_end`, the correction is exactly zero, i.e. the prediction for any historical date is identical to what the unchanged Step 1 model alone would produce. (This is a deliberate difference from how v1's clamp ended up being corrected to symmetric: v1's `gamma` was fit via regression directly against unclamped, negative `trend_years` values for training rows, so an asymmetric predict-time clamp was inconsistent with what the fit assumed. Here, Step 3's rate is measured purely from block averages with no per-row regression against `trend_years` at all, so there is no equivalent fitting-time assumption to be inconsistent with. The zero-lower-bound has no practical consequence in production regardless, since `get_price`'s real-price-first path already returns the real recorded price for any historical date before ever reaching this model.)

**Verified directly against the real, unmodified window-selection code** (not a simplified stand-in): `baseline_price[Jan]` for ANE@ES = 3.072 — identical to the value computed by today's production `select_best_window` before any trend feature existed. A real 20-year baseline-scenario forecast shows ANE@ES climbing smoothly from 3.088 (2025) to 3.883 (2029) and SAA@FR — the case that caused a 99.99%+ MAPE blowup under v1's full-strength regression — now rising gently and safely from 0.733 to 0.750 over the same period, with every other species scaling in between proportional to its own measured confidence.

## 4. Measured values for all 10 validated species/market pairs (production, full real history)

| | window (unchanged from today) | confidence | rate (log-price/yr) |
|---|---|---|---|
| ES/ANE | 3yr | 0.92 | +0.0625 |
| ES/PIL | 2yr | 0.00 | +0.0060 |
| ES/HKE | Full | 0.00 | +0.0157 |
| ES/SAA | 6yr | 0.08 | +0.0203 |
| ES/MUT | 3yr | 0.42 | +0.0150 |
| FR/ANE | 3yr | 0.08 | +0.0337 |
| FR/PIL | Full | 0.08 | −0.0027 |
| FR/HKE | 4yr | 0.17 | +0.0294 |
| FR/SAA | Full | 0.25 | +0.0226 |
| FR/MUT | 3yr | 0.42 | −0.0252 |

## 5. Implementation changes from v1's merged code

- `src/models/market/trend_gate.py`: the pass/fail gate (`passes_trend_gate`) is replaced by a continuous confidence function (same per-month OLS math, different return value and no threshold). Rename to reflect this (e.g. `trend_confidence.py` / `compute_trend_confidence`).
- `src/models/market/level_volume_model.py`: `fit_level_volume_trend_model`/`predict_level_volume_trend` (the lstsq-regression mechanism) are removed and replaced with the block-based rate functions (section 3, Steps 3-4).
- `src/models/market/CombinedMarket.py`: `fit()`'s per-key closures revert to calling the plain `fit_level_volume_model`/`predict_level_volume` unconditionally for window selection and the final re-fit (removing the `trend_eligible` branch entirely) — then, after the normal fit completes, separately compute and attach confidence + rate to the frozen params for every key (always, not gated). `get_price()`'s dispatch changes from "use the trend predictor if `gamma` is in params" to "always apply the confidence × rate × trend_years correction on top of the plain prediction" (it's now unconditional — for a zero-confidence key this correction is ~0 and the output is indistinguishable from today's plain prediction, so there's no need for a branch at all).
- Tests across `test_trend_gate.py` → confidence tests, `test_level_volume_model.py`'s trend tests, `test_CombinedMarket.py`'s trend tests, the real-data integration test (Task 5 equivalent), and the CombinedMarket fidelity backtest (Task 6 equivalent) all need rewriting against the new mechanism and the new expected values (section 4's table).
- The Step 2 regression test that the final review flagged as missing in v1 (an assertion that `baseline_price`/window choice is identical regardless of a key's confidence) becomes directly testable and should be added as a first-class regression test, not left as a prose claim.

## 6. Testing

Mirrors v1's testing plan (unit tests for the new confidence/rate functions, `CombinedMarket` wiring tests, a real-data integration test pinning the measured values in section 4, a CombinedMarket-scoped fidelity backtest, and a 20-year baseline-scenario visual check) — plus, new and required given section 1's regression: a test that fits two keys with different confidence (one ~0, one high) against the same underlying data shape and asserts their `baseline_price`/window label are identical. This is the test that would have caught v1's regression before merge.

## 7. Explicitly out of scope / deferred

Same as v1: `LocalMarket` Step 1, `LocalMarket` Step 2 (`stacked_model.py`), and revisiting these thresholds for a future region (`aegean_sea`) — revisit empirically if that region's real data calls for it.
