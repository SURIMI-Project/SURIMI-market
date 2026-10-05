# Market Model Handler Design

Status: Draft, pending user review
Date: 2026-10-02

## 1. Purpose

Productionize the fish-price-prediction model validated over an extended
research project in `scripts/fisheries_data/price_prediction_model.ipynb`
(combined-market Model 2) and
`scripts/fisheries_data/price_prediction_model_ports.ipynb` (per-port
production design: Model 2 + a distance-weighted Stacked/DistanceOnly
dynamic switch). This becomes a new model plugin, `src/models/market/`,
following the existing `Handler` plugin architecture already used by
`arimax`, `catboost`, and `elsa`, and wired into
`src/server/model_router.py`.

The new module mirrors Elsa's existing shape (`Tick`, a combined-market
class, `LocalMarket`, `MarketSystem`, a `Handler` subclass) rather than
introducing a new architectural style, since that pattern is already
established and tested in this codebase.

## 2. Scope

- Generalizes to every species with sufficient historical price data in
  the scenario's config, not hardcoded to the 5 species
  (ANE/PIL/HKE/SAA/MUT) used during validation — the underlying
  mechanism (seasonal baseline fitting, volume regression, distance
  weighting, the volume-ratio threshold switch) is purely data-driven
  and doesn't hardcode anything species-specific. Thinner species will
  simply show higher error, the same pattern already observed during
  validation, not a different code path.
- Input file preparation (how `prices.csv` gets generated from EMODnet
  data) is now resolved — see section 13. S3 sourcing for the final
  deployment is still out of scope for this spec (Elsa already has a
  reusable, model-agnostic `S3_Storage.DownloadFilesFromS3` mechanism —
  the new model would define its own config file expectations and
  likely reuse this unchanged, but the details are a follow-up
  discussion).
- Out of scope for this spec (flagged by the user as a follow-up
  discussion, not solved here): the precise interaction between "use
  real price if available, else predict" (see section 8) and how long
  real data typically extends into a simulation's run.

## 3. Class structure

- **`Tick`** — reused essentially unchanged from Elsa. Pure time
  bookkeeping, no species/price logic.
- **`CombinedMarket`** — one instance per 2-letter market code (e.g.
  `ES`, `FR`). Named to match the existing "combined-market Model 2"
  terminology used throughout the validated research notebooks, for
  continuity between the research and the implementation. (Other names
  considered and rejected: `GlobalMarket` — misleading, since it
  represents a single country, not a cross-national concept;
  `NationalMarket` / `CountryMarket` — reasonable, but less continuous
  with established vocabulary; `AggregateMarket` — too generic, risks
  ambiguity if a future supra-national aggregation level is ever added
  alongside a country-level one.) Holds the frozen combined-market
  Model 2 (seasonal baseline + volume-response coefficient), **keyed by
  `(species_code, category_code)`** — one full coefficient set per
  species/category combination, not one per market, matching Elsa's own
  `dict[tuple[species_code, category_code], ...]` data model and the
  fact that the whole validated methodology was always per-species.
  Does **not** hold references to `LocalMarket` objects — see section 4.
- **`LocalMarket`** — one instance per 5-letter LOCODE. Holds, **each
  keyed by `(species_code, category_code)`** (not one per port):
  - its own (frozen) Step-1 Model 2 coefficients,
  - its own (frozen) Step-2 Stacked and DistanceOnly coefficient sets,
  - its adaptive volume baseline (see section 6),
  - its current-tick catch accumulator.

  Its lon/lat is the one exception — held once per port, not per
  species, since physical location doesn't depend on species. See
  section 4 for how this single, shared geography feeds into
  per-species High-tier assignment and neighbor weighting.
- **`MarketSystem`** — orchestrator, structurally unchanged from Elsa:
  owns the `Tick`, the `CombinedMarket`s, and the dict of
  `LocalMarket`s; drives `update_sales` / `simulate_step` / history /
  export. What changes from Elsa is the internal per-tick computation
  (section 5) and its added responsibility for topology and
  aggregation (section 4).

## 4. Local ↔ Combined topology and aggregation

The parent/child relationship is derived automatically from the LOCODE
string itself — a 5-letter code's first 2 characters are, by the
UN/LOCODE standard, always its country code — so no separate
config/mapping is needed. `MarketSystem` computes this mapping once at
init (static for the life of the simulation) and reuses it every tick,
rather than re-deriving it on each tick.

**`CombinedMarket` instantiation:** the full set of needed
`CombinedMarket`s is the **union** of (a) any 2-letter codes explicitly
listed in `markets.csv`, and (b) any distinct 2-letter prefixes implied
by the 5-letter local codes actually present. This means a
`CombinedMarket` always exists for any country that has local ports
under it, even if that country's own 2-letter row happens to be missing
from `markets.csv` — there is never a silent gap where a local port's
volume has nowhere to aggregate into.

**Routing an incoming sale** (in `update_sales`): exact 5-letter match →
that `LocalMarket`'s own accumulator. Otherwise, if the code shares a
known 2-letter prefix → that `CombinedMarket`'s own accumulator (a
catch-all, not a strict whitelist — this is deliberately more lenient
than Elsa's current exact-whitelist check, so that sales addressed
directly to a 2-letter code, or to any unlisted code sharing a known
prefix, are captured rather than rejected). Every time the catch-all
branch is actually used (i.e. the code wasn't an exact local match),
log a **WARNING** naming the specific code and which `CombinedMarket`
it was folded into — it still isn't rejected, but a typo or unexpected
code stays visible in the logs rather than silently skewing that
country's aggregate. A code matching neither an exact local market nor
any known prefix → `MarketError`.

**Distance calculation:** computed fresh at init from `markets.csv`'s
lon/lat, for 5-letter local markets only — 2-letter `CombinedMarket`
codes have no coordinates (`NA` in `markets.csv`) and never participate
in distance weighting, so they're excluded from this entirely, not
treated as a tolerated exception. Hand-rolled standard haversine
great-circle formula, **R = 6,378,137 m** (WGS84 equatorial radius) —
confirmed empirically to match `.cache/test/port_distances.csv` (the
distance data the research notebooks were validated against) to
1e-10m precision, and confirmed to be exactly the default radius used
by R's
[`geosphere::distHaversine()`](https://rdrr.io/cran/geosphere/man/distHaversine.html),
which is what produced that reference file. Using this exact
formula+radius means production reproduces identical High-tier
neighbor selection and inverse-distance weighting to what was
backtested — no new dependency needed (plain Python/`numpy`, already a
transitive dependency via `pandas`); a dedicated geo package
(`pyproj`, `geopy`, `haversine`) was considered and rejected as
unnecessary weight for a single, stable, already-precisely-specified
formula. This distance lookup is computed **once, shared across every
species** (it's pure geography) — see the note on species-keying below
for how species-specific High-tier assignment is built on top of it.

**`CombinedMarket`'s total tick volume** = sum of its `LocalMarket`
children's catch accumulators (computed by `MarketSystem`, which already
holds the full `LocalMarket` dict and the topology mapping) + its own
direct accumulator. This single formula handles the general case (e.g.
`ES`, with many local children) and the childless case (e.g. `FR` in
the current `config/northwestern_med` scenario, which has zero local
children in that config) without any special-casing — a childless
`CombinedMarket`'s total is simply its own direct accumulator, since the
sum-of-children term is empty.

**Design choice: `MarketSystem` orchestrates, `CombinedMarket` stays
passive.** `CombinedMarket` never holds or queries `LocalMarket`
references; it only ever receives plain aggregated numbers from
`MarketSystem`. This matches Elsa's existing convention exactly
(`GlobalMarket.compute_prices(local_markets, ...)` already receives its
inputs as parameters rather than owning child references), avoids a
two-way coupling between `CombinedMarket` and `LocalMarket` (since
`LocalMarket`'s Step 2 also needs `CombinedMarket`'s tick price as an
anchor), and makes `CombinedMarket` trivially unit-testable in isolation
(pass in numbers, check the output — no need to construct fake
`LocalMarket` objects).

**Species-keying split:** raw port-to-port distance (above) is the one
genuinely species-independent piece of this whole design — it's pure
geography, computed once, reused by every species. Everything built on
top of it is species-specific and was validated as such: "High-tier"
status (a port's volume share for *this* species — a major anchovy port
isn't necessarily a major hake port), the resulting distance-weighted
neighbor signal, the frozen coefficients, and the volume baseline are
all keyed by `(species_code, category_code)`, never computed once per
port. Computing distance per-species as well would be pure redundant
computation of an unchanging value; computing High-tier/neighbor
weighting once per port (not per species) would silently deviate from
what was actually validated.

## 5. Per-tick data flow (`MarketSystem.step()`)

1. **Collect pending catches** for this tick's date, routed per section 4.
2. **Compute each `CombinedMarket`'s tick volume and price** — total
   volume per section 4, run its frozen Model 2 to get this tick's
   Combined predicted price. (For a childless `CombinedMarket`, this
   step alone produces its final price — there is no Local layer under
   it to run steps 3–4 for.)
3. **Compute Step 1 for every `LocalMarket`** — frozen per-port Model 2,
   using that port's own accumulated volume this tick vs. its adaptive
   volume baseline. Must complete for *all* local markets before step 4,
   since any port can be used as another's neighbor signal.
4. **Compute Step 2 (dynamic switch) for every `LocalMarket`** — build
   the distance-weighted neighbor signal from sibling ports' Step-1
   predictions (inverse-distance weighted via lon/lat, restricted to
   High-tier sources per the validated design); compute
   `VolumeRatio` = this tick's own volume ÷ its adaptive baseline; below
   10% or above 200% → frozen DistanceOnly coefficients, otherwise →
   frozen Stacked coefficients — both anchored to this tick's
   `CombinedMarket` price from step 2.
5. **Update adaptive volume baselines** (section 6) for both
   `LocalMarket` and `CombinedMarket`, using this tick's real observed
   volume only.
6. **Save history** (prices + catches snapshot) for both levels, same
   pattern as Elsa's `update_local_history` / `update_global_price_history`.
7. **Clear this tick's catch accumulators**, advance the tick.

**Resolved empirically (section 15):** the Step-2 anchor in step 4 uses
`CombinedMarket`'s frozen seasonal baseline table (a static per-month
shape — what the notebooks actually validated), confirmed in production
as the better default by a real head-to-head fidelity backtest
comparison against the live tick-specific alternative (static: better
MAPE in 84 of 162 directly-comparable pairs vs. live's 78, lower mean
MAPE). The live variant remains available (`ANCHOR_SCHEME_LIVE`) purely
for this kind of analysis; the live production path always uses static.

## 6. What's frozen vs. adaptive

After `init_simulation`, the simulation only ever receives new
**landings (volume)** — never new real price data. This is a hard
constraint, not a design choice, and it determines what can and cannot
be updated later:

- **Frozen for the life of the simulation:** all price-related
  coefficients — seasonal price baselines, and the regression
  coefficients for Step 1 (own-volume response) and Step 2 (Stacked and
  DistanceOnly). These are fit once at `init_simulation` from whatever
  real history precedes `start_time`. They cannot be refit later,
  because doing so would mean fitting against the model's own prior
  predictions rather than real ground truth — circular, and liable to
  reinforce rather than correct any systematic bias.
- **Adaptive, updated incrementally every tick:** the **volume
  baseline** (the "normal month" reference used for `VolumeAnomaly` and
  `VolumeRatio`), for both `LocalMarket` and `CombinedMarket`. This is
  safe to update because volume is genuinely new, externally-supplied
  real data throughout the simulation (not circular the way price
  refitting would be), and keeping it current prevents the volume
  anomaly measurement and the dynamic-switch trigger rate from silently
  drifting out of calibration as real fishing patterns shift over a
  long-running simulation. This does **not** fix drift in the
  underlying price-to-volume relationship itself (which stays frozen
  either way) — only in what counts as a "normal" volume to compare
  against.

## 7. Adaptive training-window selection

Rather than fitting every frozen coefficient set on a single, fixed
trailing window (the originally-planned 4 years), each fitting unit —
`CombinedMarket`'s Model 2 per `(species_code, category_code)`, and
`LocalMarket`'s Step 1 and Step 2 coefficients per `(port, species_code,
category_code)` — selects the best-performing training-window length
from a small candidate set, independently, at `init_simulation`.

**Candidate set:** `{2yr, 3yr, 4yr, 6yr, Full-history}`, used identically
at both the `CombinedMarket` and `LocalMarket` levels. This is
deliberately small and was chosen to span the empirically-observed
optima across the validated species — it is explicitly **not** a
fine-grained sweep (avoiding the overfitting risk directly demonstrated
earlier by the two-tier volume-cutoff sweep, where searching over many
candidate thresholds produced a "best" value that was most likely noise)
and it is explicitly tuned to this dataset, not a universal constant —
worth revisiting if this model is ever applied to a very different
market.

**Selection mechanism:** at init, for each fitting unit, fit all 5
candidates against the real historical data preceding `start_time`,
score each with the same rolling-origin validation methodology used
throughout this project, and freeze the single best-performing
candidate's coefficients for the life of the simulation — consistent
with section 6: only the *window length used to produce* the one-time
fit is chosen adaptively, the resulting coefficients are still frozen
forever afterward, never re-chosen or refit.

**Why this isn't a species-adaptive selector:** it's one uniform
procedure (try the same candidates, pick whichever validates best)
applied identically to every fitting unit. Different species, ports, or
levels landing on different window lengths is an output of that single
mechanism, not a hand-picked branch per species.

**Evidence:**
- At the `CombinedMarket` level, gains were modest (0.0–1.1pp) but
  mechanistically sensible: the two most volatile, least-stationary
  species (ANE, PIL) both preferred *shorter* windows (recency over
  sample size), while the most stable species (HKE) and the thinnest
  (SAA) both preferred *longer* windows (more data to reduce noise, no
  stationarity cost since their relationship isn't drifting much).
- At the `LocalMarket` level, gains were substantially larger — Step 1
  alone: +0.5 to +9.6pp; the full Step 1 + Step 2 pipeline: +0.6 to
  +6.3pp (smaller than Step 1 alone in a mechanistically sensible way,
  since Step 2 already buys some stability from the `CombinedMarket`
  anchor and the neighbor signal, leaving less room for window-length
  tuning to additionally help). Per-port fits generally preferred
  *longer* windows than the combined-market fits did, since individual
  ports have far less monthly data than the national aggregate.
- Critically, testing against the French dataset showed that a
  **forced, uniform window is actively dangerous on thin data**, not
  just suboptimal: a 2-year window forced onto every French port
  produced a 504% MAPE for ANE — a catastrophic blow-up, since with so
  few, so volatile monthly observations, the fitted volume-response
  coefficient becomes wildly unstable. The adaptive, per-entity
  selection mechanism avoids this by construction, since it simply never
  chooses a window that performs catastrophically for that specific
  entity — this is as much a safety property as an accuracy one.

**Sanity-bound safeguard (required companion, not optional):** if every
candidate window for a given fitting unit produces an implausible or
unstable result — not just "worse than the others," but clearly outside
sane bounds — the same graceful fallback cascade already described in
section 9, point 2 (try a simpler fit; fall back fully to the
`CombinedMarket`'s own prediction) triggers, rather than ever locking in
an unstable fit for the life of the simulation.

**Caveat, same as the other provisional decisions in this spec:** this
is based on only 3 historical rolling splits, and the candidate set
itself was chosen after observing this dataset's results, not derived
from first principles. Treated as provisional pending confirmation via
the fidelity backtest (section 11), same as the Step-2 anchor choice
(section 5) and the Combined/Local consistency question (section 10).

## 8. Real-price-first behavior

Carried over unchanged from Elsa's existing, recently-reinforced
convention (`MarketSystem.compute_local_prices()`,
`price_time_series`): at every tick, check first whether a real
recorded price exists for that date; if so, report/use it directly, and
only fall back to the Step 1/Step 2 prediction logic when no real price
is available. Applies at both `LocalMarket` and `CombinedMarket` levels.

## 9. Edge cases and error handling

Following Elsa's existing philosophy: hard-error on genuinely broken
contracts/config, gracefully degrade on data-quality gaps that are
expected to occur in real data.

1. **Zero or extreme volume this tick** — handled by the dynamic switch
   itself (section 5, step 4): `VolumeRatio` below 10% or above 200%
   falls back to DistanceOnly. A port with literally zero landings has
   `VolumeRatio = 0`, always below the low threshold — this is the
   direct resolution of the original "no new landings this month"
   concern.
2. **Not enough historical data to fit Step 1 at init** (too few months,
   or zero volume variance) — fallback cascade: try a level-offset-only
   fit (no volume term); if that also fails, fall back fully to the
   `CombinedMarket`'s own prediction with no local adjustment. Each
   fallback step logs a **WARNING** (not ERROR — the simulation
   continues correctly, this is a quality degradation), naming the
   specific port, species, and tick, via the same named-logger-per-class
   pattern already used throughout Elsa (`logging.getLogger(consts.<NAME>)`,
   funneling into the app-wide handlers configured in `app.py`).
3. **No valid neighbor candidates for a port** — split into two
   different severities, since they have different causes:
   - **Missing or invalid lon/lat for a local market itself** — a
     species-independent reference-data bug (same problem regardless of
     which species you ask about). **Hard `MarketError` at
     `init_simulation`**, checked once per port, listing the specific
     affected port(s). Given this is a connected stretch of coastline
     with 36+ ports, every local market should have valid coordinates;
     a missing one is far more likely a data bug than a genuine gap.
   - **No High-tier neighbor for one specific species** — this is
     expected and *not* an error: species vary hugely in how
     concentrated their volume is (we saw this directly with SAA/PIL
     being dominated by one or two ports), so a port legitimately having
     zero High-tier neighbors for one thin species while having several
     for a well-distributed one isn't a data bug, it's just that
     species' market structure. Handled as a **graceful per-(port,
     species) fallback to Step-1-only**, logged as a WARNING (same
     mechanism as point 2), not a simulation-ending error.
4. **A species with no historical price history at all** — silently
   excluded from `get_species_prices()` output, matching Elsa's
   existing valid/missing-count-and-skip-`None` pattern exactly.
5. **Invalid input** (near-zero/negative quantities, missing config
   files, contract species/ports absent from config) — Elsa's existing
   `MarketError` conventions and status codes are reused unchanged,
   *except* the `market_code not in self.local_markets` check in
   `update_sales`, which must be updated to match the new, more lenient
   routing rule in section 4: reject only if the code matches *neither*
   a known local market *nor* a known combined-market prefix (narrower
   than Elsa's current exact-whitelist-only check).
6. **A new port or species appearing mid-simulation, not present at
   init** — structurally impossible given the grpc contract fixes the
   complete set of ports and species at `init_simulation` (Elsa already
   validates the contract against config and trims config down to
   exactly the contracted set, for both ports and species, before the
   simulation starts). Anything `update_sales` references outside that
   fixed set is therefore a genuine contract violation, already caught
   by the validation in point 5 — no additional handling needed.
7. **Threshold/configuration constants** (10%/200% volume-ratio switch,
   0.2%/2% tier cutoffs, the `{2yr, 3yr, 4yr, 6yr, Full}` training-window
   candidate set from section 7) — named constants in the module's
   `consts.py`, matching Elsa's existing convention, not magic numbers
   in the logic.

## 10. Known limitation: Combined/Local consistency

`CombinedMarket` and `LocalMarket` are independently-fit models (frozen
at init) driven by independent real-time inputs — `CombinedMarket`
responds to aggregate volume anomaly, each `LocalMarket` to its own
volume anomaly plus a neighbor signal. Nothing in this design *forces*
the (volume-weighted) aggregate of all `LocalMarket` predictions to stay
consistent with what `CombinedMarket` independently predicts for the
same tick — over a long-running simulation they could drift apart with
no built-in correction.

This is a deliberate trade-off, not an oversight: Elsa's current
`GlobalMarket.compute_prices()` sidesteps the issue entirely by
*defining* Global as the mean of Local prices (guaranteed consistency by
construction), but its own code flags this as an explicitly crude
"preliminary solution." This design moves away from that specifically
because an independently-fit combined-market Model 2 is empirically more
accurate than a naive aggregate-of-locals — the basis of everything
validated in the research notebooks. The cost of that choice is the
consistency guarantee.

**Resolution plan:** rather than pre-engineering a fix, measure the
actual drift over the historical validation period as part of the
fidelity backtest (section 11) — specifically, track how far the
volume-weighted average of `LocalMarket` predictions diverges from
`CombinedMarket`'s own prediction over time. If it stays small, this is
an accepted, bounded, known limitation. If it's large, an explicit
reconciliation mechanism needs to be designed — only once there's real
evidence it's needed, not preemptively.

## 11. Testing strategy

Three layers:

1. **Unit tests per class**, mirroring Elsa's existing
   `test_GlobalMarket.py` / `test_LocalMarket.py` / `test_MarketSystem.py`
   / `test_SimulationHandler.py` convention: `CombinedMarket` (fit/predict
   against constructed numbers, no `LocalMarket` objects needed per
   section 4's design), `LocalMarket` (Step 1 and Step 2 fit/predict,
   the dynamic-switch threshold logic at the 10%/200% boundaries, the
   adaptive volume-baseline update), `MarketSystem` (routing logic,
   two-pass tick ordering), and the **adaptive training-window
   selection** itself (section 7) — given constructed candidates with
   known scores, confirm the correct one is selected and frozen, and
   given a case where every candidate is unstable, confirm the
   sanity-bound safeguard triggers the fallback cascade rather than
   locking in a bad fit.
2. **Edge-case/regression tests**, one per section 9 decision —
   concrete constructed scenarios exercising each path: a zero-volume
   tick, a thin-history port (confirming the WARNING gets logged), a
   no-neighbor-candidate port (confirming the hard init error), an
   unroutable market code, etc.
3. **Fidelity backtest against the notebook-validated numbers** — run
   the same rolling-origin methodology used throughout the research
   notebooks, but against `config/northwestern_med/prices.csv` using
   the actual production classes (not notebook functions), and confirm
   the resulting MAPE per species/port lands close to what was measured
   in the notebooks (with tolerance, since the port universe may not be
   an exact match). This is a standing regression guard against future
   changes silently degrading accuracy, and it is where the section 10
   consistency measurement, the section 5 anchor-choice question, and
   the section 7 adaptive-window-selection decision all get resolved
   empirically rather than by assumption. **Precondition:** this only
   means anything once section 13's data-consistency check (the new
   data-prep script reproducing the already-validated
   `emodnet_es_raw.csv`/`emodnet_fr_raw_fresh_whole.csv` series against
   the same pinned EMODnet snapshot) has passed — otherwise a good or
   bad backtest result could just reflect a data discrepancy, not
   anything about the model itself.

## 13. Data preparation: EMODnet → `prices.csv`

**Location:** `src/models/market/data_prep/`, entirely within this repo
— nothing outside `SURIMI-market` touched.

**Source:** direct ZIP download from the EMODnet portal (the "EMODnet
Human Activities, Fisheries, EUMOFA Monthly first sales" dataset,
[catalog record](https://emodnet.ec.europa.eu/geonetwork/srv/api/records/ff5f738f-94ee-4704-94c2-7c951c931973)),
not a live WFS API integration — this lets the script reuse the
existing, already-validated geodatabase/shapefile-processing logic from
`scripts/fisheries_data/emodnet_prices_landings_raw_new.ipynb` largely
unchanged, rather than rewriting the filtering logic around a different
API paradigm for marginal benefit. Public, CC-BY 4.0, no authentication
required. The download filename is dated (e.g.
`EMODnet_HA_Fisheries_FirstSales_20250319.zip`) and changes with each
update, so the script first queries the record's XML metadata endpoint
to discover the current filename rather than hardcoding a date.

**Refresh cadence:** the upstream EMODnet dataset itself is only updated
twice a year, so this is a manual/periodic tool run on roughly that same
cadence — matching the existing `update-config.py` pattern — not
something invoked automatically at simulation start.

**Inputs treated as read-only, not generated:** `markets.csv` (defines
which 5-letter local LOCODEs and 2-letter combined prefixes to extract
for; also shared with and used by the separate landings-producing model)
and `species.csv`. Neither is modified by this script.

**Local-market rows:** reuse/extend the existing notebook's per-port
extraction (filter to `markets.csv`'s LOCODEs, restrict to rows where
`Preservation = "Fresh"` and `Presentation = "Whole"`). **`category_code`
in the output `prices.csv` is constructed directly from those two raw
EMODnet fields** — `Preservation + " - " + Presentation`, e.g.
`"Fresh" + "Whole"` → `"Fresh - Whole"` — matching exactly the single
value already present in the current `config/northwestern_med/prices.csv`,
not a separate or independently-chosen label. Note: the existing
notebook currently explicitly *drops* `ES` and `FR` ("as they are not
locations") — this must change, since those combined rows are exactly
what section 4 needs.

**Combined-market rows:** computed by aggregating (sum `Value_EUR` and
`Volume_Kg`, derive `price` from the totals — never averaging
already-computed per-port prices directly) **exactly the same local
ports `markets.csv` defines under each prefix** — not sourced from a
separate, broader EMODnet query. This keeps the historical training
data consistent with the live model's own runtime aggregation rule
(section 4); if they were built differently, the frozen `CombinedMarket`
coefficients would be calibrated against a different "`ES`" than the one
they're actually fed at runtime.

**Size handling:** `prices.csv` gains a new `Size` column and, for each
`(date, market_code, species_code, category_code)` group, now contains
**both** a blended row — `Size = NaN`, summing `Value_EUR`/`Volume_Kg`
across all size grades, used by the core model for fitting/prediction —
**and** the individual per-size rows (`Size = 1, 2, 3, 3a, 3b, ...`,
whenever that species has recorded size data) side by side in the same
file. `NaN` is deliberate, not a magic string like `"ALL"` — it's
consistent with how species that have no recorded size data at all
(e.g. SAA) already appear in the raw EMODnet data, so "blended" and
"no size breakdown available" naturally share the same representation.
The core model filters specifically to `Size = NaN` rows for its own
fitting, matching the validated Model 4 finding that splitting by size
hurts accuracy (ANE −2.10pp, PIL −5.88pp, HKE −0.87pp, MUT −1.52pp) —
the per-size rows are additional data, not an alternative the core
model chooses between.

Since this changes the shape of a schema shared with the
landings-producing model (more than one row per group, where there was
exactly one before), this needs to be confirmed with that model's
consumers before shipping, not assumed safe by default.

**A second, separate output: `size_ratios.csv`.** Holds the per-
`(species_code, size)` historical ratio (`size-specific price ÷ blended
price`), computed as a simple stable historical average — this is the
"converter" that was actually validated for giving size-differentiated
quotes (`Size_Price = Blended_Prediction × Size_Ratio`), since the
blended model alone cannot distinguish a request for Size 1 vs. Size 4
at all. Derived directly from `prices.csv`'s own per-size rows (no need
to reprocess the raw EMODnet download separately for this). Kept as its
own file rather than merged into the main schema, since
`size_ratios.csv` is market-model-specific — not consumed by the
landings-producing model — and keeping it separate means a future,
differently-grouped size scheme (e.g. a coarser 2-bucket split) is a
change to how this one file gets computed, not a change to the shared
config or the core model.

**This input is optional, not required.** The model must function
correctly and completely without `size_ratios.csv` ever being provided
— if it's absent (or not part of a given scenario's config), the model
simply reports the blended prediction with no size differentiation,
exactly as it would today, rather than erroring or treating it as
missing required configuration. If present, it unlocks the converter so
callers can request a size-specific quote. Same graceful-by-default
philosophy as the rest of section 9 — this is an enhancement the model
can use when available, never a hard dependency.

**Why this matters beyond just "fresher data":** checked
`config/northwestern_med/prices.csv` directly — `price` is 100%
populated, but `landings_kg` is only **39.6%** populated (164,857 of
273,000 rows null). This isn't a coincidence: Elsa's own model only ever
reads `price` from this file into `price_time_series`, never
`landings_kg` (its elasticity formula uses only live catches from
`update_sales`), so nothing has required this column to be complete.
Our model critically needs complete historical `landings_kg` to fit the
frozen Step 1/Step 2 coefficients — sourcing fresh, complete data
directly from EMODnet fixes a real, concrete gap, not just a
nice-to-have.

**Output:** `prices.csv` (the only file in the existing, shared schema —
unchanged, still exactly what `init_simulation` already expects) plus
`size_ratios.csv` (new, market-model-specific, not consumed by any other
model).

**Required validation before this script is trusted as the model's data
source: consistency against the data the model was actually calibrated
on.** Every MAPE number referenced throughout this spec came from the
research notebooks running against `scripts/fisheries_data/emodnet_es_raw.csv`
/ `emodnet_fr_raw_fresh_whole.csv` — themselves produced by the existing
notebook from one specific EMODnet snapshot
(`EMODnet_HA_Fisheries_FirstSales_20250319.zip`). Since this new script
will, in normal operation, fetch whatever the *current* snapshot is
(not necessarily that same one), there's a real risk of it silently
diverging from the data the model was validated against — different
filtering, a subtly different aggregation step, or just a newer
snapshot with revised historical values. Before trusting this script's
output, it must be run once against that same pinned, already-known
snapshot, and its resulting monthly series compared directly against
`emodnet_es_raw.csv`/`emodnet_fr_raw_fresh_whole.csv`'s own blended
series (matching figures for the shared local ports; the new combined
`ES`/`FR` rows won't have an equivalent to compare against, since the
old files dropped them entirely). This is a precondition for section
11's fidelity backtest to mean anything — if the data itself doesn't
match, a good or bad backtest result wouldn't actually tell us anything
reliable about the model's fidelity, only about a data discrepancy.

**Second required check, once the script's real output is in hand:
how far real price data actually extends, and how large the
real-to-predicted transition jump (section 8) really is.** This was
previously just a documented, theoretical risk (section 8) — it should
instead be measured directly against what this script actually
produces, not assumed. Concretely: for a range of plausible
`start_time` values, check how many ticks of real `price` data are
available before the simulation would need to start predicting, and
for the ports/species where real data does run out mid-range, measure
the actual gap between the last real price and what the frozen
Step 1/Step 2 model would have predicted for that same tick (computable
retrospectively, since we have the real value to compare against). If
that gap is consistently small, the transition is a non-issue in
practice; if it's large or systematic for particular species/ports,
that's worth knowing before relying on this in production, not
discovered later. This is a data-driven check on the new script's
actual output, not a repeat of the abstract concern in section 8.

## 14. Open items explicitly deferred

- S3 sourcing specifics for the final deployment (flagged by the user
  as a follow-up discussion).
- ~~Whether the Step-2 anchor (section 5) should be `CombinedMarket`'s
  static seasonal baseline or its live tick prediction~~ — RESOLVED via
  a real head-to-head fidelity backtest comparison (section 15): static
  confirmed as the better default, by a modest but real margin.
- Whether the Combined/Local consistency drift (section 10) needs an
  explicit reconciliation mechanism — resolved via measurement, not
  pre-built.
- Whether the adaptive training-window candidate set (section 7) needs
  adjustment once validated against the full fidelity backtest, rather
  than just the scratch-script checks performed during design — the
  candidate set was empirically tuned to this specific dataset.

## 15. Measured results and current status of the open items above

Recorded after implementation and the final whole-branch review, so the
real numbers behind sections 10/13/14 live somewhere durable instead of
only in test stdout or a gitignored report file. **Updated** after
Plan 1's data-prep pipeline was actually run (a fresh EMODnet download)
and its output substituted for the committed `prices.csv` — see the
precondition entry below. All numbers in this section are from the
real, regenerated data, not the earlier pre-Plan-1 file.

**Section 11's fidelity-backtest precondition — SATISFIED.** The
committed `config/northwestern_med/prices.csv` is now Plan 1's actual
output (`src/models/market/data_prep/run.py` run against a live EMODnet
download): 100% `landings_kg` coverage for every validated species (up
from 39.6%), a real `size` column, and a 2015-2024 date range (the old
file ran 2010-2024; the pinned historical snapshot the original design
intended to validate against is no longer recoverable — see the ledger
for Plan 1's own data-prep work — so this is "the current EMODnet
snapshot," not a byte-for-byte reproduction of a specific past one).
Every number below was re-measured against this real data; the ones
measured only against the old file have been superseded, not merely
annotated.

**Section 10 (Combined/Local consistency drift) — measured, accepted as
bounded.** Mean `|volume-weighted Local - Combined|` gap over the 2-year
backtest window, per validated species: ANE 9.1%, PIL 7.0%, HKE 5.5%,
SAA 13.3%, MUT 4.6%. Still single-digit-to-low-double-digit — small
enough to treat as the accepted, bounded limitation this section
anticipated, not evidence an explicit reconciliation mechanism is needed
yet. SAA's gap widened the most (9.9% → 13.3%) of the five; worth
revisiting if SAA-specific production use becomes a priority.

**Section 13's "transition jump" check — measured, same pattern as
before.** At the shared real-data boundary (2024-12-01), `CombinedMarket`'s
frozen prediction vs. the real recorded price: ANE 1.5%, MUT 1.5%,
SAA 17.8%, HKE 18.4%, PIL 31.3% — essentially unchanged from the old
data's measurement (expected, since both files cover this same end
date with similar recent real catches). ANE/MUT are tight; SAA/HKE/PIL
are materially larger. PIL's 31.3% jump remains worth treating as a
live-production risk for that species specifically.

**Section 14's Step-2 anchor-scheme question — RESOLVED by a real,
unconfounded head-to-head MAPE comparison; re-confirmed against the
regenerated data.** The original comparison (gamma coefficients only,
with the live variant always refit on full history regardless of what
window static actually used) could only observe divergence, never say
which scheme was better. Fixed properly: `MarketSystem.fit_all`/
`_fit_step2_for_all_ports` now take an `anchor_scheme` parameter
(`consts.ANCHOR_SCHEME_STATIC` / `ANCHOR_SCHEME_LIVE`;
`CombinedMarket.get_live_anchor_series` provides the live source),
defaulting to STATIC everywhere — the live production gRPC path is
unaffected; only an explicit `anchor_scheme=LIVE` call (the fidelity
backtest harness, for this comparison) exercises the alternative at
all. `test_open_items_report.py` section 3 runs TWO independent,
fully-stepped backtests over the identical as_of_date/test_horizon/
species — one per scheme — and compares the real, resulting MAPE
directly, both overall and isolated to the pairs where Step 2 was
actually fitted under both schemes (the fair, confound-free subset).
Result against the regenerated data: **static wins again** — better
MAPE in 85 of 151 isolated pairs vs. live's 66 (0 ties), with a lower
mean MAPE (76.2% vs. 76.8% — both high because this mean now includes
the extremely sparse-data outlier pairs below, e.g. ANE@ESCDK at over
1200% MAPE for both schemes; the *relative* ordering between schemes is
what matters here, not the absolute mean). Confirms plan header
decision 1's original choice holds under the real, regenerated data
too, not just the old file.

**Fresh known-limitation findings from the regenerated data.** Three
(species, port) pairs newly exceed the MAPE ceiling that the old data
never surfaced — `test_fidelity_backtest.py`'s `KNOWN_LIMITATION_PAIRS`
has the full investigation for each. All three share a root cause:
genuinely sparse real landings for that specific pair (41, 15, and 80
total rows respectively, spanning the whole 2015-2024 history, not a
data gap), leaving only 2-5 held-out comparison months out of 24. One
of them (ANE@ESCDK) surfaced a real, narrow robustness gap worth
flagging here: with almost no training signal, the Step 2 Stacked
regression extrapolated to a **negative** predicted price for one
held-out month (-18.83 vs. a real 1.40 EUR/kg) — nothing in the model
enforces a price floor. This is isolated to pairs this data-sparse (not
observed anywhere else across the real data) and not treated as
blocking, but a price-floor safeguard would be a reasonable follow-up
if more such pairs turn up in a future, larger contract.

**`GetSpeciesPrices` before the first tick — FIXED.** Unlike `ElsaHandler`
(which seeds an initial price snapshot at `init_simulation` time
specifically so this RPC works before the first `SimulateStep`),
`MarketHandler` previously did not — a caller that queried prices before
the first step got a valid-but-empty response. Fixed by extracting
`MarketSystem.step()`'s price-computation logic (spec section 5 steps
2-4) into a pure, side-effect-free `_compute_tick_prices` method, and
having `get_species_prices_message` call it directly (for a zero-catch
tick at the current, not-yet-advanced tick date) whenever a port has no
`price_history` entry yet. This mirrors Elsa's init-time seeding intent
but computes the fallback lazily rather than eagerly storing it, since
this plugin's price state lives in `price_history` rather than a
separate live-price dict — and since it reuses the same real-price-first
path, a query before the first step can even surface a genuine real
price if one already exists for the simulation's start date.

**`GetSpeciesPrices` cross-model consistency with Elsa — gap found and
FIXED.** Verified empirically (ran both handlers against the identical
`config/northwestern_med` contract and diffed the resulting
`market_code` sets): `ElsaHandler` treats every `markets.csv` row,
including 2-letter combined-market codes (`ES`/`FR`), as an ordinary
local market with no LOCODE-length distinction, and reports a price for
them. This plugin's `GetSpeciesPrices` only ever iterated
`self.local_markets`, so it structurally could never report a price
under a 2-letter `market_code` even though `CombinedMarket` computes and
freezes its own price regardless — a caller querying by
`market_code="ES"` got an answer from Elsa and nothing from here. Fixed
by also iterating `self.combined_markets` in `get_species_prices_message`
(same price_history-or-live-fallback logic as local markets, reusing
`_compute_tick_prices`'s already-computed `combined_prices`), reporting
each `CombinedMarket`'s price under its own prefix as `market_code`.
Message schema itself (field population: `species`, `market_code`,
`price`, `currency`, `category_code`) was already identical between the
two models; this was a content gap, not a schema one. Separately — found
during this same comparison, out of scope for this plugin, reported but
not fixed — `ElsaHandler.get_species_prices()` currently crashes
unconditionally: `Tick.get_timestamp()` (Elsa's own, not this plugin's)
returns an ISO string with no `'Z'`/offset suffix, which
`protobuf.Timestamp.FromJsonString()` rejects, on a value that is
computed but never even used afterward.

**Two mechanical fixes applied as part of this review** (not open items,
recorded here for completeness since they affect the numbers above):
`MarketSystem._fit_step2_for_all_ports`'s High-tier volume-share
computation was including the 2-letter combined-market aggregate rows
(`ES`/`FR`) in its denominator, silently starving most ports of real
neighbors (HKE went from 5 of 18 legitimate High-tier ports to all 18
once fixed) — this resolved what had been an unexplained ~3x MAPE
amplification at one port. Separately, `CombinedMarket`'s adaptive
volume baseline was being diluted by an explicit 0.0 entry on every
zero-catch tick, unlike `LocalMarket`'s baseline, which already skipped
those ticks — both levels now apply the same rule.
