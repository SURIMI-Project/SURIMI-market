# Supply-Driven Seasonal Price Model — Architecture

## 1. Overview

The Supply-Driven Seasonal Price Model supplies first-sale fish prices to
the dynamic system formed by the coupled SURIMI models. At every
simulation step, it takes the landings reported by the other SURIMI
models (POSEIDON, EwE, Mice) and returns a price per species for
individual markets at local ports and accumulated prices at the national
level.

Each price is built up in layers:

- **Season:** each species follows its typical seasonal price pattern.
- **Supply:** prices shift when landings/catches depart from their
  seasonal baseline.
- **Trend:** at the national level, the model projects the long-run price
  trend forward: the more statistically certain the trend, the stronger
  its effect on future prices.
- **Local ports:** prices at individual local ports draw on the national
  price and on nearby ports, so thinly traded ports still receive a
  sensible price.

The model is fitted to historical first-sale prices derived from EMODnet
(`src/models/market/data_prep/`). When the simulation reaches a month for
which a real price was recorded, it reports that recorded price as-is;
for months without historic prices, it reports the model's prediction.

The model is implemented in the price simulation framework
(`src/models/market/`) and registered with its simulation handler
(`src/models/market/MarketHandler.py`).

---

## 2. Architecture

### 2.1 Domain Model

```mermaid
classDiagram
    class MarketHandler {
        -simulation_id: str
        -market_system: MarketSystem
        -size_ratios: DataFrame
        +init_simulation(simulation, root_config_dir)
        +update_sales(request)
        +simulate_step(controller_simulation_time)
        +get_species_prices()
        +finalise_simulation(folder_path)
    }

    class MarketSystem {
        -tick: Tick
        -combined_markets: Dict~str, CombinedMarket~
        -local_markets: Dict~str, LocalMarket~
        -prefix_to_locals: Dict~str, list~
        -distance_lookup: Dict~tuple, float~
        -donor_candidates_by_key: Dict~tuple, set~
        +setup_topology(markets_df)
        +fit_all(prices_df, latest_date, anchor_resolver)
        +step()
        +get_species_prices_message()
        +export_results(simulation_id, folder_path)
    }

    class Tick {
        -tick: int
        -start_time: datetime
        -step_months: int
        +increase_tick()
        +get_tick() int
        +get_datetime() datetime
    }

    class CombinedMarket {
        -market_code: str
        -models: Dict~tuple, PriceModel~
        -window_labels: Dict~tuple, str~
        -volume_baseline: Dict~tuple, VolumeBaselineTracker~
        -real_prices: RealPriceLookup
        +fit(monthly_by_key, latest_date)
        +get_price(species_code, category_code, date, volume) float
        +get_baseline_price_table(species_code, category_code) Series
        +add_direct_catch(species_code, category_code, quantity)
    }

    class LocalMarket {
        -market_code: str
        -models: Dict~tuple, PriceModel~
        -neighbor_models: Dict~tuple, StackedModel~
        -volume_baseline: Dict~tuple, VolumeBaselineTracker~
        -real_prices: RealPriceLookup
        +fit_base_models(monthly_by_key, latest_date)
        +fit_neighbor_models(species_code, category_code, inputs: NeighborFitInputs)
        +predict_from_base_model(species_code, category_code, date, volume) float
        +predict_from_neighbor_model(species_code, category_code, date, volume, anchor_value, neighbor_value) float
        +get_real_price(species_code, category_code, date) float
        +add_monthly_catch(species_code, category_code, quantity)
    }

    MarketHandler "1" o-- "1" MarketSystem
    MarketSystem "1" o-- "1" Tick
    MarketSystem "1" o-- "many" CombinedMarket
    MarketSystem "1" o-- "many" LocalMarket
    LocalMarket "many" --> "1" CombinedMarket : anchors on (by 2-letter prefix)
    LocalMarket "many" --> "many" LocalMarket : neighbor signal (distance-weighted)
```

| Component | Responsibility |
|---|---|
| `MarketHandler` | The gRPC-facing entry point. Translates each request into a call on `MarketSystem` and back into a response. |
| `MarketSystem` | Orchestrates every `CombinedMarket`/`LocalMarket`. It sets up the market structure, fits the price models, and drives the simulation forward one tick at a time, computing and recording each tick's prices. |
| `Tick` | The simulation's current date and tick counter. |
| `CombinedMarket` | The national-level price model for one 2-letter market, with a trend correction. |
| `LocalMarket` | The port-level price model for one 5-letter port, with its own base and neighbor-informed models. |

### 2.2 Price Models and Shared Components

```mermaid
classDiagram
    class PriceModel {
        <<interface>>
        +predict(date, volume) float
        +predict_series(monthly) Series
        +update_baseline_volume(month, new_mean)
    }
    class LevelVolumeModel {
        -baseline_price: Series
        -baseline_log_price: Series
        -baseline_volume: Series
        -alpha: float
        -beta: float
        -vol_anom_min: float
        -vol_anom_max: float
        -trend: TrendCorrection
        +predict(date, volume) float
        +predict_series(monthly) Series
        +update_baseline_volume(month, new_mean)
    }
    class TrendCorrection {
        -confidence: float
        -rate: float
        -years_trained: float
        -train_end: Timestamp
        +identity(train_end) TrendCorrection
        +factor(date) float
        +factor_series(dates) Series
    }
    class LevelOnlyModel {
        -baseline_price: Series
        -alpha: float
        -trend: TrendCorrection
        +predict(date, volume) float
        +predict_series(monthly) Series
    }
    class StackedModel {
        -delta: float
        -beta: float
        -gamma: float
        -port_baseline_volume: Series
        -neighbor_baseline: Series
        -vol_anom_min: float
        -vol_anom_max: float
        +predict(date, volume, anchor_value, neighbor_value) float
        +update_baseline_volume(...)
    }
    class VolumeBaselineTracker {
        +seed(monthly, train_start, train_end)
        +update(date, volume)
        +ratio(date, volume) float
        +snapshot() dict
    }
    class RealPriceLookup {
        +load(prices_df, market_code)
        +get(species_code, category_code, date) float
    }
    class CatchAccumulator {
        +add(species_code, category_code, quantity)
        +take() dict
        +clear()
        +snapshot() dict
    }

    PriceModel <|.. LevelVolumeModel
    LevelVolumeModel o-- TrendCorrection
    PriceModel <|.. LevelOnlyModel
    CombinedMarket o-- PriceModel : models
    LocalMarket o-- PriceModel : models
    CombinedMarket o-- VolumeBaselineTracker
    LocalMarket o-- VolumeBaselineTracker
    CombinedMarket o-- RealPriceLookup
    LocalMarket o-- RealPriceLookup
    CombinedMarket o-- CatchAccumulator
    LocalMarket o-- CatchAccumulator
    LocalMarket o-- StackedModel : neighbor_models
```

| Component | Responsibility |
|---|---|
| `PriceModel` | The common interface every price model implements, so `CombinedMarket` and `LocalMarket` can treat any of them the same way. |
| `LevelVolumeModel` | The seasonal-baseline-plus-volume-response model, kept in sync with real landings/catches as the simulation runs. |
| `TrendCorrection` | The long-run price trend applied on top of a `LevelVolumeModel`; ports use a no-op version instead of a real trend. |
| `LevelOnlyModel` | The no-volume-term fallback, used when there isn't enough data to support a volume response. |
| `StackedModel` | `LocalMarket`'s neighbor-informed model, blending a port's own data with its nearest geographic neighbors. |
| `VolumeBaselineTracker` | Tracks the landings/catches for one (species, category) key by calendar month, kept in sync every tick. |
| `RealPriceLookup` | Looks up a market's historic prices. |
| `CatchAccumulator` | Collects the landings/catches reported between ticks. |

### 2.3 Simulation Lifecycle

The framework layer above this (Controller → App → `MarketService`, shared
across every model plugin) is documented in `market_architecture.md`
Section 3. This diagram starts at `MarketHandler`, where `market`-specific
behavior begins.

```mermaid
sequenceDiagram
    participant MPS as MarketService
    participant MH as MarketHandler
    participant MS as MarketSystem
    participant CM as CombinedMarket[*]
    participant LM as LocalMarket[*]

    Note over MPS,LM: Initialise
    MPS->>MH: init_simulation(simulation, root_config_dir)
    MH->>MH: check config files exist, parse them, and check ports/species/price categories are known
    opt config files missing, malformed, or ports/species/price categories unknown
        MH-->>MPS: MarketError(NOT_FOUND or INVALID_ARGUMENT)
    end
    MH->>MS: setup_topology(markets_df)
    opt markets.csv missing required columns, or a local port missing lon/lat
        MS-->>MH: MarketError(FAILED_PRECONDITION)
    end
    MS->>MS: build prefix_to_locals and distance_lookup (distances floored at MIN_DISTANCE_M)
    MH->>MS: fit_all(prices_df, latest_date)
    MS->>MS: build volumes_by_port_by_key once
    loop each 2-letter prefix
        MS->>CM: fit(monthly_by_key, latest_date)
        CM->>CM: window selection, LevelVolumeModel fit with TrendCorrection
    end
    loop each 5-letter port
        MS->>LM: fit_base_models(monthly_by_key, latest_date)
        LM->>LM: window selection, LevelVolumeModel or LevelOnlyModel fit
    end
    MS->>MS: _fit_neighbor_models_for_all_ports() (builds monthly_by_key_by_port and volumes_by_port_by_key once)
    loop each port with a nearest-K eligible neighbor
        MS->>MS: anchor series (CM seasonal baseline), distance-weighted neighbor signal
        MS->>LM: fit_neighbor_models(species_code, category_code, inputs: NeighborFitInputs)
        LM->>LM: StackedModel fit on the base model's selected window
    end

    Note over MPS,LM: Per tick
    MPS->>MH: update_sales(request)
    MH->>MS: update_sales(market_sales, start_date_time)
    MS->>MS: reject unknown market codes, non-finite quantities, and out-of-range or past timestamps
    alt start_date_time is in the future
        MS->>MS: stash the sale in pending_sales_summary for that date
    else start_date_time matches the current tick
        alt market_code is a known local port
            MS->>LM: add_monthly_catch(species_code, category_code, quantity)
        else market_code falls back to its 2-letter prefix
            MS->>CM: add_direct_catch(species_code, category_code, quantity)
        end
    end

    MPS->>MH: simulate_step(current_date_time)
    opt current_date_time does not match the simulation's own tick date
        MH-->>MPS: MarketError(NOT_FOUND)
    end
    MH->>MS: step()
    MS->>MS: apply any pending_sales_summary stashed for this tick's date
    MS->>LM: take_monthly_catch(), atomic snapshot and clear
    MS->>CM: take_direct_catch(), atomic snapshot and clear
    alt prices computed
        MS->>MS: compute every LocalMarket's base-model prediction fresh this tick (base_predictions_by_key), needed as neighbor fuel for sibling ports
        MS->>CM: get_price(...): recorded price if one exists, else LevelVolumeModel/LevelOnlyModel x TrendCorrection
        MS->>LM: recorded price if one exists, else predict_from_base_model(...), else predict_from_neighbor_model(...)
        MS->>LM: update_volume_baseline(...) for the base model and the StackedModel
        MS->>CM: update_volume_baseline(...)
        MS->>MS: record_history(), tick.increase_tick()
    else an exception is raised
        MS->>LM: add_monthly_catch(...) merges the taken volumes back
        MS->>CM: add_direct_catch(...) merges the taken volumes back
        MS-->>MH: re-raise, the tick can be retried
    end
    MH->>MH: log the completed tick

    MPS->>MH: get_species_prices()
    MH->>MS: get_species_prices_message()

    Note over MPS,LM: Finalise
    MPS->>MH: finalise_simulation(folder_path)
    MH->>MS: export_results(simulation_id, folder_path)
```

---

## 3. Supply-Driven Seasonal Price Model

### 3.1 CombinedMarket (national level)

For each species at each national market, the model sets the price of a
simulated month in three steps:

1. **Seasonal price.** It starts from the typical price for that calendar
   month, learned from the historical prices.
2. **Supply adjustment.** It compares the month's landings/catches with
   the typical amount for that calendar month. Landings above the typical
   amount move the price one way, landings below it move the price the
   other way, and `beta` sets how strongly.
3. **Trend.** It carries the long-run price trend forward in time. The
   more statistically certain the trend, the larger its effect.

The model works with the logarithm of the price, so each adjustment acts
as a percentage change of the price.

#### Seasonal price and supply adjustment

Steps 1 and 2 as a formula:

```
log(price) = baseline_log_price[month] + alpha + beta * vol_anom
vol_anom   = (volume - baseline_volume[month]) / baseline_volume[month]
```

where, for one species at one national market:

| Variable | Meaning |
|---|---|
| `price` | The first-sale price the model reports for the month being simulated. |
| `month` | The calendar month (1–12) of the simulated date. It selects the seasonal values of baseline_log_price and baseline_volume. |
| `baseline_log_price[month]` | The typical log-price for the given calendar month, estimated from the historical prices in the selected training window. It carries the seasonal pattern. |
| `alpha` | A fitted constant that shifts the whole seasonal baseline up or down to the price level of the training window. |
| `beta` | The fitted volume-response coefficient: how strongly log-price moves for a given relative change in landings/catches. |
| `volume` | The landings/catches of the species reported for the simulated month by the other SURIMI models. |
| `baseline_volume[month]` | The typical landings/catches for the given calendar month, seeded from historical data and updated as the simulation runs. |
| `vol_anom` | The relative volume anomaly, i.e. how far volume lies above or below baseline_volume[month], as a fraction of it. |

`vol_anom` is clipped to the minimum and maximum relative anomaly seen
during training, so the volume response never exceeds the most extreme
response observed. When `baseline_volume[month] <= 0`, `vol_anom` is
`NaN`. Both rules apply to `CombinedMarket`, to the `LocalMarket` base
model (Section 3.2) and to `StackedModel` (Section 3.3).

**Interpretation:**
- Landings/catches above the typical amount for the month push the price
  down; below it, the price rises.
- `beta` sets the strength of that response — a larger `beta` means the
  same volume swing moves the price further.
- The response never exceeds what was seen during training: an extreme
  volume swing holds at the most extreme response observed, rather than
  extrapolating further.
- A calendar month with no usable baseline volume has no volume response
  at all.

If the volume-anomaly series is degenerate, the fit falls back to a
level-only model (no volume term).

#### Trend

Step 3, as a formula: it scales the step 1+2 price up or down by the
measured long-run trend, compounded over the time elapsed since training
ended — capped at the training window's own length, and muted in
proportion to how statistically confident that trend is:

```
elapsed_years = (date - train_end) / days_per_year
trend_years   = clip(elapsed_years, min_trend_years, years_trained)
price_hat     = exp(log_price_above) * exp(confidence * rate * trend_years)
```

where:

| Variable | Meaning |
|---|---|
| `date` | The simulated date being priced. |
| `train_end` | The last date of the training window used to fit the model. |
| `years_trained` | The length of the training window in years. It caps how far ahead the trend is extended. |
| `trend_years` | The years elapsed between train_end and date, clipped to the range 0 to years_trained. |
| `log_price_above` | The log-price from the seasonal level-volume formula, before any trend correction. |
| `confidence` | A score from 0 to 1 for how statistically certain the long-run trend is. |
| `rate` | The long-run trend in log-price per year. |
| `price_hat` | The trend-corrected first-sale price the model reports. |
| `days_per_year` | The average length of a year in days, including leap years (365.25). It converts the day difference between date and train_end into years. |
| `min_trend_years` | The lower bound of trend_years (0). For dates on or before train_end the trend has no effect, so it is only applied forward in time. |
| `clip(x, lower, upper)` | A function with three arguments that limits x to the range lower to upper. In the formula, x is elapsed_years, lower is min_trend_years and upper is years_trained. |
| `elapsed_years` | The years between train_end and date, before any limits are applied. |

`confidence` comes from per-calendar-month OLS significance
(`trend_confidence.compute_trend_confidence`), computed on data before
`train_end` only. Each month's `|t|` is tested against a critical value
scaled by its degrees of freedom
(`consts.trend_confidence_critical_value`), so a month backed by few years
of data needs a higher `|t|` to count as significant. `rate` comes from
sequential 3-year block-average log-prices, anchored backward from
`train_end` (`level_volume_model.fit_trend_rate_model`). For a species
without a multi-year trend, `confidence` is close to 0, so the correction
has no effect without any special case.

**Interpretation:**
- A species with a well-established, statistically confident long-run
  trend sees that trend continue into the forecast; a species with a
  weak or inconsistent trend doesn't.
- The trend's effect grows with time since training ended, but only up
  to the length of the training window itself — it is never extended
  further into the future than the data that measured it.
- For any date at or before the training window's end, the trend has no
  effect at all: it only ever pushes the price forward in time, never
  backward.

#### Price resolution

```mermaid
flowchart LR
    A[Price requested for a species, market and date] --> B{Real price recorded for this date?}
    B -->|Yes| C[Return the recorded price]
    B -->|No| D["Compute the seasonal + volume-response price<br/>(LevelVolumeModel, or LevelOnlyModel if no volume term could be fit)"]
    D --> E[Apply the trend correction]
    E --> F[Return price_hat]
```

### 3.2 LocalMarket base model (port level)

The same level-volume model as Section 3.1 (same `level_volume_model.py`
functions, including the `NaN` guard for non-positive baseline volume),
fitted independently per 5-letter port. Ports have sparser data than the
national aggregate, so the level-only fallback is more frequent here
(Section 4).

The port level has no trend correction: its models use
`TrendCorrection.identity`. The anchor that ports read from
`CombinedMarket.get_baseline_price_table()` is the untrended seasonal
baseline, so port prices carry no national trend either. This is a
deliberate design choice. Per-port series have fewer years per calendar
month, which makes a genuine multi-year trend hard to separate from
noise. A test on the 5 port/species pairs with the longest history (10
cutoffs between 2010 and 2023, forecasts up to 11 years ahead) confirmed
it: for 4 pairs the measured confidence was close to 0, so the correction
had no effect, and for FRSET/SAA (confidence 0.22–0.42) it worsened
long-horizon MAPE by 6.2 points on average and by up to 50 points at
single cutoffs.

### 3.3 LocalMarket neighbor-informed model

Built on top of the base model (Section 3.2) for ports with a usable
neighbor among the **TOP_K_NEAREST_DONORS** (5) geographically closest
local ports, weighted by inverse distance
(`compute_inverse_distance_weights`). Pairwise port distances
(`utility.build_distance_lookup`) are floored at `consts.MIN_DISTANCE_M`
(1 m), so two ports geocoded to the same coordinates still receive a
valid weight. `MarketSystem` computes two inputs and passes them to
`LocalMarket.fit_neighbor_models` as one `NeighborFitInputs` object:

- **Anchor series**: `CombinedMarket`'s frozen seasonal baseline for this
  species, via `MarketSystem.fit_all`'s `anchor_resolver` injection point
  (production never overrides it, so this static baseline is the only
  anchor source ever used live). Analysis code (the fidelity backtest)
  can inject an alternate resolver anchoring on `CombinedMarket`'s own
  live tick-by-tick prediction instead, to compare both; that resolver
  lives in `test/fidelity_backtest_harness.py`, not in production.
- **Neighbor signal**: a distance-weighted blend of *other* ports' own
  base-model predicted series, using `utility.build_distance_lookup`.

One model is fitted, `StackedModel`. It has the same formula shape as the
base model (Section 3.2): log-space, an own-volume anomaly relative to
the port's baseline volume and clipped to the training range, plus a
`gamma * neighbor_anom` term:

```
log(price) - log(anchor) = delta + beta * vol_anom + gamma * neighbor_anom
vol_anom      = clip((volume - port_baseline_volume[month]) / port_baseline_volume[month], vol_anom_min, vol_anom_max)
neighbor_anom = neighbor_value - neighbor_baseline[month]
```

where, for one species at one port:

| Variable | Meaning |
|---|---|
| `price` | The first-sale price the model reports for the port and month being simulated. |
| `anchor` | The national seasonal baseline for this species, read from `CombinedMarket` (the Anchor series, above). |
| `delta` | A fitted constant shift between the port's price and the anchor. |
| `beta` | The fitted volume-response coefficient, the same role as in the base model. |
| `volume` | The port's own landings/catches for the simulated month. |
| `port_baseline_volume[month]` | The port's typical landings/catches for the given calendar month. |
| `vol_anom` | The port's own relative volume anomaly, clipped to the range seen during training. |
| `gamma` | The fitted neighbor-response coefficient: how strongly the price follows the neighbor signal. |
| `neighbor_value` | The distance-weighted blend of neighboring ports' own base-model predictions for this month (the Neighbor signal, above). |
| `neighbor_baseline[month]` | The typical value of `neighbor_value` for the given calendar month. |
| `neighbor_anom` | How far `neighbor_value` lies from `neighbor_baseline[month]`. |

**Interpretation:**
- A port's price tracks the national seasonal baseline, adjusted up or
  down by its own supply and by what's happening at its nearest
  neighboring ports.
- The port's own volume response works exactly like the base model's:
  above-typical landings/catches push the price down, below-typical
  landings/catches push it up, clipped to what was seen during training.
- When neighboring ports are predicted to have higher-than-usual prices,
  `gamma` pulls this port's price up too, and vice versa.
- Because everything is computed in log-space, the reported price can
  never be zero or negative.

`StackedModel` is fitted once, on the training window already
selected for the port's base model (Section 3.5).
`StackedModel.port_baseline_volume` follows the port's baseline volume as
the simulation runs: every per-tick `update_volume_baseline` call updates
the base model and the `StackedModel` together through
`market_trackers.update_tracker_and_model`, so both models compute the
same volume anomaly.

#### Price resolution

```mermaid
flowchart LR
    A[Price requested for a species, port and date] --> B{Real price recorded for this date?}
    B -->|Yes| C[Return the recorded price]
    B -->|No| D["Compute the base-model prediction<br/>(always — needed as neighbor fuel for sibling ports too)"]
    D --> E{Eligible neighbor-informed model,<br/>and a neighbor value available this tick?}
    E -->|Yes| F[Compute the neighbor-informed StackedModel prediction]
    F --> G[Return the neighbor-informed prediction]
    E -->|No| H[Return the base-model prediction]
```

### 3.4 Real-price-first

Both `CombinedMarket.get_price()` and `LocalMarket.get_real_price()`
check their own `real_prices` lookup (a `RealPriceLookup` instance,
loaded from `prices.csv`'s blended rows) first, and return the real
recorded price directly whenever one exists — prediction is only ever
reached for a date with no real price.

### 3.5 Window selection (shared methodology)

`CombinedMarket.fit` and `LocalMarket.fit_base_models` select their
training-window length the same way: score each candidate window
(`WINDOW_CANDIDATE_YEARS = {2yr, 3yr, 4yr, 6yr, Full}`) via rolling-origin
validation (`window_selection.py`, model-agnostic — it only knows about
`fit_fn`/`score_fn` closures the caller supplies), then re-fit on the
single best-scoring window against the complete real history.

`LocalMarket.fit_neighbor_models` reuses the window that
`fit_base_models` selected for the same key and fits the `StackedModel`
once on it. If the base model is level-only, or the fit on that window
fails, the port has no neighbor-informed model for that key.

---

## 4. Validation

The model is validated on first-sale data for 5 species in Spain and
France (category Fresh - Whole, January 2010 to December 2024), with a
rolling backtest over 5 splits (`as_of_date` = 1 January 2018 to 2022,
2-year horizon).

| Measure | Result |
|---|---|
| National fits using the level-volume model | 10 of 10 (species, market) pairs |
| Port fits using the level-only fallback | 19 of 174 (about 11%) |
| Port/key pairs with a neighbor-informed model (split 2022-01-01) | 156 of 172 (90.7%) |
| Backtest MAPE, 5 splits, 800 comparable pairs | mean 42.32%, median 29.40% |
| Full fit in `init_simulation` with the real config data | 30.65 s |

`test_fidelity_backtest.py` checks the 2022-01-01 split against a 50%
MAPE ceiling. ANE at ESGLE, a volatile pair, reaches 50.32% at that split
and is on the test's documented exemption list. A 240-tick (20-year)
simulation through the gRPC interface with the full `northwestern_med`
config data runs without errors.

---

## 5. Known Limitations

- **Spurious national trends.** A cyclical price history can produce a
  confidently measured trend that is not real. SAA in France is the
  known case (`confidence` 0.25).
- **Flat constants file.** All project constants live in `consts.py`,
  grouped only by comment headers (currency and units, distance, donor
  eligibility, window candidates, rolling-origin parameters, anchor
  schemes, trend confidence and rate).
