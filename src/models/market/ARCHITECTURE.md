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
which a historic price was recorded, it reports that price as-is; for
months without a historic price, it reports the model's prediction.

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
        MS->>CM: get_price(...): historic price if one exists, else LevelVolumeModel/LevelOnlyModel x TrendCorrection
        MS->>LM: historic price if one exists, else predict_from_base_model(...), else predict_from_neighbor_model(...)
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

### 3.1 Combined Market

```mermaid
flowchart LR
    A[Price requested for a species, market and date] --> B{Historic price recorded for this date?}
    B -->|Yes| C[Return the historic price]
    B -->|No| D["Compute the seasonal + volume-response price<br/>"]
    D --> E[Apply the trend correction]
    E --> F[Return price]
```

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
`NaN`. Both rules apply to `CombinedMarket` here, and to the `LocalMarket`
base model and to `StackedModel` in Section 3.2.

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

If the volume-anomaly series has too few usable training rows, or
doesn't vary at all, the fit falls back to a level-only model (no
volume term).

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

`confidence` measures how statistically certain the long-run trend is,
using only data before `train_end`; a month backed by fewer years of
data needs stronger evidence to count as confident. `rate` is the
measured long-run trend in log-price per year, computed over the years
before `train_end`. For a species without a real multi-year trend,
`confidence` is close to 0, so the correction has no effect.

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

### 3.2 Local Market

```mermaid
flowchart LR
    A[Price requested for a species, port and date] --> B{Historic price recorded for this date?}
    B -->|Yes| C[Return the historic price]
    B -->|No| D["Compute the base-model prediction<br/>"]
    D --> E{Eligible neighbor-informed model,<br/>and a neighbor value available this tick?}
    E -->|Yes| F[Compute the neighbor-informed StackedModel prediction]
    F --> G[Return the neighbor-informed price]
    E -->|No| H[Return the base-model price]
```

Each port's price is built in two layers:

1. **Base model.** Fitted independently on the port's own history, using
   the same formula as Section 3.1.
2. **Neighbor-informed model.** For ports with a usable geographic
   neighbor, blends in nearby ports' own predictions on top of the base
   model.

#### Base model (port level)

The same seasonal-plus-volume-response model as Section 3.1, fitted
independently per 5-letter port, without the trend term: ports anchor on
`CombinedMarket`'s untrended seasonal baseline instead of fitting their
own trend. Where the volume-anomaly series has too few usable training
rows, or doesn't vary at all, the fit falls back to the level-only model
(no volume term), the same as at the national level.

#### Neighbor-informed model

A port with a usable neighbor gets a second model layered on top of the
base model. Its neighbors are its 5 closest local ports by geographic
distance, with closer ports counting for more. Two inputs feed this fit:

- **Anchor series**: `CombinedMarket`'s own seasonal baseline for this
  species.
- **Neighbor signal**: a distance-weighted blend of the other ports' own
  predicted prices.

`StackedModel` fits one combined formula: the port's own volume
response, same as the base model, plus a term for the neighboring
ports' predicted prices:

```
log(price) - log(anchor) = delta + beta * vol_anom + gamma * neighbor_anom
vol_anom      = clip((volume - port_baseline_volume[month]) / port_baseline_volume[month], vol_anom_min, vol_anom_max)
neighbor_anom = neighbor_value - neighbor_baseline[month]
```

where, for one species at one local port:

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

`port_baseline_volume[month]` is kept in sync with the base model's own
baseline: every per-tick update applies to both, so the two models
compute the same volume anomaly.

**Interpretation:**
- A port's price tracks the national seasonal baseline, adjusted up or
  down by its own supply and by its nearest neighboring ports' own
  predicted prices.
- The port's own volume response works exactly like the base model's:
  above-typical landings/catches push the price down, below-typical
  landings/catches push it up, clipped to what was seen during training.
- When neighboring ports are predicted to have higher-than-usual prices,
  `gamma` pulls this port's price up too, and vice versa.
- Because everything is computed in log-space, the reported price can
  never be zero or negative.

### 3.3 Shared Mechanisms

#### Historic-price-first

Both `CombinedMarket` and `LocalMarket` check their own historic-price
lookup (a `RealPriceLookup` instance, loaded from `prices.csv`'s blended
rows) first, and return that price directly whenever one exists —
prediction is only ever reached for a date with no historic price.

#### Window selection

`CombinedMarket` and `LocalMarket`'s base models select their
training-window length the same way: score each candidate window
(`WINDOW_CANDIDATE_YEARS = {2yr, 3yr, 4yr, 6yr, Full}`) via model-agnostic
rolling-origin validation, then re-fit on the single best-scoring window
against the complete real history.

The neighbor-informed model reuses the window the base model already
selected for the same key, and fits `StackedModel` once on it. If the
base model is level-only, or the fit on that window fails, the port has
no neighbor-informed model for that key.

---

## 4. Input and Output Files

### 4.1 Input Configuration Files

| File | Key columns | Description |
|---|---|---|
| `species.csv` | `species_code` | The contracted species. |
| `markets.csv` | `market_code`, `market_name`, `lon`, `lat` | Every port and national market; 5-letter codes are local ports, 2-letter codes are national markets. |
| `prices.csv` | `date`, `market_code`, `species_code`, `category_code`, `price`, `landings_kg`, `size` | Historical first-sale prices and landings/catches. Rows with no `size` value are the blended real-price rows the model fits and reports against. |

An optional `size_ratios.csv` is also loaded if present, but not yet used
by any model logic.

### 4.2 Output Files

Both files are written under `simulations/<simulation_id>/` when the
simulation finishes.

**`combined_prices.csv`**

| Column | Description |
|---|---|
| `tick` | Simulation tick number. |
| `market_code` | The 2-letter national market code. |
| `species` | Species code. |
| `category_code` | Price category code. |
| `price` | The reported price (blank if none). |
| `volume` | Total landings/catches for this market, species and category this tick. |

**`local_prices.csv`**

| Column | Description |
|---|---|
| `tick` | Simulation tick number. |
| `market_code` | The 5-letter local port code. |
| `species` | Species code. |
| `category_code` | Price category code. |
| `price` | The reported price (blank if none). |
| `catch` | Landings/catches at this port, species and category this tick. |

---

## 5. Known Limitations

- **Spurious national trends.** A cyclical price history can produce a
  confidently measured trend that is not real. SAA in France is the
  known case (`confidence` 0.25).
- **Flat constants file.** All project constants live in `consts.py`,
  grouped only by comment headers (currency and units, distance, donor
  eligibility, window candidates, rolling-origin parameters, anchor
  schemes, trend confidence and rate).
