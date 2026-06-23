# SURIMI Market Model — Technical Documentation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Simulation Lifecycle](#3-simulation-lifecycle)
4. [Market Price Model](#4-market-price-model)
   - 4.1 [Global Market](#41-global-market)
   - 4.2 [Local Market](#42-local-market)
   - 4.3 [Price Update Formula](#43-price-update-formula)
   - 4.4 [Baseline Price Initialization](#44-baseline-price-initialization)
5. [gRPC API](#5-grpc-api)
   - 5.1 [WorkflowService](#51-workflowservice)
   - 5.2 [MarketProviderService](#52-marketproviderservice)
6. [Data Management](#6-data-management)
   - 6.1 [Input Configuration Files](#61-input-configuration-files)
   - 6.2 [Output Files](#62-output-files)
   - 6.3 [AWS S3 Integration](#63-aws-s3-integration)
   - 6.4 [Nightly Config Sync](#64-nightly-config-sync)
7. [Configuration & Environment Variables](#7-configuration--environment-variables)
8. [Deployment](#8-deployment)
9. [Infrastructure & Observability](#9-infrastructure--observability)
10. [Known Limitations](#10-known-limitations)

---

## 1. Overview

The **SURIMI Market Model** is a gRPC microservice that simulates fish market price dynamics as part of the broader **SURIMI** research project — a European initiative (2024–2027) developing socio-ecological models to support marine management strategy evaluation under the EU's **European Digital Twin of the Ocean (EU DTO)**.

The service answers the question: *how do fish prices respond to changes in catch volume?* It models two levels of price formation:

- **Local markets** (individual ports, identified by UN/LOCODE) where prices vary with supply according to species-specific elasticity.
- A **global market** that computes a supply-weighted average across all active local markets.

The service integrates into a larger simulation framework alongside other models (e.g., EwE, Poseidon, CMSY). An external workflow controller drives the simulation tick by tick, pushing catch data in and reading prices back out via gRPC.

Geographic scope: **Western Mediterranean Sea**, ~30 ports, ~150 species, historical price data from 2013 onwards.

---

## 2. Architecture

### Component Overview

TODO ask Rik about whether the Controller side is represented correctly.
TODO Explaining the model in plain text
- What "config/CSV files" are we reading on init?

![Component diagram of the Market Server/Model](./doc/plantuml_image_renders/DOCUMENTATION_component_diagram.svg)

### Domain Model

<!-- TODO use an automated tool to create the class diagrams. -->

This diagram reflects the class structure in `src/market_model.py`.

```mermaid
classDiagram
    class MarketSystem {
        -currency: str
        -species_dict: Dict~str, Species~
        -local_markets: Dict~str, LocalMarket~
        -global_market: GlobalMarket
        -tick: Tick
        +add_species(s: Species)
        +step(catches: list)
        +export_results(simulation_id, folder_path)
        +get_species_prices_message() GetSpeciesPricesResponse
    }

    class Species {
        -name: str
        -code: str
        -epsilon: float
    }

    class LocalMarket {
        -market_code: str
        -market_name: str
        -lon: float
        -lat: float
        -lambda_mix: float
        -theta_mix: float
        -monthly_catch: Dict~str, float~
        -local_prices: Dict~str, float~
        -price_state: Dict~str, tuple~
        -price_history: list
        -price_time_series: Dict~tuple, float~
        +record_monthly_catch(species_code, quantity)
        +get_last_state(s_code) tuple
        +set_new_state(s_code, price, qty)
        +calculate_local_price(species_code, other_market_avg)
    }

    class GlobalMarket {
        -species_dict: Dict~str, Species~
        -global_prices: Dict~str, float~
        -global_price_history: list
        +get_global_price(species_code) float
    }

    class Tick {
        -tick: int
        -start_time: datetime
        -step_days: int
        +increase_tick()
        +get_tick() int
        +get_timestamp() str
    }

    MarketSystem "1" o-- "many" Species
    MarketSystem "1" o-- "many" LocalMarket
    MarketSystem "1" o-- "1" GlobalMarket
    MarketSystem "1" o-- "1" Tick
    LocalMarket "many" --> "1" GlobalMarket
    GlobalMarket --> Species : references
    LocalMarket --> Species : references (shared)
```

---

## 3. Simulation Lifecycle

A simulation proceeds in three phases: **Initialise**, one or more **tick iterations**, and **Finalise**. The external controller orchestrates this sequence.

<!-- Ask third-part person for feedback on this diagram. -->

```mermaid
sequenceDiagram
    participant CTL as Controller
    participant APP as App
    participant WF as VaultService
    participant MPS as MarketService
    participant SH as SimulationHandler
    participant MS as MarketSystem
    participant LM as LocalMarket[*]
    Note over LM: for each local market
    participant GM as GlobalMarket
    participant S3 as AWS S3

    %% Start the gRPC server
    CTL->>APP: start_market_model_app()
    APP->>WF: load_secrets_from_vault()

    Note over CTL,S3: Initialise Simulation
    CTL->>APP: initialise_simulation(simulation_id, contract)
    APP->>MPS: initialise_simulation(simulation_id, contract)
    
    %% Note: We already have the config CSVs in the repository. This only
    %%       updates them.
    alt Enable downloading from S3
        MPS->>S3: download_files_from_S3("surimi-market/config")
        S3-->>MPS: return_config_csv_files()
        MPS->>MPS: update_local_config(csv_files)
    end

    MPS->>SH: init_simulation(simulation_id)

    SH->>SH: copy config/ to simulations/{id}/config/
    SH->>SH: load_config_csvs()
    
    SH->>MS: set_initial_local_price()
    MS->>LM: set_initial_price()
    MS->>LM: update_local_history()

    SH->>MS: set_initial_global_prices()
    MS->>GM: set_initial_prices(local_markets)
    MS->>GM: update_history()

    MS->>LM: set_fallback_price(global_market_prices)
    
    SH->>MS: increase_tick()

    SH-->>MPS: return simulation_pb2.InitialiseSimulationResponse(simulation_id)
    MPS-->>APP: return simulation_pb2.InitialiseSimulationResponse(simulation_id)
    APP-->>CTL: return simulation_pb2.InitialiseSimulationResponse(simulation_id)

    Note over CTL,S3: Per Tick (repeated N times)
    loop for each simulation month
        %% UpdateSales
        loop for each update sales request
            Note over APP,MPS: Each iteration buffers the catches 
            CTL->>APP: update_sales(simulation_id, sales_summaries)
            APP->>MPS: update_sales(simulation_id, sales_summaries)
            MPS->>SH: update_sales(request)
            SH->>MS: update_sales(market_sales)

            MS->>LM: add_monthly_catch(species_code, gear_code, quantity)
            MS->>SH: return successor_or_fail
            SH-->>MPS: return (update_sales_pb2.UpdateSalesResponse, MarketError)
            MPS-->>APP: update_sales_pb2.UpdateSalesResponse
            APP-->>CTL: update_sales_pb2.UpdateSalesResponse
        end
        %% Step
        CTL->>APP: simulate_step(simulation_id, current_date_time)
        APP->>MPS: simulate_step(simulation_id, current_date_time)
        MPS->>SH: simulate_step(simulation_id, current_date_time)
        SH->>MS: step()
        loop for each local market
            MS->>MS: compute_local_prices()
        end
        MS->>LM: set_local_price(species_codes, gear_codes, computed_local_prices)
        MS->>LM: update_local_history()
        MS->>MS: increase_tick()
        SH-->>WF: simulate_step_pb2.SimulateStepResponse(simulation_id)
        WF-->>APP: simulate_step_pb2.SimulateStepResponse(simulation_id)
        APP-->>CTL: simulate_step_pb2.SimulateStepResponse(simulation_id)

        %% GetSpeciesPrices
        CTL->>APP: get_species_prices(simulation_id, date_time)
        APP->>MPS: get_species_prices(simulation_id, date_time)
        MPS->>SH: get_market_prices(datetime)
        SH->>MS: get_local_market_prices()
        MS->>LM: get_prices(species_codes, gear_codes)
        LM-->>MS: return price or None
        MS-->>SH: return get_species_prices_pb2.GetSpeciesPricesResponse(price_summary)
        SH-->>MPS: return get_species_prices_pb2.GetSpeciesPricesResponse(price_summary)
        MPS-->>APP: return get_species_prices_pb2.GetSpeciesPricesResponse(price_summary)
        APP-->>CTL: return get_species_prices_pb2.GetSpeciesPricesResponse(price_summary)
    end

    Note over CTL,S3: Finalise
    CTL->>APP: finalise_simulation(simulation_id)
    APP->>MPS: finalise_simulation(simulation_id)
    MPS->>SH: finalise_simulation()
    SH->>MS: export_results(simulation_id, "simulations")
    MS->>MS: write_global_prices(filepath)
    MS->>MS: write_local_prices(filepath)
    MPS-->>APP: return finalise_simulation_pb2.FinaliseSimulationResponse(simulation_id)
    APP-->>CTL: return finalise_simulation_pb2.FinaliseSimulationResponse(simulation_id)
```

---

## 4. Market Price Model

### 4.1 Global Market

At the moment global market (`GlobalMarket`) does not apply any price-adjustment model. It computes a **simple arithmetic mean** of valid local prices for each species across all active local markets at the end of every tick.

```
GlobalPrice(s) = mean({ LocalPrice(m, s) | LocalPrice(m, s) is valid })
```

If no local market has a valid price for a species, `GlobalPrice(s)` is `None` for that tick.

The global price serves two roles:
1. **Fallback during baseline initialization** — if a local market has no CSV price for a species, it inherits the global average.
2. **Optional mixing signal** — if `lambda_mix > 0` on a local market, the global price pulls the local price toward the global average each tick (see §4.2).

### 4.2 Local Market

Each `LocalMarket` tracks its own price per species. After every catch update, prices are driven by **supply changes** via the Elser adjustment (§4.3). Two optional mixing parameters can then blend the result with external signals:

| Parameter | Symbol | Effect |
|---|---|---|
| `lambda_mix` | λ | Blends local price toward the global average |
| `theta_mix` | θ | Blends local price toward a neighbor-market average |

Mixing is applied **after** the Elser update:

```
P = (1 - λ) * P_local + λ * P_global      # global pull
P = (1 - θ) * P       + θ * P_neighbors   # neighbor pull
```

Both parameters default to `0.0` in the current implementation (no mixing by default).

### 4.3 Price Update Formula

Local prices are updated each tick using the **Elser elasticity adjustment**, derived from Elser et al. (2018) — [doi:10.1016/j.ecolecon.2018.09.008](https://doi.org/10.1016/j.ecolecon.2018.09.008).

The formula applied in `MarketSystem._update_local_prices()` (`src/market_model.py:219`):

```
P_new = P_last * (Q_last / Q_new) ^ ε
```

| Symbol | Meaning |
|---|---|
| `P_new` | Updated price this tick (EUR/kg) |
| `P_last` | Price from the previous tick |
| `Q_last` | Catch quantity from the previous tick (kg) |
| `Q_new` | Catch quantity this tick (kg); floored at `1e-6` to avoid division by zero |
| `ε` | Price elasticity coefficient for the species (default `0.5`) |

**Interpretation:**
- If catch increases (`Q_new > Q_last`), price falls — more supply, lower price.
- If catch decreases (`Q_new < Q_last`), price rises — less supply, higher price.
- `ε` controls sensitivity. At `ε = 1` the response is proportional; at `ε = 0.5` it is dampened (square-root scaling).
- Only species with a recorded catch this tick are updated. Species with no catch keep their previous price unchanged.

### 4.4 Baseline Price Initialization

Before tick 0 the model initializes prices in three phases (`MarketSystem._initialize_baseline_prices_from_csv`):

```
Phase 1 — CSV:        LocalPrice(m, s) = earliest price in prices.csv for (m, s)
Phase 2 — Global avg: GlobalPrice(s)   = mean of all valid LocalPrice(*, s) from Phase 1
Phase 3 — Fallback:   If LocalPrice(m, s) still None → set to GlobalPrice(s)
                       If GlobalPrice(s) also None → leave as None (logged as warning)
```

The global price computed in Phase 2 is not recorded in `global_price_history`; history starts from the first `step()` call.

---

## 5. gRPC API

The protocol is defined in the external [SURIMI Protocol repository](https://github.com/Official-EwE/SURIMI-protocol) and consumed as a versioned Python package from `buf.build`.

### 5.1 WorkflowService

Manages simulation lifecycle. Implemented in `server/workflow_service.py`.

| RPC | Request | Response | Description |
|---|---|---|---|
| `Initialise` | `InitialiseRequest` | `InitialiseResponse` | Creates a new simulation instance. Downloads config from S3, loads CSV data, initializes baseline prices. Fails if `simulation_id` already exists. |
| `SimulateStep` | `SimulateStepRequest` | `SimulateStepResponse` | Executes one monthly tick: flushes buffered catches into the market engine, updates prices, advances the tick counter. |
| `Finalise` | `FinaliseRequest` | `FinaliseResponse` | Exports `global_prices.csv` and `local_prices.csv`, then uploads the simulation directory to S3. |
| `Cancel` | `CancelRequest` | `CancelResponse` | Removes the simulation from the active dictionary without writing results. |
| `GetProtocolVersion` | `GetProtocolVersionRequest` | `GetProtocolVersionResponse` | Returns the git commit hash of the installed protobuf SDK package (derived from the package version string). |

### 5.2 MarketProviderService

Handles per-tick data exchange. Implemented in `server/market_provider_service.py`.

| RPC | Request | Response | Description |
|---|---|---|---|
| `GetSpeciesPrices` | `GetSpeciesPricesRequest` | `GetSpeciesPricesResponse` | Returns current prices for all (species, market) combinations that have a valid price. Skips combinations where price is `None`. |
| `UpdateSales` | `UpdateSalesRequest` | `UpdateSalesResponse` | Receives catch data (sales summaries with species, market, gear, and quantity). Aggregates quantities into `pending_catch_dict`; they are consumed on the next `SimulateStep` call. May be called multiple times per tick. |

Both services share a `simulation_dictionary: dict[str, SimulationHandler]` injected at startup. All RPCs return `INVALID_ARGUMENT` if the `simulation_id` is unknown.

---

## 6. Data Management

### 6.1 Input Configuration Files

All three files live in `config/` and are synced nightly from `SURIMI-western_med` (see §6.4).

| File | Key columns | Description |
|---|---|---|
| `species.csv` | `species_code`, `species_name` | ~150 Western Mediterranean species. `epsilon` defaults to `0.5` for all (not in CSV). |
| `ports.csv` | `port_code`, `port_name`, `lon`, `lat` | ~30 ports, identified by UN/LOCODE (e.g. `ESALG`). |
| `prices.csv` | `date`, `market_code`, `species_code`, `category_code`, `price`, `currency`, `measurement_unit` | Historical price time series, 2013 to present. ~2.3 MB. |

On `Initialise`, the global `config/` directory is copied to `simulations/<simulation_id>/config/` to snapshot the inputs at the time of the run.

### 6.2 Output Files

Both files are written under `simulations/<simulation_id>/` on `Finalise`.

**`global_prices.csv`**

| Column | Type | Description |
|---|---|---|
| `tick` | int | Simulation tick number (0-based) |
| `species` | str | Species code |
| `avg_price` | float | Mean price across all local markets (blank if None) |
| `total_catch` | float | Sum of catches across all local markets (kg) |

**`local_prices.csv`**

| Column | Type | Description |
|---|---|---|
| `tick` | int | Simulation tick number |
| `market` | str | Port UN/LOCODE |
| `species` | str | Species code |
| `price` | float | Local price (blank if None) |
| `catch` | float | Catch recorded at this market this tick (kg) |

### 6.3 AWS S3 Integration

Implemented in `server/s3_storage.py`. AWS credentials are injected via environment variables loaded from Vault (see §9).

| Event | Direction | S3 path |
|---|---|---|
| `Initialise` | Download | `surimi-market/config/` → `config/` |
| `Finalise` | Upload | `simulations/<id>/` → `surimi-market/simulations/<id>/` |

`.bin` files are skipped on upload. `.keep` files and S3 directory entries are skipped on download. If `AWS_BUCKET_NAME` or `AWS_ACCESS_KEY_ID` are not set, the operation is silently skipped.

### 6.4 Nightly Config Sync

Defined in `.github/workflows/sync-market-inputs.yml`. Runs at **02:00 UTC** every day (also triggerable manually).

```mermaid
sequenceDiagram
    participant GHA as GitHub Actions
    participant Src as SURIMI-western_med
    participant Dst as SURIMI-market

    GHA->>Src: sparse checkout SURIMI-market_inputs/ (with LFS, using SURIMI_SYNC_CSVS_TOKEN)
    GHA->>GHA: diff ports.csv, prices.csv, species.csv
    alt any file changed
        GHA->>Dst: copy files to config/, git add + commit + push
    else no changes
        GHA->>GHA: echo "Nothing to commit"
    end
```

The Personal Access Token (`SURIMI_SYNC_CSVS_TOKEN`) expires **2027-02-12**.

---

## 7. Configuration & Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| **gRPC** | | | |
| `GRPC_HOST` | No | `[::]` | Bind address for the gRPC server |
| `GRPC_PORT` | No | `5001` | Listening port |
| **HashiCorp Vault** | | | |
| `VAULT_ADDR` | Yes (prod) | — | Vault server URL |
| `VAULT_TOKEN` | Yes (prod) | — | Authentication token |
| `VAULT_TOP_DIR` | Yes (prod) | — | Top-level path in Vault KV store |
| `VAULT_RELATIVE_PATH` | Yes (prod) | — | Relative secret path under `VAULT_TOP_DIR` |
| `VAULT_MOUNT` | Yes (prod) | — | KV mount point |
| **AWS S3** | | | |
| `AWS_BUCKET_NAME` | Yes (prod) | — | S3 bucket name |
| `AWS_ACCESS_KEY_ID` | Yes (prod) | — | AWS access key (loaded from Vault) |
| `AWS_SECRET_ACCESS_KEY` | Yes (prod) | — | AWS secret key (loaded from Vault) |
| `AWS_SESSION_TOKEN` | No | — | Temporary session token (if using STS) |
| `AWS_DEFAULT_REGION` | Yes (prod) | — | AWS region |
| `AWS_S3_ENDPOINT` | Yes (prod) | — | S3 endpoint hostname (without `https://`) |
| **OpenTelemetry** | | | |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | `http://localhost:4317` | OTLP collector endpoint |
| `OTEL_SERVICE_NAME` | No | `market` | Service name reported to the collector |

Vault and AWS variables can be provided via a `.env` file (not committed to the repository). If all Vault variables are absent, Vault loading is silently skipped; if AWS variables are absent, S3 operations are silently skipped.

---

## 8. Deployment

### Docker

The service ships as a single Docker image built from `python:3.12-slim`.

```dockerfile
# Build
docker build -t surimi-market .

# Run (minimal, no Vault/S3)
docker run -p 5001:5001 surimi-market

# Run with credentials
docker run -p 5001:5001 \
  -e VAULT_ADDR=... \
  -e VAULT_TOKEN=... \
  ... \
  surimi-market
```

The container exposes port `5001`. The entrypoint is `python -u server/app.py` (`-u` ensures stdout/stderr are unbuffered for immediate log visibility).

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Create .env with Vault credentials (optional — service runs without S3/Vault in dev)
python server/app.py
```

### Dependency Note

The gRPC Python SDK (`surimi-surimi-protocol-grpc-python`) is pinned to a specific build from `buf.build`. Newer versions of `grpcio` can introduce OpenTelemetry / protobuf conflicts in Docker builds — update dependencies cautiously and test inside Docker before releasing.

---

## 9. Infrastructure & Observability

### HashiCorp Vault (`server/vault_service.py`)

On startup, `VaultService.LoadVaultSecretsInEnvironmentVariables()` authenticates to Vault using `VAULT_TOKEN` and reads all key-value pairs from the configured secret path into the process environment. This is the primary mechanism for injecting AWS credentials in production without storing them in the image or the repository.

### OpenTelemetry

The service exports traces and metrics to an OTLP collector (default: `localhost:4317`). This is configured via the `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_SERVICE_NAME` environment variables, which are set in the Dockerfile and can be overridden at runtime.

### Logging (`logger.py`)

All modules use a shared logger factory (`get_logger(name)`). Each logger writes to two sinks simultaneously:

| Sink | Location | Level |
|---|---|---|
| Console (stdout) | — | `DEBUG` |
| Rotating file | `logs/surimi.log` | `DEBUG` |

Log format: `YYYY-MM-DD HH:MM:SS LEVEL    logger_name - message`

The `logs/` directory is created automatically on first run and is excluded from git.

---

## 10. Known Limitations

The following issues are tracked in `BUGS.md`.

| # | Severity | Status | Description |
|---|---|---|---|
| 1 | High | **Fixed** | Duplicate tick 0 in `global_prices.csv`. `_initialize_baseline_prices_from_csv` called `_compute_global_prices` which wrote a history entry before the first tick. Fixed by adding `save_history=False` to the init call. |
| 2 | High | Open | **Orphaned simulation on failed init.** The simulation is added to `simulation_dictionary` before S3 download and CSV loading. If either fails, the entry remains in a broken state, blocking re-initialization with the same ID. |
| 3 | Medium | Open | **Race condition on `pending_catch_dict`.** `UpdateSales` and `SimulateStep` run in separate gRPC threads with no lock guarding shared state. Concurrent calls can corrupt or silently drop catch data. |
| 4 | Medium | Open | **Per-simulation config copy never loaded.** `init_simulation` copies `config/` to `simulations/<id>/config/` but `load_initialization_data` always reads from the top-level `config/`. The per-simulation copy is unused. |
| 5 | Medium | Open | **Wrong tick index in `calculate_local_price`.** History entries use `len(self.price_history)` instead of the actual simulation tick, and omit the `"catches"` key present in `_update_local_history` snapshots. |
| 6 | Low | Open | **CSV parsing crashes on malformed data.** `float(row["price"])`, `float(row["lon"])`, and `float(row["lat"])` have no error handling; blank or non-numeric cells raise an uncaught exception and abort initialization. |
