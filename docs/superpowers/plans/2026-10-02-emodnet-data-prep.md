# EMODnet Data Preparation Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/models/market/data_prep/`, a script that downloads the current EMODnet "First Sales" dataset, transforms it into a complete `prices.csv` (local + combined-market rows, blended + per-size rows, in the existing shared schema plus a new `size` column) and a new `size_ratios.csv` (market-model-specific), ready for the market model's `init_simulation` to consume.

**Architecture:** A pipeline of small, pure-pandas transformation functions (easily unit-testable with synthetic data) wrapped by a thin I/O layer (HTTP download, geodatabase reading) that's tested separately via mocking, orchestrated by a single CLI entrypoint.

**Tech Stack:** Python, pandas, geopandas, pyogrio (Esri geodatabase reading), requests (HTTP download), pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-market-model-handler-design.md`, section 13 ("Data preparation: EMODnet → `prices.csv`") — this plan implements that section only; the core model classes are a separate plan.

## Global Constraints

- Nothing outside the `SURIMI-market` repo is touched — no access to the external `SURIMI-data_preprocessing` repo.
- `markets.csv` and `species.csv` are read-only inputs (shared with the landings-producing model) — this script never generates or modifies them.
- `category_code` in the output is constructed as `Preservation + " - " + Presentation` (e.g. `"Fresh - Whole"`), matching the value already present in `config/northwestern_med/prices.csv` exactly.
- Combined-market (`ES`/`FR`) rows are aggregated by summing `Value_EUR`/`Volume_Kg` across exactly the local ports `markets.csv` defines under that prefix — never by averaging already-computed per-port prices, and never sourced from a broader/separate EMODnet query.
- `size` is `NaN` for blended rows (not a string like `"ALL"`) — consistent with how species with no recorded size data already appear in the raw EMODnet data.
- The consistency check against `emodnet_es_raw.csv`/`emodnet_fr_raw_fresh_whole.csv` is a **one-time manual verification**, not a standing automated test (confirmed explicitly by the user) — it is the final task in this plan, not a `pytest` file.

---

## File Structure

```
src/models/market/__init__.py              (new, empty — package root, shared with the core-model plan)
src/models/market/data_prep/__init__.py    (new, empty)
src/models/market/data_prep/download.py   (EMODnet snapshot discovery + ZIP download/extract)
src/models/market/data_prep/extract.py    (raw geodatabase read + Fresh/Whole + market filter + category_code)
src/models/market/data_prep/aggregate.py  (local blended + per-size row aggregation)
src/models/market/data_prep/combine.py    (combined-market ES/FR row aggregation)
src/models/market/data_prep/size_ratios.py (per-(market, species, size) ratio computation)
src/models/market/data_prep/output.py     (final schema mapping + CSV writers)
src/models/market/data_prep/run.py        (CLI orchestration entrypoint)
src/models/market/data_prep/test/__init__.py        (new, empty)
src/models/market/data_prep/test/test_download.py
src/models/market/data_prep/test/test_extract.py
src/models/market/data_prep/test/test_aggregate.py
src/models/market/data_prep/test/test_combine.py
src/models/market/data_prep/test/test_size_ratios.py
src/models/market/data_prep/test/test_output.py
src/models/market/data_prep/test/test_run.py
requirements.txt                           (modified — add geopandas, pyogrio, requests)
```

Each module has one responsibility: `download` never parses data, `extract` never aggregates, `aggregate`/`combine` never touch I/O, `output` never computes anything. This mirrors how the existing notebook's steps are already conceptually separated, just split into testable units.

---

### Task 1: Package scaffold and dependencies

**Files:**
- Create: `src/models/market/__init__.py`
- Create: `src/models/market/data_prep/__init__.py`
- Create: `src/models/market/data_prep/test/__init__.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: importable `data_prep` package under `src/models/market/`.

- [ ] **Step 1: Create the package files**

Create `src/models/market/__init__.py` (empty file).
Create `src/models/market/data_prep/__init__.py` (empty file).
Create `src/models/market/data_prep/test/__init__.py` (empty file).

- [ ] **Step 2: Add new dependencies to `requirements.txt`**

Add these lines (alphabetical position doesn't matter, existing file has no strict ordering):

```
geopandas
pyogrio
requests
```

- [ ] **Step 3: Install and verify**

Run: `pip install -r requirements.txt`
Then run: `python -c "import geopandas, pyogrio, requests; print('ok')"`
Expected: prints `ok` with no import errors.

- [ ] **Step 4: Commit**

```bash
git add src/models/market/__init__.py src/models/market/data_prep/__init__.py src/models/market/data_prep/test/__init__.py requirements.txt
git commit -m "chore: scaffold data_prep package and add geo dependencies"
```

---

### Task 2: EMODnet snapshot discovery and ZIP download

**Files:**
- Create: `src/models/market/data_prep/download.py`
- Test: `src/models/market/data_prep/test/test_download.py`

**Interfaces:**
- Produces: `get_current_zip_filename(metadata_xml: str) -> str`, `get_current_zip_url(session=None) -> str`, `download_and_extract(zip_url: str, dest_dir, session=None) -> Path` (returns path to the extracted `.gdb` directory). Consumed by Task 8 (`run.py`).

- [ ] **Step 1: Write the failing tests**

Create `src/models/market/data_prep/test/test_download.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock
import zipfile

import pytest

from data_prep.download import get_current_zip_filename, get_current_zip_url, download_and_extract


def test_get_current_zip_filename_extracts_dated_filename():
    xml = (
        "<record><citationIdentifier>EMODnet_HA_Fisheries_FirstSales_20250319"
        "</citationIdentifier><link>.../EMODnet_HA_Fisheries_FirstSales_20250319.zip"
        "</link></record>"
    )
    assert get_current_zip_filename(xml) == "EMODnet_HA_Fisheries_FirstSales_20250319.zip"


def test_get_current_zip_filename_raises_when_not_found():
    with pytest.raises(ValueError):
        get_current_zip_filename("<record>no filename here</record>")


def test_get_current_zip_url_builds_full_url():
    mock_session = MagicMock()
    mock_session.get.return_value.text = (
        "<record>EMODnet_HA_Fisheries_FirstSales_20250920.zip</record>"
    )
    mock_session.get.return_value.raise_for_status.return_value = None

    url = get_current_zip_url(session=mock_session)

    assert url == (
        "https://ows.emodnet-humanactivities.eu/geonetwork/srv/api/records/"
        "ff5f738f-94ee-4704-94c2-7c951c931973/attachments/"
        "EMODnet_HA_Fisheries_FirstSales_20250920.zip"
    )


def test_download_and_extract_finds_single_gdb(tmp_path):
    source_dir = tmp_path / "source"
    gdb_dir = source_dir / "Fake_20250920.gdb"
    gdb_dir.mkdir(parents=True)
    (gdb_dir / "placeholder.txt").write_text("fake gdb contents")

    zip_path = tmp_path / "fake.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in gdb_dir.rglob("*"):
            zf.write(file, file.relative_to(source_dir))

    mock_session = MagicMock()
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()
    mock_session.get.return_value.iter_content.return_value = [zip_bytes]
    mock_session.get.return_value.raise_for_status.return_value = None

    dest_dir = tmp_path / "dest"
    result = download_and_extract("http://fake-url/fake.zip", dest_dir, session=mock_session)

    assert result == dest_dir / "Fake_20250920.gdb"
    assert (result / "placeholder.txt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/models/market/data_prep/test/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_prep.download'` (the module doesn't exist yet).

- [ ] **Step 3: Write the implementation**

Create `src/models/market/data_prep/download.py`:

```python
"""Downloads the current EMODnet First Sales dataset snapshot."""
import re
import zipfile
from pathlib import Path

import requests

METADATA_RECORD_URL = (
    "https://emodnet.ec.europa.eu/geonetwork/srv/api/records/"
    "ff5f738f-94ee-4704-94c2-7c951c931973/formatters/xml"
)
ATTACHMENT_BASE_URL = (
    "https://ows.emodnet-humanactivities.eu/geonetwork/srv/api/records/"
    "ff5f738f-94ee-4704-94c2-7c951c931973/attachments/"
)


def get_current_zip_filename(metadata_xml: str) -> str:
    """Extract the current dated ZIP filename (e.g.
    "EMODnet_HA_Fisheries_FirstSales_20250319.zip") from the record's XML
    metadata -- the filename changes with each twice-yearly update, so it
    is never hardcoded."""
    match = re.search(r"EMODnet_HA_Fisheries_FirstSales_\d{8}\.zip", metadata_xml)
    if match is None:
        raise ValueError(
            "Could not find a dated EMODnet_HA_Fisheries_FirstSales_*.zip "
            "filename in the metadata record."
        )
    return match.group(0)


def get_current_zip_url(session=None) -> str:
    """Query the EMODnet metadata record and return the current ZIP download URL."""
    session = session or requests
    response = session.get(METADATA_RECORD_URL, timeout=30)
    response.raise_for_status()
    filename = get_current_zip_filename(response.text)
    return ATTACHMENT_BASE_URL + filename


def download_and_extract(zip_url: str, dest_dir, session=None) -> Path:
    """Download the EMODnet ZIP to dest_dir and extract it. Returns the
    path to the extracted .gdb directory."""
    session = session or requests
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "emodnet_first_sales.zip"
    response = session.get(zip_url, timeout=300, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    gdb_paths = list(dest_dir.glob("*.gdb"))
    if len(gdb_paths) != 1:
        raise ValueError(
            f"Expected exactly one .gdb directory after extraction, found {len(gdb_paths)}."
        )
    return gdb_paths[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/models/market/data_prep/test/test_download.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/market/data_prep/download.py src/models/market/data_prep/test/test_download.py
git commit -m "feat: add EMODnet snapshot discovery and download"
```

---

### Task 3: Raw extraction, Fresh/Whole + market filter, category_code construction

**Files:**
- Create: `src/models/market/data_prep/extract.py`
- Test: `src/models/market/data_prep/test/test_extract.py`

**Interfaces:**
- Consumes: path to `.gdb` (from Task 2's `download_and_extract`).
- Produces: `find_data_layer(gdb_path) -> str`, `load_fish_sales_data(gdb_path) -> pd.DataFrame`, `filter_and_clean(df: pd.DataFrame, market_codes: list[str]) -> pd.DataFrame` — columns `date, market_code, species_code, category_code, Value_EUR, Volume_Kg, Price_EUR_kg, Size`. Consumed by Task 4 (`aggregate.py`) and Task 8 (`run.py`).

- [ ] **Step 1: Write the failing test for `filter_and_clean`**

Create `src/models/market/data_prep/test/test_extract.py`:

```python
import pandas as pd

from data_prep.extract import filter_and_clean


def _sample_raw_rows():
    return pd.DataFrame([
        {"Date": pd.Timestamp("2020-01-01"), "Locode": "ESALC", "ERS_Code": "ANE",
         "Preservation": "Fresh", "Presentation": "Whole",
         "Value_EUR": 100.0, "Volume_Kg": 50.0, "Price_EUR_kg": 2.0, "Size": "1"},
        {"Date": pd.Timestamp("2020-01-01"), "Locode": "ESALC", "ERS_Code": "ANE",
         "Preservation": "Frozen", "Presentation": "Whole",
         "Value_EUR": 40.0, "Volume_Kg": 20.0, "Price_EUR_kg": 2.0, "Size": "1"},
        {"Date": pd.Timestamp("2020-01-01"), "Locode": "ESXXX", "ERS_Code": "ANE",
         "Preservation": "Fresh", "Presentation": "Whole",
         "Value_EUR": 10.0, "Volume_Kg": 5.0, "Price_EUR_kg": 2.0, "Size": "1"},
    ])


def test_filter_and_clean_restricts_to_fresh_whole_and_known_markets():
    raw = _sample_raw_rows()
    result = filter_and_clean(raw, market_codes=["ESALC"])

    assert len(result) == 1
    assert result.iloc[0]["market_code"] == "ESALC"
    assert result.iloc[0]["category_code"] == "Fresh - Whole"


def test_filter_and_clean_renames_columns():
    raw = _sample_raw_rows()
    result = filter_and_clean(raw, market_codes=["ESALC", "ESXXX"])
    expected_columns = {
        "date", "market_code", "species_code", "category_code",
        "Value_EUR", "Volume_Kg", "Price_EUR_kg", "Size",
    }
    assert expected_columns.issubset(set(result.columns))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/models/market/data_prep/test/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_prep.extract'`.

- [ ] **Step 3: Write the implementation**

Create `src/models/market/data_prep/extract.py`:

```python
"""Extracts and filters raw EMODnet First Sales data."""
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio


def find_data_layer(gdb_path) -> str:
    """Find the First Sales data layer name within the geodatabase. The
    layer name is dated (e.g. "EMODNET_HA_Fisheries_FirstSales_Data_20250319"),
    so this searches by pattern rather than hardcoding a date."""
    layers = pyogrio.list_layers(gdb_path)
    layer_names = [name for name, _ in layers]
    matches = [name for name in layer_names if "FirstSales_Data" in name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one First Sales data layer, found {len(matches)}: {matches}"
        )
    return matches[0]


def load_fish_sales_data(gdb_path) -> pd.DataFrame:
    """Load the First Sales data layer as a plain DataFrame, with a Date
    column derived from Year + Month_N."""
    layer_name = find_data_layer(gdb_path)
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    df["Date"] = pd.to_datetime(
        df[["Year", "Month_N"]].rename(columns={"Year": "year", "Month_N": "month"}).assign(day=1)
    )
    return df


def filter_and_clean(df: pd.DataFrame, market_codes: list) -> pd.DataFrame:
    """Restrict to the given market codes and Fresh+Whole sales only, and
    construct category_code from Preservation + Presentation."""
    filtered = df[
        (df["Locode"].isin(market_codes))
        & (df["Preservation"] == "Fresh")
        & (df["Presentation"] == "Whole")
    ].copy()

    filtered["category_code"] = filtered["Preservation"] + " - " + filtered["Presentation"]

    return filtered[
        [
            "Date", "Locode", "ERS_Code", "category_code",
            "Value_EUR", "Volume_Kg", "Price_EUR_kg", "Size",
        ]
    ].rename(columns={"Date": "date", "Locode": "market_code", "ERS_Code": "species_code"})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/models/market/data_prep/test/test_extract.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/market/data_prep/extract.py src/models/market/data_prep/test/test_extract.py
git commit -m "feat: add raw EMODnet extraction and Fresh/Whole filtering"
```

---

### Task 4: Local blended and per-size row aggregation

**Files:**
- Create: `src/models/market/data_prep/aggregate.py`
- Test: `src/models/market/data_prep/test/test_aggregate.py`

**Interfaces:**
- Consumes: output of `filter_and_clean` (Task 3) — columns `date, market_code, species_code, category_code, Value_EUR, Volume_Kg, Price_EUR_kg, Size`.
- Produces: `GROUP_KEYS = ["date", "market_code", "species_code", "category_code"]` (a module-level constant, reused by Task 5), `build_local_rows(cleaned: pd.DataFrame) -> pd.DataFrame` — columns `date, market_code, species_code, category_code, size, landings_kg, price`. Consumed by Task 5 (`combine.py`) and Task 8 (`run.py`).

- [ ] **Step 1: Write the failing test**

Create `src/models/market/data_prep/test/test_aggregate.py`:

```python
import pandas as pd
import pytest

from data_prep.aggregate import build_local_rows


def test_build_local_rows_produces_blended_and_per_size_rows():
    cleaned = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "Value_EUR": 100.0, "Volume_Kg": 50.0,
         "Price_EUR_kg": 2.0, "Size": "1"},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "Value_EUR": 60.0, "Volume_Kg": 20.0,
         "Price_EUR_kg": 3.0, "Size": "2"},
    ])

    result = build_local_rows(cleaned)

    blended = result[result["size"].isna()]
    assert len(blended) == 1
    assert blended.iloc[0]["landings_kg"] == 70.0
    assert blended.iloc[0]["price"] == pytest.approx(160.0 / 70.0)

    size_1 = result[result["size"] == "1"]
    assert size_1.iloc[0]["landings_kg"] == 50.0
    assert size_1.iloc[0]["price"] == 2.0

    size_2 = result[result["size"] == "2"]
    assert size_2.iloc[0]["landings_kg"] == 20.0
    assert size_2.iloc[0]["price"] == 3.0


def test_build_local_rows_handles_rows_with_no_recorded_size():
    cleaned = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "SAA",
         "category_code": "Fresh - Whole", "Value_EUR": 30.0, "Volume_Kg": 10.0,
         "Price_EUR_kg": 3.0, "Size": None},
    ])

    result = build_local_rows(cleaned)

    assert len(result) == 1
    assert result.iloc[0]["size"] is None or pd.isna(result.iloc[0]["size"])
    assert result.iloc[0]["landings_kg"] == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/models/market/data_prep/test/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_prep.aggregate'`.

- [ ] **Step 3: Write the implementation**

Create `src/models/market/data_prep/aggregate.py`:

```python
"""Aggregates cleaned per-transaction rows into blended and per-size
prices.csv rows."""
import numpy as np
import pandas as pd

GROUP_KEYS = ["date", "market_code", "species_code", "category_code"]


def _summarise(df: pd.DataFrame, group_keys: list) -> pd.DataFrame:
    grouped = df.groupby(group_keys, as_index=False).agg(
        landings_kg=("Volume_Kg", "sum"),
        value_eur=("Value_EUR", "sum"),
    )
    grouped["price"] = grouped["value_eur"] / grouped["landings_kg"]
    return grouped.drop(columns="value_eur")


def build_local_rows(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Build both the blended (size=NaN, summed across all size grades)
    and the per-size rows for each (date, market_code, species_code,
    category_code) group."""
    blended = _summarise(cleaned, GROUP_KEYS)
    blended["size"] = np.nan

    with_size = cleaned.dropna(subset=["Size"])
    per_size = _summarise(with_size, GROUP_KEYS + ["Size"])
    per_size = per_size.rename(columns={"Size": "size"})

    combined = pd.concat([blended, per_size], ignore_index=True)
    return combined[GROUP_KEYS + ["size", "landings_kg", "price"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/models/market/data_prep/test/test_aggregate.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/market/data_prep/aggregate.py src/models/market/data_prep/test/test_aggregate.py
git commit -m "feat: add local blended and per-size row aggregation"
```

---

### Task 5: Combined-market (ES/FR) row aggregation

**Files:**
- Create: `src/models/market/data_prep/combine.py`
- Test: `src/models/market/data_prep/test/test_combine.py`

**Interfaces:**
- Consumes: `build_local_rows` output (Task 4), and `GROUP_KEYS` from `aggregate.py`.
- Produces: `build_combined_rows(local_rows: pd.DataFrame, prefix_to_locals: dict) -> pd.DataFrame` — same columns as `build_local_rows`, with `market_code` set to the 2-letter prefix. Consumed by Task 6 (`size_ratios.py`) and Task 8 (`run.py`).

- [ ] **Step 1: Write the failing test**

Create `src/models/market/data_prep/test/test_combine.py`:

```python
import pandas as pd
import pytest

from data_prep.combine import build_combined_rows


def test_build_combined_rows_sums_local_ports_under_shared_prefix():
    local_rows = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 50.0, "price": 2.0},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESBCN", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 30.0, "price": 3.0},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "FRAGD", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 10.0, "price": 5.0},
    ])
    prefix_to_locals = {"ES": ["ESALC", "ESBCN"], "FR": ["FRAGD"]}

    result = build_combined_rows(local_rows, prefix_to_locals)

    es_row = result[result["market_code"] == "ES"].iloc[0]
    assert es_row["landings_kg"] == 80.0
    assert es_row["price"] == pytest.approx((50.0 * 2.0 + 30.0 * 3.0) / 80.0)

    fr_row = result[result["market_code"] == "FR"].iloc[0]
    assert fr_row["landings_kg"] == 10.0
    assert fr_row["price"] == 5.0


def test_build_combined_rows_preserves_size_grouping():
    local_rows = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": "1", "landings_kg": 20.0, "price": 2.5},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESBCN", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": "1", "landings_kg": 10.0, "price": 2.0},
    ])
    prefix_to_locals = {"ES": ["ESALC", "ESBCN"]}

    result = build_combined_rows(local_rows, prefix_to_locals)

    assert len(result) == 1
    assert result.iloc[0]["size"] == "1"
    assert result.iloc[0]["landings_kg"] == 30.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/models/market/data_prep/test/test_combine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_prep.combine'`.

- [ ] **Step 3: Write the implementation**

Create `src/models/market/data_prep/combine.py`:

```python
"""Aggregates local rows into combined (national) rows, matching the live
model's own runtime aggregation rule (spec section 4)."""
import pandas as pd

from .aggregate import GROUP_KEYS


def build_combined_rows(local_rows: pd.DataFrame, prefix_to_locals: dict) -> pd.DataFrame:
    """For each combined prefix (e.g. "ES", "FR"), sum landings_kg and the
    implied value (price * landings_kg) across exactly its local ports --
    for both the blended (size=NaN) and per-size rows -- deriving price
    from the summed totals, never averaging already-computed per-port
    prices directly."""
    rows = []
    for prefix, local_codes in prefix_to_locals.items():
        subset = local_rows[local_rows["market_code"].isin(local_codes)].copy()
        if subset.empty:
            continue
        subset["value_eur"] = subset["price"] * subset["landings_kg"]

        group_keys = ["date", "species_code", "category_code", "size"]
        grouped = subset.groupby(group_keys, as_index=False, dropna=False).agg(
            landings_kg=("landings_kg", "sum"),
            value_eur=("value_eur", "sum"),
        )
        grouped["price"] = grouped["value_eur"] / grouped["landings_kg"]
        grouped["market_code"] = prefix
        rows.append(grouped.drop(columns="value_eur"))

    if not rows:
        return pd.DataFrame(columns=GROUP_KEYS + ["size", "landings_kg", "price"])

    return pd.concat(rows, ignore_index=True)[GROUP_KEYS + ["size", "landings_kg", "price"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/models/market/data_prep/test/test_combine.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/market/data_prep/combine.py src/models/market/data_prep/test/test_combine.py
git commit -m "feat: add combined-market row aggregation"
```

---

### Task 6: `size_ratios.csv` computation

**Files:**
- Create: `src/models/market/data_prep/size_ratios.py`
- Test: `src/models/market/data_prep/test/test_size_ratios.py`

**Interfaces:**
- Consumes: `build_combined_rows` output (Task 5) — has both blended (`size=NaN`) and per-size rows for each combined market.
- Produces: `compute_size_ratios(combined_rows: pd.DataFrame) -> pd.DataFrame` — columns `market_code, species_code, size, ratio`. Consumed by Task 8 (`run.py`).

- [ ] **Step 1: Write the failing test**

Create `src/models/market/data_prep/test/test_size_ratios.py`:

```python
import pandas as pd
import pytest

from data_prep.size_ratios import compute_size_ratios


def test_compute_size_ratios_averages_ratio_over_time():
    combined_rows = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 100.0, "price": 2.0},
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": "1", "landings_kg": 40.0, "price": 2.2},
        {"date": pd.Timestamp("2020-02-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 90.0, "price": 2.5},
        {"date": pd.Timestamp("2020-02-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": "1", "landings_kg": 30.0, "price": 2.75},
    ])

    result = compute_size_ratios(combined_rows)

    row = result[
        (result["market_code"] == "ES")
        & (result["species_code"] == "ANE")
        & (result["size"] == "1")
    ]
    assert row.iloc[0]["ratio"] == pytest.approx((2.2 / 2.0 + 2.75 / 2.5) / 2)


def test_compute_size_ratios_excludes_months_missing_a_blended_row():
    combined_rows = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": "1", "landings_kg": 40.0, "price": 2.2},
    ])

    result = compute_size_ratios(combined_rows)

    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/models/market/data_prep/test/test_size_ratios.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_prep.size_ratios'`.

- [ ] **Step 3: Write the implementation**

Create `src/models/market/data_prep/size_ratios.py`:

```python
"""Computes the per-(market, species, size) historical price ratio used
to convert a blended prediction into a size-specific quote."""
import pandas as pd


def compute_size_ratios(combined_rows: pd.DataFrame) -> pd.DataFrame:
    blended = combined_rows[combined_rows["size"].isna()][
        ["date", "market_code", "species_code", "category_code", "price"]
    ].rename(columns={"price": "blended_price"})

    per_size = combined_rows[combined_rows["size"].notna()]

    merged = per_size.merge(
        blended, on=["date", "market_code", "species_code", "category_code"], how="inner"
    )
    merged["ratio"] = merged["price"] / merged["blended_price"]

    return (
        merged.groupby(["market_code", "species_code", "size"], as_index=False)["ratio"]
        .mean()
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/models/market/data_prep/test/test_size_ratios.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/market/data_prep/size_ratios.py src/models/market/data_prep/test/test_size_ratios.py
git commit -m "feat: add size_ratios.csv computation"
```

---

### Task 7: Final schema mapping and CSV writers

**Files:**
- Create: `src/models/market/data_prep/output.py`
- Test: `src/models/market/data_prep/test/test_output.py`

**Interfaces:**
- Consumes: local rows (Task 4), combined rows (Task 5), ratios (Task 6).
- Produces: `PRICES_CSV_COLUMNS` (module-level constant), `build_prices_csv_rows(*row_frames) -> pd.DataFrame`, `write_prices_csv(all_rows, path) -> None`, `write_size_ratios_csv(ratios, path) -> None`. Consumed by Task 8 (`run.py`).

- [ ] **Step 1: Write the failing tests**

Create `src/models/market/data_prep/test/test_output.py`:

```python
import pandas as pd

from data_prep.output import build_prices_csv_rows, write_prices_csv, write_size_ratios_csv


def test_build_prices_csv_rows_adds_constant_columns():
    local = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 50.0, "price": 2.0},
    ])
    combined = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "market_code": "ES", "species_code": "ANE",
         "category_code": "Fresh - Whole", "size": float("nan"), "landings_kg": 50.0, "price": 2.0},
    ])

    result = build_prices_csv_rows(local, combined)

    assert len(result) == 2
    assert (result["currency"] == "EUR").all()
    assert (result["measurement_unit"] == "kg").all()


def test_write_prices_csv_round_trips(tmp_path):
    rows = pd.DataFrame([
        {"date": "2020-01-01", "market_code": "ESALC", "species_code": "ANE",
         "category_code": "Fresh - Whole", "landings_kg": 50.0, "price": 2.0,
         "currency": "EUR", "measurement_unit": "kg", "size": float("nan")},
    ])
    path = tmp_path / "prices.csv"

    write_prices_csv(rows, path)
    reloaded = pd.read_csv(path)

    assert reloaded.iloc[0]["market_code"] == "ESALC"
    assert pd.isna(reloaded.iloc[0]["size"])


def test_write_size_ratios_csv_round_trips(tmp_path):
    ratios = pd.DataFrame([{"market_code": "ES", "species_code": "ANE", "size": "1", "ratio": 1.1}])
    path = tmp_path / "size_ratios.csv"

    write_size_ratios_csv(ratios, path)
    reloaded = pd.read_csv(path)

    assert reloaded.iloc[0]["ratio"] == 1.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/models/market/data_prep/test/test_output.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_prep.output'`.

- [ ] **Step 3: Write the implementation**

Create `src/models/market/data_prep/output.py`:

```python
"""Writes the final prices.csv (shared schema) and size_ratios.csv
(market-model-specific) output files."""
import pandas as pd

PRICES_CSV_COLUMNS = [
    "date", "market_code", "species_code", "category_code",
    "landings_kg", "price", "currency", "measurement_unit", "size",
]


def build_prices_csv_rows(*row_frames) -> pd.DataFrame:
    """Concatenate local + combined (blended + per-size) rows and add the
    constant currency/measurement_unit columns expected by the shared
    schema."""
    all_rows = pd.concat(row_frames, ignore_index=True)
    all_rows["currency"] = "EUR"
    all_rows["measurement_unit"] = "kg"
    return all_rows[PRICES_CSV_COLUMNS]


def write_prices_csv(all_rows: pd.DataFrame, path) -> None:
    all_rows.sort_values(["date", "market_code", "species_code", "category_code"]).to_csv(
        path, index=False
    )


def write_size_ratios_csv(ratios: pd.DataFrame, path) -> None:
    ratios.sort_values(["market_code", "species_code", "size"]).to_csv(path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/models/market/data_prep/test/test_output.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/market/data_prep/output.py src/models/market/data_prep/test/test_output.py
git commit -m "feat: add prices.csv/size_ratios.csv schema mapping and writers"
```

---

### Task 8: Top-level orchestration CLI

**Files:**
- Create: `src/models/market/data_prep/run.py`
- Test: `src/models/market/data_prep/test/test_run.py`

**Interfaces:**
- Consumes: all prior tasks' functions.
- Produces: `build_prefix_to_locals(markets_csv_path) -> dict`, `run(markets_csv_path, output_dir, gdb_path=None, dest_dir=None) -> None`, `main()` (CLI entrypoint). `gdb_path` lets a caller skip the download step and point at an already-extracted geodatabase — used both for testing and for the pinned-snapshot consistency check in Task 9.

- [ ] **Step 1: Write the failing tests**

Create `src/models/market/data_prep/test/test_run.py`:

```python
from unittest.mock import patch

import pandas as pd

from data_prep.run import build_prefix_to_locals, run


def test_build_prefix_to_locals_groups_by_prefix(tmp_path):
    markets_csv = tmp_path / "markets.csv"
    pd.DataFrame({
        "market_code": ["ES", "FR", "ESALC", "ESBCN", "FRAGD"],
        "market_name": ["Spain", "France", "Alicante", "Barcelona", "Agde"],
        "lon": [None, None, -0.5, 2.2, None],
        "lat": [None, None, 38.3, 41.4, None],
    }).to_csv(markets_csv, index=False)

    result = build_prefix_to_locals(markets_csv)

    assert set(result["ES"]) == {"ESALC", "ESBCN"}
    assert set(result["FR"]) == {"FRAGD"}


def test_build_prefix_to_locals_infers_prefix_with_no_explicit_combined_row(tmp_path):
    markets_csv = tmp_path / "markets.csv"
    pd.DataFrame({
        "market_code": ["ESALC"],
        "market_name": ["Alicante"],
        "lon": [-0.5],
        "lat": [38.3],
    }).to_csv(markets_csv, index=False)

    result = build_prefix_to_locals(markets_csv)

    assert set(result["ES"]) == {"ESALC"}


def test_run_builds_prices_and_size_ratios_csv(tmp_path):
    markets_csv = tmp_path / "markets.csv"
    pd.DataFrame({
        "market_code": ["ES", "ESALC"],
        "market_name": ["Spain", "Alicante"],
        "lon": [None, -0.5],
        "lat": [None, 38.3],
    }).to_csv(markets_csv, index=False)

    fake_raw = pd.DataFrame([
        {"Date": pd.Timestamp("2020-01-01"), "Locode": "ESALC", "ERS_Code": "ANE",
         "Preservation": "Fresh", "Presentation": "Whole",
         "Value_EUR": 100.0, "Volume_Kg": 50.0, "Price_EUR_kg": 2.0, "Size": "1"},
    ])

    output_dir = tmp_path / "out"
    with patch("data_prep.run.load_fish_sales_data", return_value=fake_raw):
        run(markets_csv, output_dir, gdb_path=tmp_path / "fake.gdb")

    prices = pd.read_csv(output_dir / "prices.csv")
    assert len(prices) > 0
    assert "ES" in prices["market_code"].values
    assert "ESALC" in prices["market_code"].values

    ratios_path = output_dir / "size_ratios.csv"
    assert ratios_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/models/market/data_prep/test/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_prep.run'`.

- [ ] **Step 3: Write the implementation**

Create `src/models/market/data_prep/run.py`:

```python
"""Orchestrates the full EMODnet -> prices.csv / size_ratios.csv pipeline."""
import argparse
from pathlib import Path

import pandas as pd

from .aggregate import build_local_rows
from .combine import build_combined_rows
from .download import download_and_extract, get_current_zip_url
from .extract import filter_and_clean, load_fish_sales_data
from .output import build_prices_csv_rows, write_prices_csv, write_size_ratios_csv
from .size_ratios import compute_size_ratios


def build_prefix_to_locals(markets_csv_path) -> dict:
    """Group markets.csv's 5-letter LOCODEs by their 2-letter prefix. The
    set of needed prefixes is the union of any 2-letter codes explicitly
    listed and any prefixes implied by the 5-letter codes present, so a
    combined market always exists for any country with local ports under
    it, even if its own 2-letter row is missing."""
    markets = pd.read_csv(markets_csv_path)
    codes = markets["market_code"].dropna().unique().tolist()
    local_codes = [c for c in codes if len(c) == 5]
    explicit_prefixes = {c for c in codes if len(c) == 2}
    implied_prefixes = {c[:2] for c in local_codes}
    prefixes = explicit_prefixes | implied_prefixes

    return {
        prefix: [c for c in local_codes if c.startswith(prefix)]
        for prefix in prefixes
    }


def run(markets_csv_path, output_dir, gdb_path=None, dest_dir=None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if gdb_path is None:
        zip_url = get_current_zip_url()
        gdb_path = download_and_extract(zip_url, dest_dir or output_dir / "_download")

    prefix_to_locals = build_prefix_to_locals(markets_csv_path)
    all_local_codes = [code for codes in prefix_to_locals.values() for code in codes]

    raw = load_fish_sales_data(gdb_path)
    cleaned = filter_and_clean(raw, market_codes=all_local_codes)

    local_rows = build_local_rows(cleaned)
    combined_rows = build_combined_rows(local_rows, prefix_to_locals)

    prices_rows = build_prices_csv_rows(local_rows, combined_rows)
    write_prices_csv(prices_rows, output_dir / "prices.csv")

    ratios = compute_size_ratios(combined_rows)
    write_size_ratios_csv(ratios, output_dir / "size_ratios.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Build prices.csv and size_ratios.csv from EMODnet."
    )
    parser.add_argument("--markets-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--gdb-path", default=None,
        help="Skip download, use an already-extracted .gdb (e.g. for the "
             "pinned-snapshot consistency check).",
    )
    args = parser.parse_args()
    run(args.markets_csv, args.output_dir, gdb_path=args.gdb_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/models/market/data_prep/test/test_run.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Run the full test suite for this plan**

Run: `pytest src/models/market/data_prep/ -v`
Expected: all tests across all 7 test files PASS.

- [ ] **Step 6: Commit**

```bash
git add src/models/market/data_prep/run.py src/models/market/data_prep/test/test_run.py
git commit -m "feat: add data_prep CLI orchestration entrypoint"
```

---

### Task 9: One-time manual consistency verification (not an automated test)

**Files:** none created — this is a verification procedure run once against the existing repo data, not new source code.

This step is **required before trusting this script's output**, per spec section 13: the fidelity backtest for the core model only means something once this check has passed. This is a manual, one-time check, not a `pytest` file — the script will, in normal operation, always fetch the *current* EMODnet snapshot, so there's no ongoing "latest reference file" to keep comparing against after this one confirmation.

- [ ] **Step 1: Confirm the pinned snapshot is available locally**

The existing notebook (`scripts/fisheries_data/emodnet_prices_landings_raw_new.ipynb`) already downloaded the pinned snapshot used to produce the validated data. Confirm it's present:

```bash
ls ".cache/EMODnet_HA_Fisheries_FirstSales_20250319/"
```

Expected: the `.gdb` directory and `.shp` files are present. If not present, this check cannot proceed without the original download — stop and flag this rather than substituting a different snapshot.

- [ ] **Step 2: Run the pipeline against the pinned snapshot**

```bash
python -m src.models.market.data_prep.run \
  --markets-csv config/northwestern_med/markets.csv \
  --output-dir /tmp/consistency_check \
  --gdb-path ".cache/EMODnet_HA_Fisheries_FirstSales_20250319/EMODnet_HA_Fisheries_FirstSales_20250319.gdb"
```

Expected: completes without error, producing `/tmp/consistency_check/prices.csv` and `/tmp/consistency_check/size_ratios.csv`.

- [ ] **Step 3: Compare against the already-validated reference data**

Run this comparison script (save as a throwaway local file, e.g. `/tmp/compare.py`, not committed to the repo):

```python
import pandas as pd

new_prices = pd.read_csv("/tmp/consistency_check/prices.csv")
new_local = new_prices[new_prices["size"].isna() & (new_prices["market_code"].str.len() == 5)]

old_es = pd.read_csv("scripts/fisheries_data/emodnet_es_raw.csv", low_memory=False)
old_fr = pd.read_csv("scripts/fisheries_data/emodnet_fr_raw_fresh_whole.csv", low_memory=False)
old = pd.concat([old_es, old_fr], ignore_index=True)

for species in ["ANE", "PIL", "HKE", "SAA", "MUT"]:
    old_species = old[old["ERS_Code"] == species]
    old_monthly = old_species.groupby("Date")["Volume_Kg"].sum()

    new_species = new_local[new_local["species_code"] == species]
    new_monthly = new_species.groupby("date")["landings_kg"].sum()

    comparison = pd.DataFrame({"old": old_monthly, "new": new_monthly}).dropna()
    max_diff = (comparison["old"] - comparison["new"]).abs().max()
    print(f"{species}: {len(comparison)} months compared, max abs diff = {max_diff:.4f} kg")
```

Run: `python /tmp/compare.py`

Expected: `max abs diff` is at or near 0 (floating-point tolerance, e.g. < 0.01 kg) for every species listed.

- [ ] **Step 4: Decide based on the result**

If every species' max diff is near zero: the script is confirmed consistent with the validated data. Proceed to use it as the model's data source.

If any species shows a meaningfully large diff: **stop** — do not proceed to use this script's output for the core model's fidelity backtest until the discrepancy is investigated and understood. A mismatch here means either this script's filtering/aggregation logic differs from the existing notebook's in some way, or the two files were never actually from the same snapshot — either way, the fidelity backtest would not be measuring what it claims to measure until this is resolved.

---

## Self-Review

**Spec coverage:** section 13's location, source/download mechanism, refresh cadence, read-only `markets.csv`/`species.csv` inputs, local-row reuse, `category_code` construction, combined-row aggregation rule, size handling (blended `NaN` + per-size rows), `size_ratios.csv` as a separate optional output, and the one-time consistency check are all covered by tasks 1–9 above. The `size_ratios.csv` optionality (the model must work without it) and the markets.csv optionality of the explicit `ES`/`FR` row are both reflected in the implementation (`build_prefix_to_locals`'s union logic; `size_ratios.csv` is just a separate file the model may or may not load — that consumption logic belongs to the other plan, not this one).

**Placeholder scan:** no "TBD"/"TODO"/"add appropriate handling" phrasing anywhere in the tasks above; every step has concrete, runnable code or an exact command.

**Type consistency:** `build_local_rows` and `build_combined_rows` both return frames with the same column set (`date, market_code, species_code, category_code, size, landings_kg, price`), which `build_prices_csv_rows` (Task 7) correctly consumes via `*row_frames`. `GROUP_KEYS` is defined once in `aggregate.py` and imported, not redefined, in `combine.py`.
