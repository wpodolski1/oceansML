# oceansML

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/status-active-success.svg)  
![Data](https://img.shields.io/badge/datasets-NOAA%20%7C%20UNEP--WCMC-blueviolet)  

**oceansML** is a Python project for exploring, visualizing, and modelling ocean datasets. It provides tools for preprocessing, model evaluation, and visualization of oceanographic data, focusing on coral reefs and coral bleaching events.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Motivation & Problem Statement](#motivation--problem-statement)  
3. [Data Sources](#data-sources)  
4. [Installation](#installation)  
5. [Code Structure](#code-structure)  
6. [Usage](#usage)  
7. [Results & Evaluation](#results--evaluation)  
8. [Roadmap](#roadmap)  
9. [Future Work](#future-work)  
10. [Contributing](#contributing)  
11. [Acknowledgments](#acknowledgments)  
12. [License](#license)

---

## Project Overview

oceansML is designed to support the analysis of coral reef datasets by providing:

- Loading and preprocessing sea surface temperature (SST) and coral reef location data  
- Visualizing coral bleaching hotspots following NOAA guidelines  
- Building and evaluating machine learning models (e.g., LSTM) for forecasting reef stress  

The long-term goal is to estimate yearly reef damage and predict bleaching risk under warming trends.

---

## Motivation & Problem Statement

Coral reefs are among the most fragile ecosystems on Earth, yet they are poorly described in global datasets. Monitoring bleaching events is challenging due to:

- Lack of high-quality datasets  
- Limited temporal and spatial coverage  
- Quality issues in existing observations  

This project addresses these issues by:

- Combining SST and reef polygon data to identify bleaching hotspots  
- Estimating the percentage of reefs at risk under NOAA bleaching thresholds  
- Building predictive models to forecast reef bleaching given ongoing climate trends  

---

## Data Sources

### Sea Surface Temperature (SST)  
- **NOAA PSL Climate Data Repository**  
  Used for both climatology baseline and model training.  

### Coral Reef Locations  
- **UNEP-WCMC (2025)** — Ocean+ Habitats [Online]. *Coral Reefs (2018 dataset)*  
  Available at: [habitats.oceanplus.org](https://habitats.oceanplus.org)  

### Datasets & Preprocessing

**Climatology baseline (1985–1990 & 1993):**
- `sst.day.mean.1985.nc`  
- `sst.day.mean.1986.nc`  
- `sst.day.mean.1987.nc`  
- `sst.day.mean.1988.nc`  
- `sst.day.mean.1989.nc`  
- `sst.day.mean.1990.nc`  
- `sst.day.mean.1993.nc`  

Steps:  
1. Calculate monthly means.  
2. Extract maximum monthly mean (MMM) across baseline years.  

**Forecast model inputs:**

- Consecutive annual `reef_sst_YEAR.nc` partitions produced by Stage 1.
- Stable reef-cell IDs and coordinates across every partition.
- Explicit chronological training, validation, and test cutoffs.

The forecasting path keeps each temporal sequence within one reef cell. It
preserves daily ordering, excludes windows containing missing SST, fits seasonal
climatology and normalization from training dates only, and predicts every daily
lead directly rather than recursively.

**Visualization of bleaching impact:**  
- `sst.day.mean.2022.nc`  
- `sst.day.mean.2023.nc`  

---

## Installation

```bash
# Clone the repo
git clone https://github.com/Aspanu1/oceansML.git
cd oceansML

# Install the locked Python 3.11.18 environment and development tools
uv sync --locked --group dev
```

Rasterio wheels bundle GDAL but still need Expat, zlib, and the C++ runtime.
On Ubuntu (including CI), install these distribution packages:

```bash
sudo apt-get update
sudo apt-get install --yes --no-install-recommends libexpat1 zlib1g libstdc++6
```

On NixOS, enter the repository's pinned development shell **before** running uv:

```bash
nix-shell
uv sync --locked --group dev
```

`shell.nix` pins nixpkgs by revision and hash, supplies Python 3.11.18 and uv,
and derives the native library search path with `lib.makeLibraryPath`.
No machine-specific store paths or global loader changes are needed. The shell
uses Nix Python rather than downloading an interpreter that assumes an FHS loader.
A pre-existing uv environment may be recreated to use that interpreter.

Notebook CSS is optional: `css_formating()` skips styling when IPython is absent.
The core scripts and CI do not require IPython.


## Development and pull requests

This is currently a script-based project rather than a published Python package.
Install the locked development environment and run the same checks as pull-request
CI:

```bash
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest
```

BasedPyright retains `recommended` mode, including failure on warnings. Published
GeoPandas/Shapely development stubs and precise scientific-library boundary types
avoid suppressions. The small `typings/rasterio` interface describes only the
Rasterio 1.4 APIs used here; published Rasterio stubs require Python 3.12 and target
Rasterio 1.5. Update that interface when changing those calls or runtime versions.

The workflow pins actions and uv, reads Python's patch version from
`.python-version`, and sets `UV_LOCKED=true` for every step. CI tests synthetic
scientific contracts, not real-data forecasting skill. See the
[CI repair review](REVIEW.md#ci-repair-review--2026-11-18) for failure classification,
verification results, and remaining debt.

Use focused branches and pull requests. Routine documentation and engineering
changes follow the lightweight Mode A process; preprocessing, dataset splitting,
forecasting, DHW, scaling, evaluation, and checkpoint changes follow the
scientific Mode B process with explicit leakage and reproducibility checks.
Neither mode requires a GitHub issue for trivial work. See
[the agentic Git and CI/CD plan](plan.md#agentic-git-and-cicd-workflow) for the
review policy, scientific evaluation gate, and static artifact publication flow.

---
## Stage 1: observed reef-cell SST preparation

This pipeline prepares observations, not forecasts, bleaching probabilities, or
mortality estimates. It does not train models or calculate DHW. Historical
research scripts remain separate from this validated-data path.

### Reproduce a regional extraction

Inputs are local; no downloads or archive extraction are required:

- Reef archive: `data/reef-polygons/WCMC008_CoralReefs2018_v4_1.zip`
- Layer: `14_001_WCMC008_CoralReefs2018_v4_1/01_Data/WCMC008_CoralReef2018_Py_v4_1.shp`
- SST: `data/sst/sst.day.mean.YEAR.nc`

For a Great Barrier Reef regional example, with bounds ordered west, south,
east, north:

```bash
uv run python reef_data.py \
  --years 2004 \
  --bounds 142 -25 155 -10 \
  --output-dir data/reef-sst/gbr-2004

uv run pytest tests/test_reef_data.py
uv run ruff check reef_data.py tests/test_reef_data.py
```

Supply more years with `--years 1981 1982 1983`; each is processed and closed
before the next. Supply only completed `.nc` files. The CLI requires a region.
Bounds use `[-180,180]`; west greater than east explicitly crosses the dateline.
Selection includes cells whose **centers** lie within the bounds.
Existing outputs are refused unless `--overwrite` is supplied.
Use `uv run python reef_data.py --help` for archive/layer overrides. An extracted
EPSG:4326 shapefile can also be passed as `--reef-archive PATH.shp`; preserve its
sidecar files and consult the original archive's license.

### Spatial and time rules

- NOAA OISST is daily, 0.25° (about 28 km north–south), in degrees Celsius.
  The input product is declared v2.1. Dedicated conflicting product-version
  metadata fails validation. Older PSL headers may not explicitly verify v2.1:
  outputs distinguish a user declaration from source-confirmed version evidence
  and preserve the original global and SST attributes. Confirm distribution
  provenance before claiming that such headers independently verify v2.1.
- Polygons must declare EPSG:4326. Every source feature is retained through
  `make_valid` repair or the run fails with positional feature indices and reasons.
  Reports include input count, invalid count, repairs, and non-polygonal remnants.
  Repair can change topology; collapsed features are not silently discarded.
- Crossing polygon rings and their holes are unwrapped before planar geometry
  repair, then split/wrapped at ±180. Disconnected parts are handled independently.
  This assumes reef polygon edges do not intentionally span more than 180°.
  Output longitudes use `[-180,180)` while `source_lon` preserves the native value.
- The mask is **all-touched boolean**: any polygon touching a cell marks it as
  reef-associated. Raster edges are half a grid spacing beyond cell centers.
  This retains small reefs but overstates spatial extent relative to actual reef
  area. Cell counts are not reef-area percentages or fractional coverage.
- `METADATA_I` is not an identifier. Stable integer cell IDs are computed from
  canonical grid coordinates, independently of SST validity or feature ordering:
  `lat_index = round((lat + 89.875) / 0.25)`,
  `lon_index = round((lon + 179.875) / 0.25)`,
  `cell = lat_index * 1440 + lon_index`.
- Grid dimensions, unique coordinates, strict spatial monotonicity, quarter-degree
  spacing/center alignment, units, source identity, and cross-year grid equality
  are validated. Duplicate, out-of-order, non-midnight, or out-of-year dates fail.
- Missing calendar dates are explicitly reindexed, never compressed out of time.
  Each annual period is January 1–December 31, except 1981 starts September 1.
  Leap years have 366 expected observations. Partial/current years retain unknown
  slots through December 31; these placeholders are **not future forecasts**.
- SST is selected spatially before loading. Source Dask chunks are bounded
  (31 days × 90 latitudes × 180 longitudes), evaluated single-threaded, with at
  most one year's extracted reef-cell array in memory. SHA-256 hashing streams
  source bytes with bounded memory; it adds an input-read pass.

### Output schema and provenance

Each `reef_sst_YEAR.nc` has dimensions **`time × cell`**. Cell IDs are stable
across years, even when a cell has no valid SST in one year.

| Field | Dimensions | Meaning |
| --- | --- | --- |
| `time`, `cell` | Coordinate axes | Daily calendar and stable canonical grid-cell ID |
| `lat`, `lon` | `cell` | WGS84 centers; normalized longitude |
| `lat_index`, `lon_index` | `cell` | Canonical global OISST grid indices |
| `source_lat_index`, `source_lon_index`, `source_lon` | `cell` | Native indices and longitude in this source file |
| `sst` | `time, cell` | Observed °C; unavailable/non-finite observations remain NaN |
| `sst_valid` | `time, cell` | 1 for finite SST; 0 for unavailable observations |
| `source_time_present` | `time` | Whether the date existed in the source, independently of cell validity |
| `valid_observation_count` | `cell` | Number of finite observations in this annual period |
| `expected_observation_count` | `cell` | Expected calendar days, including missing dates |
| `valid_coverage_fraction` | `cell` | Valid count divided by expected count; zero means no valid coverage, not safe conditions |
| `source_file`, `source_year` | `time` | Intended annual source; consult `source_time_present` for absent dates |

Annual coverage summaries must not be summed once per day or averaged blindly
across years. To aggregate coverage, sum annual valid and expected counts, then
divide. Construct later temporal windows by cell ID and timestamp, not flattened
DataFrame row order. Selected years can be nonconsecutive: downstream consumers
must explicitly handle gaps between annual partitions.

Global attributes preserve original source metadata, product-version evidence,
source/archive SHA-256, layer, region, repair report, missing-data semantics, and
scientific limitations. `manifest.json` records sources, output names, software
versions, expected/source day counts, and missing dates. Outputs contain no
run-clock timestamp; reproducibility means equal scientific arrays and recorded
provenance for identical inputs/configuration/software, not guaranteed identical
NetCDF bytes across library versions. A failed multi-year run may leave completed
annual partitions; it does not publish a successful final manifest.

### Reef source licensing and scientific limitations

The WCMC008 Coral Reefs v4.1 compilation (the archive is labelled CoralReefs2018,
released in 2021) is used as a **static spatial mask**, not observed 2018 boundaries.
Its source dates and verification quality vary. It does not establish unchanged
reef boundaries throughout 1981–2026. OISST cell averages cannot resolve individual
reefs, shallow-water conditions, or coral-colony temperatures; land/coastal cells
can remain entirely unknown.

The archive includes the **UNEP-WCMC General Data License (excluding WDPA),
2014-07-21**. Its terms restrict commercial use and redistribution, including
derived works and downloadable map data; permission may be required. The pipeline
copies the archive's license bytes unchanged to `reef_source_LICENSE.txt` (the
supplied license is Windows-1252). Retain the source's required attribution and
check its metadata/current terms before publication.
The repository's software license does not license these data. Do not publish
the extracted cells or mask as unrestricted downloads merely because they were
generated here.

Outputs record the current product citation: UNEP-WCMC, WorldFish Centre, WRI,
TNC (2021), *Global distribution of coral reefs*, version 4.1,
[DOI: 10.34892/t2wk-5t34](https://doi.org/10.34892/t2wk-5t34).
Consult the archived metadata as well: older package citation text can differ
from the current product record; source provenance is preserved, not rewritten.

This is OISST, not NOAA Coral Reef Watch's approximately 5 km CoralTemp product.
CRW MMM cannot be assumed interchangeable with OISST climatology. No climatology,
projected DHW, biological risk, or forecasting skill is inferred by this cleaning
pipeline.

## Stage 2: GBR 28-day experimental SST forecasting

`time_series.py` trains a regional LSTM on Stage 1 reef-cell outputs. The
default model uses 120 observed days and emits 28 daily corrections to a
training-only anomaly-persistence baseline. Inputs are generated lazily from a
bounded in-memory `time × cell` SST matrix; overlapping windows are never
materialized. Every context and target block belongs to one stable cell.

Freeze split dates before model selection. The following dates are an example
development split, not published untouched-test evidence:

```bash
uv run python time_series.py \
  --data data/reef-sst/gbr \
  --train-end 2018-12-31 \
  --validation-end 2021-12-31 \
  --test-end 2025-12-31 \
  --device cuda \
  --checkpoint models/gbr_sst_lstm.pt
```

CUDA automatic mixed precision is enabled when the selected Python process can
access the GPU. `--samples-per-epoch N` explicitly samples training windows with
replacement when a bounded epoch is needed; omitting it uses every eligible
training example. Validation controls scheduling, early stopping, and checkpoint
selection. The test split is evaluated only by `model_eval.py` after model selection.

The versioned checkpoint contains the state dictionary, model configuration,
exact feature order, context and horizon, target units, training-only
climatology and scaling, split dates, seed, source provenance, stable reef-cell
IDs and coordinates, Git state, and validation metrics. Evaluation rejects
incompatible units, features, preprocessing dimensions, source product
contracts, or spatial identities. Legacy format-1 checkpoints remain loadable
with an explicit warning because they do not contain spatial identity metadata.
Git revision/dirty state and annual file hashes are provenance, not equality
requirements for inference. Relocated files and new annual observations are
allowed when their product/mask/missing-data contract and ordered spatial
identity match. Invalid scalers, malformed metadata, non-finite model weights,
and ambiguous input cell IDs are rejected. Unsupported-cell climatology NaNs
remain valid. Loading uses `weights_only=True`, explicit device mapping, and
evaluation mode (including disabled dropout). Only load artifacts from trusted
sources; compatibility checks are not authenticity checks.
Re-evaluate a checkpoint without refitting:

```bash
uv run python model_eval.py \
  --data data/reef-sst/gbr \
  --checkpoint models/gbr_sst_lstm.pt \
  --split test \
  --device cuda
```

Metrics are reported separately for leads 1–28 for both the LSTM and anomaly
persistence. The 28-day output is experimental: publish only the consecutive
lead range that demonstrates useful held-out skill. This stage forecasts SST;
it does not yet calculate projected DHW or predict biological bleaching.



## Results & Evaluation

Historical LSTM metrics and plots were removed because flattened sequences mixed
reef locations and did not constitute trustworthy forecast evidence. No
corrected model-skill claim has been made yet.

New results must compare the corrected LSTM and anomaly persistence on identical
eligible cells and dates, report each forecast lead separately, and preserve an
untouched chronological test period.
---

## Roadmap

- [ ] **Fix climatology baseline calculation**  
  Implement linear regression models for each unique lat × lon point to account for warming trend.  
  *(Current baseline underestimates bleaching events.)*
  
- [ ] **Add error handling and QOL features (config file etc.)**  
  Build an interactive dashboard to visualize bleaching events.  
  
- [ ] **Predict future reef stress**  
  Extend pipeline to generate forecasts for future years and assess how rising sea surface temperature affects coral reefs and their societal impact.  

- [ ] **Bleaching event pipeline**  
  Deliver an automated end-to-end pipeline to predict possible bleaching events in the nearest future using NOAA daily SST data and neural networks.  

- [ ] **Model improvements**  
  Experiment with alternative architectures.  

- [ ] **Visualization**  
  Build an interactive dashboard to visualize bleaching events.  
  

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.  
Note: Dataset licenses (NOAA, UNEP-WCMC) must be respected separately.
