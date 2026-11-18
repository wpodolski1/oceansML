# Repository and ML Review

Date: 2026-11-16

## Assessment and scope

**Research prototype—not yet a reliable bleaching-forecasting system.** Fix data construction and scientific calculations before improving the neural network.

Reviewed all six Python modules, README, and dependency configuration. Verification consisted of static diagnostics and isolated synthetic probes of actual repository functions. This review did not evaluate real SST datasets or trained checkpoints, reproduce training, or validate the README's reported metrics. Full script imports were blocked by missing dependencies; function-level probes are not end-to-end verification.

No source code was changed during the review. Locations below refer to the reviewed revision.

## Highest-priority fixes

### 1. LSTM sequences mix reef locations

**Location:** `time_series.py:57–100`

Sliding windows operate on the entire flattened DataFrame without grouping by location.

**Confirmed:** a synthetic xarray-to-DataFrame pipeline produced a window with longitudes `[10, 20, 10, 20]` and repeated dates, rather than consecutive days at one reef. Every window in that probe mixed sites. The current model therefore cannot be interpreted as the intended temporal forecaster.

**Fix:** establish stable site IDs; sort and window independently per site; require consecutive dates; split using explicit target dates.

**Acceptance:** every sequence belongs to exactly one site, has the specified daily history, and has an explicitly timestamped future target.

### 2. Degree Heating Week calculations are scientifically incorrect

**Location:** `data_visualization.py:100–119`

The implementation accumulates anomalies below the required 1°C threshold. It also converts non-positive anomalies into NaNs, causing otherwise valid rolling sums to become NaN.

Actual-function synthetic probes produced:

| Temperature history above MMM | Current DHW | Correct DHW |
| --- | ---: | ---: |
| 84 days at 1.5°C | 18 | 18 |
| 84 days at 0.5°C | 6 | 0 |
| 42 days at 1.5°C, then 42 at 0°C | NaN | 9 |

Incomplete coverage is replaced with zero, and stress classes default to zero. Missing data can consequently appear as “no stress.”

**Fix:** valid days below the threshold contribute zero; accumulate hotspot values of at least 1°C; preserve missing observations separately; mask incomplete rolling coverage rather than declaring it safe. Validate unique, consecutive daily timestamps because an 84-observation window is not necessarily 84 calendar days.

**Acceptance:** the histories above produce the expected DHW values; missing observations remain unknown; missing or duplicate dates cannot silently change the window duration.

Reference: [NOAA Coral Reef Watch methodology—DHW](https://coralreefwatch.noaa.gov/product/5km/methodology.php#dhw).

### 3. Evaluation uses the test set to influence training

**Location:** `time_series.py:244–329`

The learning-rate scheduler responds to test loss every epoch. That makes this a validation set, not an untouched test set.

**Fix:** use chronological training, validation, and test periods. Exclude the final test period from scheduling, tuning, and model selection. Add geographic holdouts separately to measure transfer to unseen reefs.

Overlapping historical inputs are not inherently leakage; target timing and information availability must be explicit. Scaling is fitted on training inputs in the current implementation; preserve that property.

**Acceptance:** split dates and site membership are recorded; the final test set is evaluated only after model selection.

### 4. Model saving and loading are incompatible

**Locations:** `time_series.py:331–352`; `model_eval.py:48–72`

Training saves a complete model object; evaluation expects a state dictionary. Their default filenames also differ.

**Confirmed:** saving and loading through the respective functions with the same filename raises `TypeError: Expected state_dict to be dict-like`.

**Fix:** save a state dictionary plus architecture configuration, feature order, scalers, training period, and run configuration. Use consistent artifact paths and explicit device mapping. Avoid unrestricted pickle loading for untrusted artifacts.

**Acceptance:** a saved model reloads in a fresh process and produces equivalent predictions, including on CPU.

### 5. Preprocessing and analysis entrypoints contain blockers

- **`reef_data.py:206`:** the yearly `save()` call omits the required `name` argument. Confirmed `TypeError`.
- **`data_visualization.py:121`:** selects 2018 although the configured main analysis dataset is 2019. With combined input this selects support-year data; a 2019-only synthetic input raises `KeyError`.
- **`reef_data.py:193–206`:** relative input paths can be prefixed twice. Existing-output checks inspect a different directory from the write destination.
- **`reef_data.py:126–145`:** merging opens every directory entry. The visualization entrypoint writes `merged_baseline.nc` into its source directory, so subsequent runs include their previous output.

**Fix:** use explicit input/output configuration and target-period arguments. Select intended source files and exclude generated outputs. Make repeated runs reproducible.

**Acceptance:** one unseen yearly input can be processed and merged; a second run correctly skips or reproduces output; the configured main year, not its support period, is returned.

### 6. Reef masks and point identities need correction

**Locations:** `reef_data.py:73–80,116,167`

- The affine transform treats coordinate centers as pixel edges, shifting rasterized reefs by half a grid cell when the source coordinates represent centers.
- `dropna("points")` removes a location when any timestamp is missing during a file's period.
- Resetting the spatial index removes the coordinate index needed for reliable alignment of differing point sets across years. Depending on the data, merges can fail or combine positions without stable geographic identity.

**Fix:** validate regular grid spacing and correct affine origins; select locations from the reef mask independently of SST completeness; preserve stable spatial identity across files. Check normalized longitude uniqueness and antimeridian geometry.

**Acceptance:** synthetic polygons map to the correct cells for both latitude orientations; a missing observation in one year does not remove or relocate a reef point in another year.

### 7. Installation is not reproducible as documented

**Locations:** `README.md:101–114`; `requirements.txt`; `pyproject.toml`

The README recommends `pip install -r requirements.txt`, but that file is a Conda export. The reviewed environment lacked imported `tqdm` and `IPython`, and they are not declared in the project dependency list. The README's Python version badge also differs from the manifest's Python 3.11 constraint.

**Fix:** standardize on the existing `pyproject.toml` and `uv.lock`; declare direct dependencies; separate notebook-only styling from script execution; document configuration and runnable commands. Provide an environment-variable example without secrets and avoid failing at import time when configuration is absent.

**Acceptance:** a clean documented installation can run a small preprocessing/training/evaluation scenario without undeclared dependencies or hidden configuration.

## Scientific interpretation and reporting

### Climatology must use a documented, product-compatible method

**Locations:** `data_visualization.py:86–95`; `README.md:134–138`

The current calculation takes monthly means from whichever baseline files are present without validating the required period or coverage.

Do not simply raise the baseline to follow warming or adjust it to obtain expected bleaching counts. NOAA's regression-based methodology derives climatology from a longer record and recenters it to a fixed historical reference; it does not assume coral tolerance tracks warming. Match the SST product, grid, reference period, and method explicitly. Do not assume climatologies from different SST products are interchangeable.

**Acceptance:** required years and all 12 calendar months are validated; baseline provenance is recorded; selected locations are compared against an appropriate reference product.

Reference: [NOAA Coral Reef Watch methodology—climatology](https://coralreefwatch.noaa.gov/product/5km/methodology.php#clim).

### Alert classification needs an explicit version

**Location:** `data_visualization.py:117–119`

Exactly 8 DHW is assigned the lower category. Current NOAA methodology begins Alert Level 2 at 8 DHW and includes higher alert levels absent here.

**Fix:** implement and document a chosen classification version, with boundary checks at every threshold.

Reference: [NOAA Coral Reef Watch methodology—alert areas](https://coralreefwatch.noaa.gov/product/5km/methodology.php#baa).

### Grid-cell percentages are not reef damage

**Locations:** `data_visualization.py:133–163`

`bleaching_bins` measures reef-grid-cell-days. `unique_bleaching_bins` counts grid cells equally after reducing over time. Neither weights geographic cell area or fractional reef coverage, and thermal stress classes are not observations of bleaching or damage.

**Fix:** label denominators and proxy meaning explicitly. For reef-area percentages, weight by geographic area and polygon overlap. Damage estimation requires independent biological outcome data.

The unique-cell histogram succeeded on a synthetic 2D grid. Its NaN-to-integer cast emits a warning; filter finite values before casting. There is no confirmed dimensionality failure in its `np.bincount` call.

### Do not average ordinal alert codes

**Location:** `data_visualization.py:222–225`

A mean of category codes can imply a class that never occurred. The map's 0–7 range also does not match the implemented 0–4 classification.

**Fix:** plot maximum alert, days above a threshold, or continuous DHW, with matching labels and scale.

### Current model scope is narrower than the long-term goal

After repairing per-site sequences, the current target construction represents one-step SST prediction using 89 input rows from a 90-row window. It is not a demonstrated yearly reef-damage forecast or a validated bleaching-event forecast.

The README's RMSE, MAE, R², and bias need to be regenerated under the corrected data and evaluation contracts. Aggregate SST accuracy alone does not establish useful warning skill for extreme events.

## Recommended roadmap

| Stage | Deliverable | Acceptance gate |
| --- | --- | --- |
| **1. Correctness first** | Repair sequences, DHW, missing-data handling, masks, paths, year selection, checkpoints, and installation | Small fixture runs preprocessing → training → reload → evaluation; consumer-visible numerical edge cases have regression coverage |
| **2. Establish trustworthy skill** | Explicit multi-year splits; persistence, seasonal-climatology, and simple regression baselines; reproducible runs | Untouched-test results establish whether the LSTM beats baselines, including regional and extreme-temperature errors |
| **3. Build actual bleaching forecasts** | Defined forecast horizons; predicted SST combined with observed history to calculate future DHW; uncertainty estimates | Event precision/recall, warning lead time, and calibration are evaluated against independent bleaching observations |
| **4. Operationalize** | Configurable CLI, versioned datasets/artifacts, lazy window loading, CI, scheduled inference, then dashboard | Repeatable runs expose data coverage, provenance, model version, and uncertainty |

### Model development priorities

1. Retain the small LSTM as a benchmark; correct its inputs and evaluation first.
2. Compare against persistence, seasonal climatology, and simple regression or tree-based models using identical splits and information availability.
3. Record explicit training periods, seeds, configuration, dataset versions, and checkpoint/scaler provenance.
4. Evaluate errors by region, season, forecast horizon, and heat-stress regime—not only aggregate metrics.
5. Avoid materializing every overlapping window and repeated scaled copies when scaling to larger datasets; use lazy sequence loading and measure memory use.
6. Consider larger networks or alternative architectures only after a trustworthy baseline demonstrates where additional model capacity is useful.

Long-term reef-damage claims require biological outcome data and separate validation. A dashboard should expose validated outputs, not precede scientific correctness.

## CI repair review — 2026-11-18

This section supersedes the installation/checkpoint blockers above for the current
pipeline, not the unresolved historical scientific findings. Existing uncommitted
work was retained; no commit, push, or merge was performed.

### Initial failure classification

| Failure | Root cause | Repair |
| --- | --- | --- |
| Rasterio `libexpat.so.1` import failure | GDAL's wheel still needs host Expat, zlib, and libstdc++; `ldd` found all three unresolved on NixOS | Content-pinned Nix shell derives library paths; Ubuntu installs named distribution runtime packages |
| 21 Ruff violations | 13 unused imports, 3 import-order issues, 3 undefined-name diagnostics, bare exception and swallowed-exception block | Mechanical import cleanup; restore missing torch import; remove ineffective unbound-client cleanup |
| Formatting | Five existing Python files | Formatting-only edits; the final mechanical formatting pass preserved Python ASTs |
| 63 BasedPyright errors | Three geometry scalar accesses inherited pandas' broad scalar union | Matching GeoPandas/Shapely stubs and explicit geometry narrowing |
| 11 other BasedPyright errors | Missing IPython, optional environment paths, undefined torch, unbound client | Optional notebook boundary, explicit required environment lookup, runtime name repairs |
| 454 `Any` warnings | Dynamic NumPy/Torch/xarray/argparse values | Precise array, tensor, metadata, and parsed-argument boundaries |
| 409 `Unknown` warnings | Incomplete upstream signatures and cascading untyped arguments/returns | Targeted annotations and callable protocols; no runtime wrappers around scientific algorithms |
| 147 other typing warnings | Missing parameter/stub/class types, overrides, ignored results, imports and string concatenation | Local annotations, explicit discarded results, and mechanical cleanup |

`typeCheckingMode = "recommended"` is unchanged. Its `failOnWarnings=true`
behavior remains active. No diagnostic exclusions, ignore comments, baseline
file, or globally suppressed rules were added. `typing-extensions` was already
locked transitively and is now declared directly for Python 3.11's `@override`.
The only newly resolved packages are development-only `types-geopandas` and
`types-shapely`, matching the installed runtime series.

Published `types-rasterio` requires Python >=3.12 and targets Rasterio >=1.5.
The project remains on Python 3.11 / Rasterio 1.4; `typings/rasterio` declares
only the existing in-memory uint8 rasterization and affine-construction calls.
These partial signatures are maintenance debt, not substitute implementations.
Existing polygon/raster regression tests exercise the real native implementation.

### Native environment and Ubuntu evidence

The observed failure is the local NixOS loader environment, not a missing Python
package. The Python lock alone cannot supply system shared libraries. A pinned
`shell.nix` supplies Nix Python 3.11.18, uv 0.13.7, and
`lib.makeLibraryPath [ expat zlib stdenv.cc.cc.lib ]`; it needs no hardcoded store
paths or global loader configuration.

The GitHub job uses Ubuntu 24.04, immutable action revisions, uv 0.13.7,
`.python-version` (3.11.18), locked dependency synchronization, and explicit
`libexpat1 zlib1g libstdc++6` installation. Those runtime packages are normal
Ubuntu dependencies; no Nix library path is exported on Ubuntu.

Verification used an isolated Ubuntu 24.04.5 userspace, not host Nix libraries.
The official
[Ubuntu Base image](https://cdimage.ubuntu.com/ubuntu-base/releases/24.04/release/)
SHA-256 was `e77b6f10c2590cef872b33ee9f635a0e3fd1f57fb074c0e52b5c7f56147a0c86`,
matching the published checksum. Runtime versions were Expat
`2.6.1-2ubuntu0.6`, zlib `1:1.3.dfsg-3.1ubuntu2.3`, and libstdc++
`14.2.0-4ubuntu2~24.04.1`. Rasterio 1.4.4 / GDAL 3.10.3, Torch 2.15.0,
and netCDF4 1.7.5.1.post1 imported successfully on both platforms.

Ubuntu setup needed `APT::Sandbox::User=root` only inside the disposable,
single-UID user namespace, and bind-mounted `/proc` and `/dev` there. These
container accommodations are not repository or GitHub workflow changes.
The uv download cache was reused; each platform had its own Python environment.
An actual hosted GitHub Actions event was not dispatched. The Ubuntu image and
apt security updates are not byte-for-byte frozen; the Python lock and action
revisions are. No hosted/cold-cache timing claim is made.

### Checkpoint v2 findings and repairs

- Serialization remains state dictionaries plus tensors and primitive metadata,
  loaded with `weights_only=True` and explicit device mapping.
- Loading validates positive finite scales, float climatology shape and range,
  architecture values, chronological split dates, v2 metadata consistency,
  unique finite spatial metadata, and finite floating model-state tensors.
  NaNs in unsupported-cell climatology remain allowed.
- Loaded models enter evaluation mode. Previously, dropout could remain active
  when callers used `load_checkpoint` directly; evaluation entrypoints already
  selected evaluation mode.
- Input IDs must be nonempty unique integers on the cell axis; fractional IDs
  previously could be silently truncated to checkpoint IDs. Non-finite
  coordinates are rejected before constructing inference sequences.
- Git commits and dirty flags remain provenance only. Different Git revisions,
  relocated source files, and new annual source hashes do not invalidate an
  otherwise compatible checkpoint. Product, mask identity, units, feature
  order, preprocessing dimensions, and ordered cells/coordinates still matter.
- Format 1 remains loadable with an explicit compatibility warning and the
  checks supported by its metadata. It cannot establish spatial identity.
- Regression coverage includes nonzero model predictions and inverse-scaled
  inputs reloaded in a fresh CPU subprocess, dropout-disabled repeated inference,
  v1/v2 missing climatology, incompatible input metadata, and malformed artifacts.

No SST, climatology, rasterization, DHW, feature, split, loss, optimizer, or metric
formula was changed. Boundary casts do not convert numerical values. Scientific
pipeline expectations in existing tests were retained.

### Commands and observed results

Initial classification:

```bash
ldd .venv/lib/python3.11/site-packages/rasterio/_base.cpython-311-x86_64-linux-gnu.so
uv run --no-sync basedpyright --outputjson > /tmp/oceansml-types-before.json
uv run --no-sync ruff check . --output-format json > /tmp/oceansml-ruff-before.json
uv run --no-sync ruff format --diff > /tmp/oceansml-format-before.diff
```

These produced the classified failures above. Targeted BasedPyright diagnostics
were used during annotation work. Intermediate integration found six new
lint issues and cumulative-array annotation mismatches; these were corrected,
not suppressed. The initial reported native failure blocked collection; no
pre-change complete pytest result is claimed.

Complete NixOS sequence:

```bash
nix-shell --run 'set -e; uv sync --locked --group dev; uv run ruff check .; uv run ruff format --check .; uv run basedpyright; uv run pytest'
```

The same commands ran under Ubuntu's own Python, preceded by `uv python install`
and the native package installation documented in README. Results on each:
locked sync succeeded; Ruff lint and formatting passed; BasedPyright reported
**0 errors, 0 warnings**; pytest reported **98 passed**. The initial integrated
suite took 8.60 seconds on NixOS and 7.57 seconds in Ubuntu; these are local
measurements, not hosted runner benchmarks.

Workflow syntax validation:

```bash
nix-shell -I nixpkgs=https://github.com/NixOS/nixpkgs/archive/04906337b6d53e8bb0eaf0470ae9f070a4d4eb55.tar.gz -p actionlint --run 'actionlint .github/workflows/pr.yml'
```

Actionlint 1.7.12 exited successfully without diagnostics. The three CLI
`--help` commands also succeeded inside `nix-shell`:

```bash
uv run python reef_data.py --help
uv run python time_series.py --help
uv run python model_eval.py --help
```

A synthetic five-class histogram rendered with matplotlib's Agg backend;
headless `show()` emitted its expected non-interactive-backend warning.
No real NOAA data, model training experiment, GPU, or biological evaluation was
required for these checks.

### Remaining technical and scientific debt

- Pytest still reports **23 upstream Rasterio/Affine PendingDeprecationWarnings**
  about affine `*` versus `@`. They are visible, not filtered. A future compatible
  native dependency update should address them without changing raster results.
- Historical `data_visualization.py` DHW thresholds, missing-coverage semantics,
  fixed-year selection, climatology provenance, and ordinal-category plotting
  remain unvalidated/incorrect as described above. CI success is not scientific
  approval of that research script. Its numerical formulas were intentionally
  untouched in this CI repair.
- Legacy v1 artifacts lack spatial metadata; neither warnings nor dimensional
  checks can reconstruct it. Do not treat v1 spatial compatibility as proven.
- Seeds and source/Git metadata support reproducibility but do not guarantee
  cross-hardware CUDA determinism or reconstruction of uncommitted source edits.
  These checkpoints are inference artifacts, not complete optimizer/RNG resume
  snapshots; retain the environment lock and producing source alongside them.
- Real-data held-out forecasting skill, seasonal-baseline comparisons, and
  projected thermal-stress validation remain separate scientific acceptance
  gates. No numerical improvement or bleaching prediction is claimed here.

## Development checkpoint audit — 2026-11-18

**Status: WIP development checkpoint; not ready to merge or publish scientific
results.** Saving progress in Git is not a production-readiness claim. The audit
covered the complete uncommitted change set, including the earlier Stage 1 and
forecasting work, not just the CI repairs. The scientific issues below remain
unfixed in this checkpoint and require focused fixes, regression coverage, and
review in a draft pull request.

References below identify the audited source lines. No critical issue was
identified. High and medium findings are unresolved unless explicitly marked
otherwise.

### High severity — unresolved correctness findings

1. **H1: Unsupported climatology can report perfect empty evaluations.**
   `time_series.py:475–480,620–643,1063–1077`.
   Window eligibility checks finite observed SST but not the fitted climatology.
   A cell without training coverage can become eligible in validation with NaN
   features and baselines. Non-finite errors are discarded, then zero metric
   counts are clamped to one. A synthetic probe admitted 362 validation examples
   but returned counts `[0, 0]` and model/baseline RMSE `[0.0, 0.0]`. These metrics
   can drive checkpoint selection. Exclude unsupported windows, reject empty
   metric horizons, and prevent silently different model/baseline populations.

2. **H2: Checkpoint v2 accepts missing source identity.**
   `time_series.py:309–320,707–724,833–840`;
   `tests/test_time_series.py:42–47`.
   Missing product, product version, reef-mask hash, and missing-data semantics
   become empty strings. Matching empty contracts pass compatibility checks.
   The passing training/evaluation fixture omits three of these identity fields.
   Require correctly typed, nonempty identity metadata; continue treating Git
   differences and annual source hashes as provenance rather than equality
   requirements.

3. **H3: Overwrite leaves unrequested annual partitions available to training.**
   `reef_data.py:646–650,765–769`; `time_series.py:215–216`.
   Reproduced by generating 2000–2001 and overwriting with only 2001: the new
   manifest lists 2001, but directory ingestion still loads both years. Reconcile
   old generated partitions or make the manifest authoritative; do not silently
   expand the requested scientific data population.

### High severity — staging hazard addressed for this checkpoint

4. **H4: Restricted raw data and an unintended nested repository were eligible
   for staging.**
   Audited `.gitignore:34–37`, `data/reef-polygons/`, and `data/sst/`.
   The local reef archive is 220,904,276 bytes. Its bundled UNEP-WCMC license
   prohibits redistribution without permission; it must not be published as
   software source. `git add --dry-run --all` also failed because `data/sst/`
   contained an unborn nested repository.
   At the user's request, only `data/sst/.git` was deleted; SST files were
   preserved. The checkpoint's ignore rules exclude the local data tree,
   named reef-source archives/extractions, and the generated reef map. Staging
   uses an explicit source/test/configuration/documentation allowlist. No public
   sample data were tracked under `data/`; no tracked samples were removed.
   Dataset licensing restrictions remain in force independently of Git ignores.

### Medium severity — unresolved findings

5. **M1: Failed overwrite retains a misleading successful manifest.**
   `reef_data.py:729–738,765–769`.
   A synthetic run replaced the first annual output, then rejected the next
   source's units. The previous manifest remained unchanged with an obsolete
   source hash. Invalidate the old manifest before replacing outputs or publish
   transactionally. The current failure path contradicts the README guarantee.

6. **M2: Numeric validation precedes destructive float32 conversion.**
   `time_series.py:155–164,525–528,979–989`.
   Accepted `anomaly_scale=1e-50` and `latitude_mean=1e300` produced infinite
   features; a finite-weight model then produced NaN predictions. Separately, a
   finite float64 checkpoint output bias of `1e100` passed validation, became
   float32 infinity during loading, and produced `[inf, inf]` predictions.
   Validate the effective numeric dtype and reject non-finite loaded parameters
   and features rather than silently clipping or discarding their errors.

7. **M3: Training accepts split timestamps that checkpoint loading rejects.**
   `time_series.py:70–82,426–433,968–970`.
   `train_end="2018-12-31T12:00:00"` passes training validation; validation starts
   at noon on January 1 and selects January 2 as its first daily target. Loading
   the resulting checkpoint rejects its non-midnight boundary. Share normalized,
   timezone-naive daily-date validation across training and loading.

8. **M4: Git provenance depends on the caller's working directory.**
   `time_series.py:681–699`.
   The same imported source recorded its commit in the repository but returned
   `{"commit": null, "dirty": null}` from an external directory. An unrelated
   working repository can instead supply the wrong revision. Query Git in the
   producing source repository; do not turn revision equality into an inference
   requirement.

9. **M5: Extracted-shapefile outputs falsely claim license preservation.**
   `reef_data.py:656–665,723–726`.
   The supported `.shp` input writes a generic permission notice while recording
   that the archive license was copied. The synthetic reproduction confirmed
   both the fallback text and the conflicting metadata. Distinguish unavailable
   source-license bytes from an actual copied license.

### Low severity and verification limits

- **L1: Hosted cold-cache CI remains unverified.**
  `.github/workflows/pr.yml:16–30`; `uv.lock:1191–1204`.
  Workflow commands match the successfully tested isolated Ubuntu environment,
  but Linux Torch dependencies include substantial CUDA downloads. No hosted
  timing/capacity failure was demonstrated. Neither the Ubuntu runner image nor
  apt security updates are byte-for-byte frozen.
- Historical DHW/classification and visualization defects remain unchanged and
  documented above. Passing CI does not validate those algorithms or establish
  forecasting skill.
- Rasterio's local stubs were checked against installed Rasterio 1.4.4 signatures
  and implementation. A real uint8 mask probe succeeded; four deliberately
  unsupported/invalid calls still generated type errors. The stubs are narrow,
  not catch-all suppressions, and must track future runtime/API changes.
- Fresh-process CPU checkpoint coverage genuinely exercises nonzero predictions,
  inverse preprocessing, disabled dropout, and repeated inference. Git inequality
  is accepted. v1 coverage uses downgraded v2 payloads, not independently sourced
  historical artifacts; cross-hardware/GPU determinism remains unproven.
- The spatial and forecasting corrections are intentional: half-cell raster
  edges, stable cells, per-cell windows, chronological targets, training-only
  preprocessing, and validation-only model selection. Existing tests cover key
  corrections, but not the failure cases above. Deliberately shifted
  validation/test distributions and the actual anomaly-persistence formula also
  need stronger regression coverage. A pure seasonal-climatology baseline remains
  an outstanding scientific acceptance gate.
- The previously recorded complete CI sequence passed 98 tests with zero Ruff
  or BasedPyright diagnostics. This audit used source review and focused
  reproductions, not another full-suite run. Green CI is insufficient for merge
  readiness until the missing regression cases are addressed.

No scientific fixes are included merely to create this development checkpoint.
Resolve H1–H3 and M1–M5 in subsequent focused commits, preserving missing data
and numerical semantics, then review the resulting draft PR before merging.
