# OceansML — Agent Instructions

## Project purpose

OceansML is a scientific machine learning project forecasting sea surface temperature (SST) at reef-associated locations and estimating projected coral reef thermal stress.

Prioritize scientific correctness, reproducibility, maintainability, and honest reporting over model complexity.

## Engineering principles

- Follow Python best practices, PEP 8, and modern type annotations.
- Prefer simple, modular, testable implementations.
- Use `uv` for Python dependency management.
- Use Ruff for linting and formatting and pytest for testing.
- Avoid unnecessary abstractions, premature optimization, and new dependencies without justification.
- Keep functions small, with explicit inputs and outputs.
- Prefer vectorized operations and xarray/Dask for large scientific datasets.
- Do not load entire large datasets into memory without justification.
- Preserve existing working behavior unless changes are intentional and tested.

## Machine learning rules

- Never mix observations from different spatial locations in a single temporal sequence.
- Never use future information when constructing training features.
- Split datasets chronologically for forecasting evaluations.
- Fit scalers and preprocessing transformations using training data only.
- Keep test data separate from hyperparameter tuning and model selection.
- Compare learned models against persistence and seasonal baselines.
- Report forecast performance separately by prediction horizon.
- Use reproducible random seeds and record experiment configurations.
- Never claim improvement without measured evidence.

## Scientific correctness

- Distinguish observed SST from model-generated forecasts.
- Distinguish projected thermal stress from observed coral bleaching or mortality.
- Document NOAA data products, versions, resolution, units, and climatology.
- Validate MMM, HotSpot, and DHW calculations against documented methodologies.
- Handle missing observations explicitly.
- Preserve geographic identity and coordinate alignment.
- Do not invent performance metrics or scientific conclusions.
- Label unvalidated outputs as experimental.

## Testing and validation

- Add tests for new functionality and regression tests for bug fixes.
- Prioritize edge cases involving time, geography, missing data, and data leakage.
- Run relevant pytest tests and Ruff checks after modifications.
- Clearly report tests that could not be executed.
- Never silently modify tests merely to make a failing implementation pass.

## Agent workflow

1. Inspect relevant existing code and documentation before proposing changes.
2. For substantial changes, explain the intended approach before implementation.
3. Make focused changes rather than rewriting unrelated modules.
4. Validate behavior with appropriate tests.
5. Summarize what changed, why, and what remains unresolved.

## Git conventions

- Work on feature branches rather than modifying `main` directly.
- Prefer small, logically grouped commits.
- Never force-push or discard user changes without explicit approval.
- Do not commit credentials, large datasets, model weights, or generated caches.

## Project context

Read `PLAN.md` for the development roadmap and `REVIEW.md` for existing correctness findings when relevant.

Do not treat historical README metrics as validated forecasting skill until independently reproduced.
