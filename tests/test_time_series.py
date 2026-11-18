import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch import Tensor

from time_series import (
    SSTLSTM,
    ModelConfig,
    ReefSequenceDataset,
    SplitConfig,
    TrainingConfig,
    evaluate_checkpoint,
    fit_preprocessing,
    load_reef_sst,
    train_forecaster,
)


def _write_annual_fixture(directory: Path, year: int) -> Path:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    day = np.arange(len(dates), dtype=np.float32)
    seasonal = np.sin(2 * np.pi * day / len(dates)).astype(np.float32)
    warming = np.float32(0.1 * (year - 2000))
    sst = np.column_stack(
        [20.0 + seasonal + warming, 30.0 + seasonal + warming]
    ).astype(np.float32)
    dataset = xr.Dataset(
        data_vars={"sst": (("time", "cell"), sst)},
        coords={
            "time": dates,
            "cell": np.array([100, 200], dtype=np.int64),
            "lat": ("cell", np.array([-20.0, -10.0], dtype=np.float32)),
            "lon": ("cell", np.array([145.0, 150.0], dtype=np.float32)),
        },
        attrs={
            "source_file": f"source-{year}.nc",
            "source_sha256": f"source-hash-{year}",
            "reef_archive_sha256": "reef-hash",
            "product_version_evidence": "fixture",
        },
    )
    dataset["sst"].attrs["units"] = "degC"
    path = directory / f"reef_sst_{year}.nc"
    cast(Callable[[Path], None], dataset.to_netcdf)(path)
    return path


def _fixture_files(tmp_path: Path) -> list[Path]:
    return [_write_annual_fixture(tmp_path, year) for year in range(2000, 2004)]


def test_sequences_preserve_cell_identity_and_chronological_split(
    tmp_path: Path,
) -> None:
    data = load_reef_sst(_fixture_files(tmp_path))
    split = SplitConfig("2001-12-31", "2002-12-31", "2003-12-31")
    model = ModelConfig(context_days=30, horizon=28, hidden_dim=8)
    preprocessing = fit_preprocessing(data, split.train_end)
    validation = ReefSequenceDataset(data, preprocessing, model, split, "validation")

    for index in (0, len(validation) // 2, len(validation) - 1):
        _, residual, baseline, actual, encoded = validation[index]
        target_start, cell_index = divmod(int(encoded), data.sst.shape[1])
        assert data.dates[target_start] >= pd.Timestamp("2002-01-01")
        assert data.dates[target_start + model.horizon - 1] <= pd.Timestamp(
            "2002-12-31"
        )
        np.testing.assert_allclose(
            actual.numpy(),
            data.sst[target_start : target_start + model.horizon, cell_index],
        )
        np.testing.assert_allclose(
            baseline.numpy() + residual.numpy(), actual.numpy(), rtol=1e-6
        )
        if cell_index == 0:
            assert float(actual.max()) < 25.0
        else:
            assert float(actual.min()) > 25.0


def test_missing_observation_excludes_every_affected_window(
    tmp_path: Path,
) -> None:
    files = _fixture_files(tmp_path)
    data = load_reef_sst(files)
    missing_candidates = np.flatnonzero(
        np.asarray(data.dates == pd.Timestamp("2002-06-15"))
    )
    missing_index = int(cast(np.int64, missing_candidates[0]))
    data.sst[missing_index, 0] = np.nan
    finite = np.isfinite(data.sst).astype(np.int32)
    data.finite_cumsum = np.vstack(
        [
            np.zeros((1, data.sst.shape[1]), dtype=np.int32),
            finite.cumsum(axis=0),
        ]
    )
    split = SplitConfig("2001-12-31", "2002-12-31", "2003-12-31")
    model = ModelConfig(context_days=30, horizon=28, hidden_dim=8)
    preprocessing = fit_preprocessing(data, split.train_end)
    validation = ReefSequenceDataset(data, preprocessing, model, split, "validation")

    for encoded in cast(Iterable[np.int64], validation.encoded_examples):
        target_start, cell_index = divmod(int(encoded), data.sst.shape[1])
        if cell_index == 0:
            assert not (
                target_start - model.context_days
                <= missing_index
                < target_start + model.horizon
            )


def test_lstm_emits_28_direct_residuals() -> None:
    config = ModelConfig(context_days=120, horizon=28, hidden_dim=16)
    model = SSTLSTM(config)
    output = cast(Tensor, model(torch.randn(4, config.context_days, config.input_dim)))

    assert output.shape == (4, 28)
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_training_checkpoint_reloads_for_independent_evaluation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    files = _fixture_files(tmp_path)
    data = load_reef_sst(files)
    checkpoint = tmp_path / "model.pt"
    split = SplitConfig("2001-12-31", "2002-12-31", "2003-12-31")
    result = train_forecaster(
        data=data,
        split_config=split,
        model_config=ModelConfig(
            context_days=14, horizon=7, hidden_dim=8, num_layers=1
        ),
        training_config=TrainingConfig(
            epochs=1,
            batch_size=16,
            patience=1,
            seed=7,
            samples_per_epoch=32,
        ),
        checkpoint_path=checkpoint,
        device=torch.device("cpu"),
    )
    payload = cast(
        dict[str, object], torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    contract = cast(dict[str, object], payload["data_contract"])
    assert payload["format_version"] == 2
    assert contract["target_name"] == "sst"
    assert contract["target_units"] == "degC"
    assert contract["feature_names"] == [
        "sst_anomaly_z",
        "day_sin",
        "day_cos",
        "latitude_z",
        "longitude_sin",
        "longitude_cos",
    ]
    assert contract["context_days"] == 14
    assert contract["forecast_horizon_days"] == 7
    assert set(cast(dict[str, object], payload["git"])) == {"commit", "dirty"}

    reloaded = evaluate_checkpoint(
        data_inputs=files,
        checkpoint_path=checkpoint,
        split="test",
        batch_size=64,
        num_workers=0,
        device=torch.device("cpu"),
    )

    legacy_payload = dict(payload)
    legacy_payload["format_version"] = 1
    _ = legacy_payload.pop("data_contract")
    legacy_checkpoint = tmp_path / "legacy-model.pt"
    torch.save(legacy_payload, legacy_checkpoint)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="time_series"):
        legacy_metrics = evaluate_checkpoint(
            data_inputs=files,
            checkpoint_path=legacy_checkpoint,
            split="test",
            batch_size=64,
            num_workers=0,
            device=torch.device("cpu"),
        )
    assert "Legacy checkpoint format 1" in caplog.text
    assert len(legacy_metrics["model"]["rmse"]) == 7

    mismatched_directory = tmp_path / "mismatched"
    mismatched_directory.mkdir()
    mismatched_files: list[Path] = []
    for path in files:
        with cast(Callable[[Path], xr.Dataset], xr.open_dataset)(path) as dataset:
            mismatched = cast(Callable[[], xr.Dataset], dataset.load)()
            mismatched.coords["cell"] = dataset["cell"].values + 1
        mismatched_path = mismatched_directory / path.name
        cast(Callable[[Path], None], mismatched.to_netcdf)(mismatched_path)
        mismatched_files.append(mismatched_path)
    with pytest.raises(ValueError, match="cell_ids"):
        _ = evaluate_checkpoint(
            data_inputs=mismatched_files,
            checkpoint_path=checkpoint,
            split="test",
            batch_size=64,
            num_workers=0,
            device=torch.device("cpu"),
        )

    assert checkpoint.is_file()
    assert result["best_epoch"] == 1
    assert len(reloaded["model"]["rmse"]) == 7
    assert all(count > 0 for count in reloaded["model"]["count"])


@pytest.mark.parametrize(
    ("coordinate", "values", "message"),
    [
        ("cell", [100.1, 200.1], "cell IDs"),
        ("cell", [100, 100], "cell IDs"),
        ("lat", [float("nan"), -10.0], "coordinates"),
        ("lon", [145.0, float("inf")], "coordinates"),
    ],
)
def test_loader_rejects_ambiguous_spatial_identity(
    tmp_path: Path, coordinate: str, values: list[float], message: str
) -> None:
    path = _write_annual_fixture(tmp_path, 2000)
    with cast(Callable[[Path], xr.Dataset], xr.open_dataset)(path) as source:
        dataset = cast(Callable[[], xr.Dataset], source.load)()
    dataset.coords[coordinate] = ("cell", values)
    cast(Callable[[Path], None], dataset.to_netcdf)(path)
    with pytest.raises(ValueError, match=message):
        _ = load_reef_sst([path])
