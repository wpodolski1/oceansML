import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import replace
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
    Array,
    ModelConfig,
    Preprocessing,
    ReefSequenceDataset,
    ReefSSTData,
    SplitConfig,
    TrainingConfig,
    fit_preprocessing,
    load_checkpoint,
    load_reef_sst,
    save_checkpoint,
    validate_checkpoint_compatibility,
)


@pytest.fixture
def checkpoint(
    tmp_path: Path,
) -> tuple[Path, ReefSSTData, SSTLSTM, Preprocessing, SplitConfig]:
    dates = pd.date_range("2000-01-01", "2003-12-31", freq="D")
    day = np.arange(len(dates), dtype=np.float32)
    values = 24.0 + np.sin(day / 58.0) + day / 3000.0
    dataset = xr.Dataset(
        data_vars={"sst": (("time", "cell"), np.column_stack((values, values + 3.0)))},
        coords={
            "time": dates,
            "cell": np.array([100, 200], dtype=np.int64),
            "lat": ("cell", np.array([-20.0, -10.0], dtype=np.float32)),
            "lon": ("cell", np.array([145.0, 150.0], dtype=np.float32)),
        },
        attrs={
            "sst_product": "OISST",
            "declared_product_version": "v2.1",
            "reef_archive_sha256": "reef-hash",
            "source_sha256": "annual-source-hash",
            "missing_data_semantics": "NaN is unknown",
        },
    )
    dataset["sst"].attrs["units"] = "degC"
    source = tmp_path / "reef_sst_fixture.nc"
    cast(Callable[[Path], None], dataset.to_netcdf)(source)
    data = load_reef_sst([source])
    split = SplitConfig("2001-12-31", "2002-12-31", "2003-12-31")
    preprocessing = fit_preprocessing(data, split.train_end)
    config = ModelConfig(
        context_days=14, horizon=7, hidden_dim=4, num_layers=2, dropout=0.25
    )
    with cast(
        Callable[[list[int]], AbstractContextManager[None]], torch.random.fork_rng
    )([]):
        _ = cast(Callable[[int], torch.Generator], torch.manual_seed)(17)
        model = SSTLSTM(config).eval()
        # Nonzero output weights make this sensitive to recurrent-state reloads.
        with torch.no_grad():
            _ = model.output.weight.copy_(
                torch.arange(28, dtype=torch.float32).reshape(7, 4) / 100.0
            )
            _ = model.output.bias.copy_(torch.arange(7, dtype=torch.float32) / 20.0)
    path = tmp_path / "forecast.pt"
    save_checkpoint(
        path, model, preprocessing, config, TrainingConfig(), split, data, 1, {}
    )
    return path, data, model, preprocessing, split


def test_fresh_cpu_process_reproduces_predictions_and_inverse_scaling(
    checkpoint: tuple[Path, ReefSSTData, SSTLSTM, Preprocessing, SplitConfig],
) -> None:
    path, data, model, preprocessing, split = checkpoint
    dataset = ReefSequenceDataset(data, preprocessing, model.config, split, "test")
    samples = [dataset[index] for index in (0, 1, 120)]
    features = torch.stack([sample[0] for sample in samples])
    baseline = torch.stack([sample[2] for sample in samples])
    with torch.inference_mode():
        expected_residual = cast(Tensor, model(features))
    assert torch.count_nonzero(expected_residual) > 0
    expected_context: list[Array[np.float32]] = []
    for sample in samples:
        target, cell = divmod(int(sample[4]), len(data.cell_ids))
        expected_context.append(
            data.sst[target - model.config.context_days : target, cell]
        )
    script = """
import sys
import numpy as np
import torch
from time_series import (
    ReefSequenceDataset, load_checkpoint, load_reef_sst,
    validate_checkpoint_compatibility,
)
torch.set_num_threads(1)
model, preprocessing, split, payload = load_checkpoint(sys.argv[1], torch.device('cpu'))
assert not model.training
assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
data = load_reef_sst([sys.argv[2]])
validate_checkpoint_compatibility(payload, preprocessing, model.config, split, data)
dataset = ReefSequenceDataset(data, preprocessing, model.config, split, 'test')
samples = [dataset[index] for index in (0, 1, 120)]
features = torch.stack([sample[0] for sample in samples])
baseline = torch.stack([sample[2] for sample in samples])
inverse_context = []
for sample in samples:
    target, cell = divmod(int(sample[4]), len(data.cell_ids))
    days = dataset.day_indices[target - model.config.context_days:target]
    inverse_context.append(
        sample[0][:, 0].numpy() * preprocessing.anomaly_scale
        + preprocessing.climatology[days, cell]
    )
with torch.inference_mode():
    residual = model(features)
    repeated = model(features)
torch.save({
    'features': features,
    'residual': residual,
    'repeated': repeated,
    'forecast': baseline + residual,
    'inverse_context': torch.from_numpy(np.stack(inverse_context)),
}, sys.argv[3])
"""
    result_path = path.with_name("reloaded.pt")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["GITHUB_SHA"] = "a-different-revision"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    _ = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            str(path.with_name("reef_sst_fixture.nc")),
            str(result_path),
        ],
        cwd=path.parent,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    actual = cast(
        dict[str, Tensor],
        torch.load(result_path, map_location="cpu", weights_only=True),
    )
    torch.testing.assert_close(actual["features"], features, rtol=0, atol=0)
    torch.testing.assert_close(actual["residual"], expected_residual)
    torch.testing.assert_close(actual["repeated"], actual["residual"], rtol=0, atol=0)
    torch.testing.assert_close(actual["forecast"], baseline + expected_residual)
    np.testing.assert_allclose(
        actual["inverse_context"].numpy(), np.stack(expected_context), rtol=1e-6
    )


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_and_current_checkpoints_preserve_missing_climatology_and_ignore_git(
    checkpoint: tuple[Path, ReefSSTData, SSTLSTM, Preprocessing, SplitConfig],
    version: int,
) -> None:
    path, data, original_model, _, _ = checkpoint
    payload = cast(dict[str, object], torch.load(path, weights_only=True))
    payload["format_version"] = version
    saved_preprocessing = cast(dict[str, object], payload["preprocessing"])
    cast(Tensor, saved_preprocessing["climatology"])[:, 1] = float("nan")
    payload["git"] = {"commit": "unrelated-revision", "dirty": True}
    if version == 1:
        del payload["data_contract"]
        del payload["git"]
    torch.save(payload, path)
    model, preprocessing, split, loaded = load_checkpoint(path, torch.device("cpu"))
    validate_checkpoint_compatibility(loaded, preprocessing, model.config, split, data)
    assert np.isnan(preprocessing.climatology[:, 1]).all()
    assert np.isfinite(preprocessing.climatology[:, 0]).all()
    assert not model.training
    for name, value in cast(Mapping[str, Tensor], original_model.state_dict()).items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("field", ["cell_ids", "latitude", "longitude", "target_units"])
def test_incompatible_input_spatial_identity_and_units_are_rejected(
    checkpoint: tuple[Path, ReefSSTData, SSTLSTM, Preprocessing, SplitConfig],
    field: str,
) -> None:
    path, data, model, preprocessing, split = checkpoint
    payload = cast(dict[str, object], torch.load(path, weights_only=True))
    value = "K" if field == "target_units" else getattr(data, field) + 1
    incompatible = replace(data, **{field: value})
    with pytest.raises(ValueError, match=field):
        validate_checkpoint_compatibility(
            payload, preprocessing, model.config, split, incompatible
        )


@pytest.mark.parametrize(
    "field",
    [
        "sst_product",
        "declared_product_version",
        "reef_archive_sha256",
        "missing_data_semantics",
    ],
)
def test_incompatible_source_identity_is_rejected(
    checkpoint: tuple[Path, ReefSSTData, SSTLSTM, Preprocessing, SplitConfig],
    field: str,
) -> None:
    path, data, model, preprocessing, split = checkpoint
    payload = cast(dict[str, object], torch.load(path, weights_only=True))
    provenance = [dict(item, **{field: "different"}) for item in data.provenance]
    with pytest.raises(ValueError, match="source_contract"):
        validate_checkpoint_compatibility(
            payload,
            preprocessing,
            model.config,
            split,
            replace(data, provenance=provenance),
        )


def test_source_paths_and_annual_hashes_are_not_product_identity(
    checkpoint: tuple[Path, ReefSSTData, SSTLSTM, Preprocessing, SplitConfig],
) -> None:
    path, data, model, preprocessing, split = checkpoint
    payload = cast(dict[str, object], torch.load(path, weights_only=True))
    provenance = [
        dict(
            item,
            path="relocated.nc",
            source_file="next-year.nc",
            source_sha256="new-hash",
        )
        for item in data.provenance
    ]
    validate_checkpoint_compatibility(
        payload,
        preprocessing,
        model.config,
        split,
        replace(data, provenance=provenance),
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "format_version", True, "format"),
        (None, "format_version", 3, "format"),
        (None, "feature_names", ["wrong"], "feature order"),
        (None, "model_config", [], "model_config"),
        (None, "preprocessing", None, "preprocessing"),
        (None, "data_contract", None, "contract"),
        ("model_config", "context_days", 1.5, "context_days"),
        ("model_config", "horizon", True, "horizon"),
        ("model_config", "hidden_dim", 0, "hidden_dim"),
        ("model_config", "dropout", float("nan"), "dropout"),
        ("split_config", "train_end", "NaT", "train_end"),
        ("split_config", "validation_end", "2000-01-01", "Split dates"),
        ("preprocessing", "anomaly_scale", 0.0, "anomaly_scale"),
        ("preprocessing", "anomaly_scale", float("inf"), "anomaly_scale"),
        ("preprocessing", "latitude_scale", -1.0, "latitude_scale"),
        ("preprocessing", "latitude_mean", float("nan"), "latitude_mean"),
        ("preprocessing", "latitude_scale", True, "latitude_scale"),
        ("preprocessing", "climatology", [], "climatology"),
        ("preprocessing", "climatology", torch.zeros(365, 2), "climatology"),
        ("preprocessing", "climatology", torch.zeros(366, 0), "climatology"),
        (
            "preprocessing",
            "climatology",
            torch.zeros(366, 2, dtype=torch.int64),
            "climatology",
        ),
        (
            "preprocessing",
            "climatology",
            torch.full((366, 2), float("inf")),
            "climatology",
        ),
        (
            "preprocessing",
            "climatology",
            torch.zeros(366, 2, dtype=torch.complex64),
            "climatology",
        ),
        ("data_contract", "feature_names", ["wrong"], "feature_names"),
        ("data_contract", "target_units", "K", "target_units"),
        ("data_contract", "context_days", 15, "context_days"),
        ("data_contract", "forecast_horizon_days", 8, "forecast_horizon_days"),
        (
            "data_contract",
            "preprocessing_fit_end",
            "2000-12-31",
            "preprocessing_fit_end",
        ),
        ("data_contract", "cell_ids", torch.tensor([100.0, 200.0]), "cell_ids"),
        ("data_contract", "cell_ids", torch.tensor([100, 100]), "cell_ids"),
        ("data_contract", "latitude", torch.zeros(2, 1), "latitude"),
        (
            "data_contract",
            "longitude",
            torch.tensor([145.0, float("nan")]),
            "longitude",
        ),
        ("data_contract", "training_period", {}, "training_period"),
        (
            "data_contract",
            "training_period",
            {"start": "NaT", "end": "2001-12-31"},
            "training_period",
        ),
        ("data_contract", "source_contract", ["invalid"], "source_contract"),
        ("state_dict", "output.bias", torch.full((7,), float("nan")), "output.bias"),
        ("state_dict", "output.bias", torch.zeros(7, dtype=torch.int64), "output.bias"),
    ],
)
def test_malformed_checkpoint_metadata_is_rejected_at_load(
    checkpoint: tuple[Path, ReefSSTData, SSTLSTM, Preprocessing, SplitConfig],
    section: str | None,
    key: str,
    value: object,
    message: str,
) -> None:
    path = checkpoint[0]
    payload = cast(dict[str, object], torch.load(path, weights_only=True))
    destination = (
        payload if section is None else cast(dict[str, object], payload[section])
    )
    destination[key] = value
    torch.save(payload, path)
    with pytest.raises((TypeError, ValueError), match=message):
        _ = load_checkpoint(path, torch.device("cpu"))
