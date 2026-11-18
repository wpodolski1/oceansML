from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, TypedDict, TypeVar, cast

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from typing_extensions import override

LOGGER = logging.getLogger(__name__)

# NumPy's NDArray alias leaves shape entries as Any; these arrays have integer
# dimensions. Boundary casts below describe documented third-party return types.
_Scalar = TypeVar("_Scalar", bound=np.generic)
Array = np.ndarray[tuple[int, ...], np.dtype[_Scalar]]
Batch = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
_from_numpy = cast(Callable[[Array[np.generic]], Tensor], torch.from_numpy)
FEATURE_NAMES = (
    "sst_anomaly_z",
    "day_sin",
    "day_cos",
    "latitude_z",
    "longitude_sin",
    "longitude_cos",
)
SplitName = Literal["train", "validation", "test"]
CHECKPOINT_FORMAT_VERSION = 2
TARGET_NAME = "sst"
TARGET_UNITS = "degC"
SPATIAL_IDENTITY_CONVENTION = (
    "OISST 0.25-degree canonical cell ID: lat_index * 1440 + lon_index; "
    "centers start at lat=-89.875, lon=-179.875; longitude in [-180,180)"
)
MISSING_DATA_CONVENTION = (
    "Non-finite observed SST is unknown; any context or target window "
    "containing an unknown value is ineligible."
)


_LEAP_CALENDAR = pd.date_range("2000-01-01", "2000-12-31")
_CALENDAR_CODES = np.asarray(
    _LEAP_CALENDAR.month * 100 + _LEAP_CALENDAR.day, dtype=np.int16
)
_CODE_TO_INDEX = {
    int(code): index
    for index, code in enumerate(cast(Iterable[np.int16], _CALENDAR_CODES))
}


@dataclass(frozen=True)
class SplitConfig:
    train_end: str
    validation_end: str
    test_end: str

    def validate(self, dates: pd.DatetimeIndex) -> None:
        train_end = pd.Timestamp(self.train_end)
        validation_end = pd.Timestamp(self.validation_end)
        test_end = pd.Timestamp(self.test_end)
        if not train_end < validation_end < test_end:
            raise ValueError(
                "Split dates must satisfy train_end < validation_end < test_end"
            )
        if train_end < dates[0] or test_end > dates[-1]:
            raise ValueError(
                f"Split range {train_end.date()}–{test_end.date()} lies outside "
                + f"the data range {dates[0].date()}–{dates[-1].date()}"
            )


@dataclass(frozen=True)
class ModelConfig:
    context_days: int = 120
    horizon: int = 28
    input_dim: int = len(FEATURE_NAMES)
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0

    def validate(self) -> None:
        for name in (
            "context_days",
            "horizon",
            "input_dim",
            "hidden_dim",
            "num_layers",
        ):
            if type(cast(object, getattr(self, name))) is not int:
                raise TypeError(f"{name} must be an integer")
        dropout = cast(object, self.dropout)
        if isinstance(dropout, bool) or not isinstance(dropout, int | float):
            raise TypeError("dropout must be numeric")
        if self.context_days < 1 or self.horizon < 1:
            raise ValueError("context_days and horizon must be positive")
        if self.input_dim != len(FEATURE_NAMES):
            raise ValueError(
                f"input_dim must be {len(FEATURE_NAMES)} for {FEATURE_NAMES}"
            )
        if self.hidden_dim < 1 or self.num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = 42
    num_workers: int = 0
    samples_per_epoch: int | None = None

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative"
            )
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive when supplied")


@dataclass
class ReefSSTData:
    dates: pd.DatetimeIndex
    cell_ids: Array[np.int64]
    latitude: Array[np.float32]
    longitude: Array[np.float32]
    sst: Array[np.float32]
    finite_cumsum: Array[np.int32 | np.int64]
    provenance: list[dict[str, object]]
    target_units: str


def _checkpoint_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Checkpoint {key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Checkpoint {key} must be finite")
    if key.endswith("_scale") and result <= 0:
        raise ValueError(f"Checkpoint {key} must be positive")
    return result


@dataclass
class Preprocessing:
    climatology: Array[np.float32]
    anomaly_scale: float
    latitude_mean: float
    latitude_scale: float

    def to_checkpoint(self) -> dict[str, object]:
        return {
            "climatology": _from_numpy(self.climatology.copy()),
            "anomaly_scale": self.anomaly_scale,
            "latitude_mean": self.latitude_mean,
            "latitude_scale": self.latitude_scale,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, object]) -> Preprocessing:
        climatology = _checkpoint_array(payload, "climatology")
        if (
            climatology.ndim != 2
            or climatology.shape[0] != len(_CALENDAR_CODES)
            or climatology.shape[1] == 0
        ):
            raise ValueError("Checkpoint climatology must have shape (366, cells)")
        if (
            climatology.dtype.kind != "f"
            or np.isinf(climatology).any()
            or (np.abs(climatology) > np.finfo(np.float32).max).any()
        ):
            raise ValueError(
                "Checkpoint climatology must contain float32 values or NaN"
            )
        return cls(
            climatology=climatology.astype(np.float32),
            anomaly_scale=_checkpoint_float(payload, "anomaly_scale"),
            latitude_mean=_checkpoint_float(payload, "latitude_mean"),
            latitude_scale=_checkpoint_float(payload, "latitude_scale"),
        )


def resolve_data_files(inputs: Sequence[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for raw_input in inputs:
        value = str(raw_input)
        matches = [Path(match) for match in glob.glob(value)]
        if not matches:
            matches = [Path(value)]
        for match in matches:
            if match.is_dir():
                files.update(match.glob("reef_sst_*.nc"))
            elif match.is_file():
                files.add(match)
            else:
                raise FileNotFoundError(f"Reef SST input does not exist: {match}")
    if not files:
        raise ValueError("No reef_sst_*.nc inputs were found")
    return sorted(files)


def _validate_daily_dates(dates: pd.DatetimeIndex, path: Path) -> None:
    if dates.empty:
        raise ValueError(f"{path} has no time observations")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f"{path} time coordinates must be unique and sorted")
    if any(timestamp != timestamp.normalize() for timestamp in dates):
        raise ValueError(f"{path} contains non-midnight timestamps")


def load_reef_sst(inputs: Sequence[str | Path]) -> ReefSSTData:
    files = resolve_data_files(inputs)
    date_blocks: list[pd.DatetimeIndex] = []
    sst_blocks: list[Array[np.float32]] = []
    provenance: list[dict[str, object]] = []
    reference_cell: Array[np.int64] | None = None
    reference_latitude: Array[np.float32] | None = None
    reference_longitude: Array[np.float32] | None = None
    reference_units: str | None = None

    for path in files:
        with cast(Callable[[Path], xr.Dataset], xr.open_dataset)(path) as ds:
            required = {"sst", "time", "cell", "lat", "lon"}
            missing = required.difference(ds.variables)
            if missing:
                raise ValueError(f"{path} is missing required variables: {missing}")
            if ds["sst"].dims != ("time", "cell"):
                raise ValueError(
                    f"{path} sst dimensions must be ('time', 'cell'), "
                    + f"not {ds['sst'].dims}"
                )
            units = str(cast(Mapping[str, object], ds["sst"].attrs).get("units", ""))
            if units != TARGET_UNITS:
                raise ValueError(
                    f"{path} sst units must be {TARGET_UNITS!r}, not {units!r}"
                )

            dates = pd.DatetimeIndex(ds["time"].values)
            if any(ds[name].dims != ("cell",) for name in ("cell", "lat", "lon")):
                raise ValueError(
                    f"{path} cell IDs and coordinates must use the cell axis"
                )
            raw_cell = np.asarray(ds["cell"].values)
            if (
                raw_cell.dtype.kind not in "iu"
                or raw_cell.size == 0
                or np.unique(raw_cell).size != raw_cell.size
            ):
                raise ValueError(f"{path} cell IDs must be nonempty unique integers")
            _validate_daily_dates(dates, path)
            cell = np.asarray(ds["cell"].values, dtype=np.int64)
            latitude = np.asarray(ds["lat"].values, dtype=np.float32)
            longitude = np.asarray(ds["lon"].values, dtype=np.float32)
            values = np.asarray(ds["sst"].values, dtype=np.float32)
            if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
                raise ValueError(f"{path} coordinates must be finite")
            values[~np.isfinite(values)] = np.nan

            if reference_cell is None:
                reference_cell = cell.copy()
                reference_latitude = latitude.copy()
                reference_longitude = longitude.copy()
                reference_units = units
            else:
                assert reference_latitude is not None
                assert reference_longitude is not None
                if (
                    not np.array_equal(cell, reference_cell)
                    or not np.array_equal(latitude, reference_latitude)
                    or not np.array_equal(longitude, reference_longitude)
                    or units != reference_units
                ):
                    raise ValueError(
                        f"{path} cell IDs or coordinates differ from earlier inputs"
                    )

            date_blocks.append(dates)
            sst_blocks.append(values)
            attributes = cast(Mapping[str, object], ds.attrs)
            provenance.append(
                {
                    "path": str(path),
                    "source_file": str(attributes.get("source_file", "")),
                    "source_sha256": str(attributes.get("source_sha256", "")),
                    "reef_archive_sha256": str(
                        attributes.get("reef_archive_sha256", "")
                    ),
                    "sst_product": str(attributes.get("sst_product", "")),
                    "declared_product_version": str(
                        attributes.get("declared_product_version", "")
                    ),
                    "product_version_evidence": str(
                        attributes.get("product_version_evidence", "")
                    ),
                    "missing_data_semantics": str(
                        attributes.get("missing_data_semantics", "")
                    ),
                }
            )

    if (
        reference_cell is None
        or reference_latitude is None
        or reference_longitude is None
        or reference_units is None
    ):
        raise ValueError("No reef SST data were loaded")

    dates = pd.DatetimeIndex(date_blocks[0].append(date_blocks[1:]))
    values = np.concatenate(sst_blocks, axis=0)
    order = np.argsort(dates.values)
    dates = pd.DatetimeIndex(dates[order])
    values = values[order]
    if dates.has_duplicates:
        duplicates = dates[dates.duplicated()].unique()
        raise ValueError(f"Duplicate input dates: {duplicates[:5].tolist()}")
    expected = pd.DatetimeIndex(pd.date_range(dates[0], dates[-1], freq="D"))
    if not dates.equals(expected):
        missing = expected.difference(dates)
        raise ValueError(
            "Input partitions are not consecutive daily data; "
            + f"first missing dates: {missing[:5].tolist()}"
        )

    finite = np.isfinite(values).astype(np.int32)
    finite_cumsum = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=np.int32), finite.cumsum(axis=0)]
    )
    return ReefSSTData(
        dates=dates,
        cell_ids=reference_cell,
        latitude=reference_latitude,
        longitude=reference_longitude,
        sst=values,
        finite_cumsum=finite_cumsum,
        provenance=provenance,
        target_units=reference_units,
    )


def _calendar_day_indices(dates: pd.DatetimeIndex) -> Array[np.int16]:
    codes = np.asarray(dates.month * 100 + dates.day, dtype=np.int16)
    try:
        return np.asarray(
            [_CODE_TO_INDEX[int(code)] for code in cast(Iterable[np.int16], codes)],
            dtype=np.int16,
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported calendar date code: {exc.args[0]}") from exc


def fit_preprocessing(data: ReefSSTData, train_end: str) -> Preprocessing:
    train_mask = data.dates <= pd.Timestamp(train_end)
    if not train_mask.any():
        raise ValueError("Training split contains no dates")
    train_values = data.sst[train_mask]
    train_day_indices = _calendar_day_indices(data.dates[train_mask])
    climatology = np.full(
        (len(_CALENDAR_CODES), data.sst.shape[1]), np.nan, dtype=np.float32
    )
    with np.errstate(invalid="ignore"):
        for day_index in cast(Iterable[np.int16], np.unique(train_day_indices)):
            climatology[day_index] = np.nanmean(
                train_values[train_day_indices == day_index], axis=0
            )

    february_29 = _CODE_TO_INDEX[229]
    missing_february_29 = ~np.isfinite(
        cast(Array[np.float32], climatology[february_29])
    )
    climatology[february_29, missing_february_29] = np.nanmean(
        np.stack(
            [
                climatology[_CODE_TO_INDEX[228], missing_february_29],
                climatology[_CODE_TO_INDEX[301], missing_february_29],
            ]
        ),
        axis=0,
    )

    anomalies = train_values - climatology[train_day_indices]
    anomaly_scale = float(np.nanstd(anomalies))
    if not math.isfinite(anomaly_scale) or anomaly_scale <= 0:
        raise ValueError("Training SST anomalies have no finite variation")
    latitude_mean = float(np.mean(data.latitude))
    latitude_scale = float(np.std(data.latitude))
    if not math.isfinite(latitude_scale) or latitude_scale <= 0:
        latitude_scale = 1.0
    return Preprocessing(
        climatology=climatology,
        anomaly_scale=anomaly_scale,
        latitude_mean=latitude_mean,
        latitude_scale=latitude_scale,
    )


def _split_bounds(
    split: SplitName, config: SplitConfig, dates: pd.DatetimeIndex
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split == "train":
        return dates[0], pd.Timestamp(config.train_end)
    if split == "validation":
        return (
            pd.Timestamp(config.train_end) + pd.Timedelta(days=1),
            pd.Timestamp(config.validation_end),
        )
    return (
        pd.Timestamp(config.validation_end) + pd.Timedelta(days=1),
        pd.Timestamp(config.test_end),
    )


class ReefSequenceDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        data: ReefSSTData,
        preprocessing: Preprocessing,
        model_config: ModelConfig,
        split_config: SplitConfig,
        split: SplitName,
    ) -> None:
        model_config.validate()
        split_config.validate(data.dates)
        self.data: ReefSSTData = data
        self.preprocessing: Preprocessing = preprocessing
        self.model_config: ModelConfig = model_config
        self.split: SplitName = split
        self.day_indices: Array[np.int16] = _calendar_day_indices(data.dates)
        self.encoded_examples: Array[np.int64] = self._eligible_examples(split_config)
        if len(self.encoded_examples) == 0:
            raise ValueError(f"No eligible {split} examples")

    def _eligible_examples(self, split_config: SplitConfig) -> Array[np.int64]:
        context = self.model_config.context_days
        horizon = self.model_config.horizon
        split_start, split_end = _split_bounds(
            self.split, split_config, self.data.dates
        )
        first_target = max(
            context, int(self.data.dates.searchsorted(split_start, side="left"))
        )
        last_target = (
            int(self.data.dates.searchsorted(split_end, side="right")) - horizon
        )
        if last_target < first_target:
            return np.empty(0, dtype=np.int64)

        target_starts = np.arange(first_target, last_target + 1, dtype=np.int64)
        window_starts = target_starts - context
        window_ends = target_starts + horizon
        finite_counts = cast(
            Array[np.int32 | np.int64],
            self.data.finite_cumsum[window_ends]
            - self.data.finite_cumsum[window_starts],
        )
        valid = cast(Array[np.bool_], finite_counts == context + horizon)
        flat_indices = np.flatnonzero(valid.ravel())
        target_offsets, cell_indices = np.divmod(flat_indices, self.data.sst.shape[1])
        return cast(
            Array[np.int64],
            (
                target_starts[target_offsets] * self.data.sst.shape[1] + cell_indices
            ).astype(np.int64),
        )

    def __len__(self) -> int:
        return len(self.encoded_examples)

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        encoded = int(cast(np.int64, self.encoded_examples[index]))
        target_start, cell_index = divmod(encoded, self.data.sst.shape[1])
        context_start = target_start - self.model_config.context_days
        target_end = target_start + self.model_config.horizon

        context_values = self.data.sst[context_start:target_start, cell_index]
        context_days = self.day_indices[context_start:target_start]
        context_climatology = cast(
            Array[np.float32], self.preprocessing.climatology[context_days, cell_index]
        )
        anomalies = context_values - context_climatology
        context_dates = self.data.dates[context_start:target_start]
        angles = (
            2.0
            * np.pi
            * (np.asarray(context_dates.day_of_year, dtype=np.float32) - 1.0)
            / 365.2425
        )
        longitude_radians = math.radians(
            float(cast(np.float32, self.data.longitude[cell_index]))
        )
        latitude_z = (
            float(cast(np.float32, self.data.latitude[cell_index]))
            - self.preprocessing.latitude_mean
        ) / self.preprocessing.latitude_scale

        features = np.empty(
            (self.model_config.context_days, len(FEATURE_NAMES)),
            dtype=np.float32,
        )
        features[:, 0] = anomalies / self.preprocessing.anomaly_scale
        features[:, 1] = np.sin(angles)
        features[:, 2] = np.cos(angles)
        features[:, 3] = latitude_z
        features[:, 4] = math.sin(longitude_radians)
        features[:, 5] = math.cos(longitude_radians)

        target_values = self.data.sst[target_start:target_end, cell_index]
        target_days = self.day_indices[target_start:target_end]
        target_climatology = cast(
            Array[np.float32], self.preprocessing.climatology[target_days, cell_index]
        )
        current_anomaly = float(cast(np.float32, anomalies[-1]))
        baseline = target_climatology + current_anomaly
        target_residual = target_values - baseline
        return (
            _from_numpy(features),
            _from_numpy(target_residual.astype(np.float32)),
            _from_numpy(baseline.astype(np.float32)),
            _from_numpy(target_values.astype(np.float32)),
            torch.tensor(encoded, dtype=torch.int64),
        )


class SSTLSTM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config: ModelConfig = config
        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output: nn.Linear = nn.Linear(config.hidden_dim, config.horizon)
        _ = nn.init.zeros_(self.output.weight)
        _ = nn.init.zeros_(self.output.bias)

    @override
    def forward(self, features: Tensor) -> Tensor:
        output, _ = cast(tuple[Tensor, tuple[Tensor, Tensor]], self.lstm(features))
        return cast(Tensor, self.output(output[:, -1, :]))


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this Python process"
        )
    return torch.device(requested)


def make_data_loader(
    dataset: ReefSequenceDataset,
    training_config: TrainingConfig,
    device: torch.device,
    *,
    training: bool,
) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    sampler = None
    shuffle = training
    if training and training_config.samples_per_epoch is not None:
        generator = torch.Generator().manual_seed(training_config.seed)
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=training_config.samples_per_epoch,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=training_config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=training_config.num_workers > 0,
    )


def _empty_metric_accumulator(horizon: int) -> dict[str, Tensor]:
    return {
        "absolute": torch.zeros(horizon, dtype=torch.float64),
        "squared": torch.zeros(horizon, dtype=torch.float64),
        "signed": torch.zeros(horizon, dtype=torch.float64),
        "count": torch.zeros(horizon, dtype=torch.int64),
    }


def _update_metrics(accumulator: dict[str, Tensor], error: Tensor) -> None:
    finite = torch.isfinite(error)
    safe_error = torch.where(finite, error, torch.zeros_like(error))
    accumulator["absolute"] += safe_error.abs().sum(dim=0).double().cpu()
    accumulator["squared"] += safe_error.square().sum(dim=0).double().cpu()
    accumulator["signed"] += safe_error.sum(dim=0).double().cpu()
    accumulator["count"] += finite.sum(dim=0).cpu()


def _finalize_metrics(
    accumulator: dict[str, Tensor],
) -> dict[str, list[float] | list[int]]:
    count = accumulator["count"].clamp_min(1).double()
    return {
        "mae": cast(
            Callable[[], list[float]], (accumulator["absolute"] / count).tolist
        )(),
        "rmse": cast(
            Callable[[], list[float]], torch.sqrt(accumulator["squared"] / count).tolist
        )(),
        "bias": cast(
            Callable[[], list[float]], (accumulator["signed"] / count).tolist
        )(),
        "count": cast(Callable[[], list[int]], accumulator["count"].tolist)(),
    }


def evaluate_model(
    model: SSTLSTM,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    device: torch.device,
) -> dict[str, dict[str, list[float] | list[int]]]:
    _ = model.eval()
    horizon = model.config.horizon
    model_metrics = _empty_metric_accumulator(horizon)
    baseline_metrics = _empty_metric_accumulator(horizon)
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for features, _, baseline, actual, _ in cast(Iterable[Batch], loader):
            features = features.to(device, non_blocking=True)
            baseline_device = baseline.to(device, non_blocking=True)
            actual_device = actual.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                forecast = baseline_device + cast(Tensor, model(features))
            _update_metrics(model_metrics, forecast.float() - actual_device)
            _update_metrics(baseline_metrics, baseline_device - actual_device)
    return {
        "model": _finalize_metrics(model_metrics),
        "anomaly_persistence": _finalize_metrics(baseline_metrics),
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    state = cast(Mapping[str, Tensor], model.state_dict())
    return {name: value.detach().cpu() for name, value in state.items()}


def _git_metadata() -> dict[str, object]:
    commit = os.environ.get("GITHUB_SHA")
    dirty: bool | None = None
    try:
        if commit is None:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if revision.returncode == 0:
                commit = revision.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode == 0:
            dirty = bool(status.stdout)
    except OSError:
        pass
    return {"commit": commit, "dirty": dirty}


def _source_contract(provenance: list[dict[str, object]]) -> list[dict[str, str]]:
    identities = {
        (
            str(item.get("sst_product", "")),
            str(item.get("declared_product_version", "")),
            str(item.get("reef_archive_sha256", "")),
            str(item.get("missing_data_semantics", "")),
        )
        for item in provenance
    }
    return [
        {
            "sst_product": product,
            "declared_product_version": version,
            "reef_archive_sha256": reef_hash,
            "missing_data_semantics": missing_data,
        }
        for product, version, reef_hash, missing_data in sorted(identities)
    ]


def _data_contract(
    data: ReefSSTData,
    model_config: ModelConfig,
    split_config: SplitConfig,
) -> dict[str, object]:
    return {
        "contract_version": 1,
        "target_name": TARGET_NAME,
        "target_units": data.target_units,
        "feature_names": list(FEATURE_NAMES),
        "context_days": model_config.context_days,
        "forecast_horizon_days": model_config.horizon,
        "preprocessing_fit_end": split_config.train_end,
        "training_period": {
            "start": str(data.dates[0].date()),
            "end": split_config.train_end,
        },
        "spatial_identity_convention": SPATIAL_IDENTITY_CONVENTION,
        "cell_ids": _from_numpy(data.cell_ids),
        "latitude": _from_numpy(data.latitude),
        "longitude": _from_numpy(data.longitude),
        "missing_data_convention": MISSING_DATA_CONVENTION,
        "source_contract": _source_contract(data.provenance),
    }


def _checkpoint_array(payload: dict[str, object], key: str) -> Array[np.generic]:
    value = payload.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Checkpoint {key} must be a tensor")
    if value.layout != torch.strided or value.dtype not in {
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    }:
        raise ValueError(f"Checkpoint {key} has an unsupported tensor dtype or layout")
    return cast(Array[np.generic], value.detach().cpu().numpy())


def _validate_data_contract(
    payload: dict[str, object],
    preprocessing: Preprocessing,
    model_config: ModelConfig,
    split_config: SplitConfig,
) -> dict[str, object]:
    contract = _checkpoint_mapping(payload.get("data_contract"), "data_contract")
    expected_values = {
        "contract_version": 1,
        "target_name": TARGET_NAME,
        "target_units": TARGET_UNITS,
        "feature_names": list(FEATURE_NAMES),
        "context_days": model_config.context_days,
        "forecast_horizon_days": model_config.horizon,
        "preprocessing_fit_end": split_config.train_end,
        "spatial_identity_convention": SPATIAL_IDENTITY_CONVENTION,
        "missing_data_convention": MISSING_DATA_CONVENTION,
    }
    features = contract.get("feature_names")
    if not isinstance(features, list) or not all(
        isinstance(name, str) for name in cast(list[object], features)
    ):
        raise ValueError("Checkpoint data contract mismatch for feature_names")
    for key, expected in expected_values.items():
        actual = contract.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Checkpoint data contract mismatch for {key}")
    cell_count = preprocessing.climatology.shape[1]
    for key in ("cell_ids", "latitude", "longitude"):
        values = _checkpoint_array(contract, key)
        expected_kind = "i" if key == "cell_ids" else "f"
        if (
            values.shape != (cell_count,)
            or values.dtype.kind != expected_kind
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Checkpoint data contract has invalid {key}")
        if key == "cell_ids" and len(np.unique(values)) != cell_count:
            raise ValueError("Checkpoint data contract has duplicate cell_ids")
    period = _checkpoint_mapping(contract.get("training_period"), "training_period")
    if period.get("end") != split_config.train_end:
        raise ValueError("Checkpoint data contract has invalid training_period")
    start = period.get("start")
    if not isinstance(start, str):
        raise TypeError("Checkpoint training_period start must be a date string")
    try:
        start_date = pd.Timestamp(start)
        if (
            pd.isna(start_date)
            or start_date.tzinfo is not None
            or start_date != start_date.normalize()
            or start_date > pd.Timestamp(split_config.train_end)
        ):
            raise ValueError("Invalid training start")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Checkpoint data contract has invalid training_period"
        ) from exc
    sources = contract.get("source_contract")
    source_keys = {
        "sst_product",
        "declared_product_version",
        "reef_archive_sha256",
        "missing_data_semantics",
    }
    if not isinstance(sources, list):
        raise TypeError("Checkpoint data contract has invalid source_contract")
    for source in cast(list[object], sources):
        item = _checkpoint_mapping(source, "source_contract")
        if set(item) != source_keys or any(
            not isinstance(value, str) for value in item.values()
        ):
            raise ValueError("Checkpoint data contract has invalid source_contract")
    return contract


def validate_checkpoint_compatibility(
    payload: dict[str, object],
    preprocessing: Preprocessing,
    model_config: ModelConfig,
    split_config: SplitConfig,
    data: ReefSSTData,
) -> None:
    expected_climatology_shape = (len(_CALENDAR_CODES), len(data.cell_ids))
    if preprocessing.climatology.shape != expected_climatology_shape:
        raise ValueError(
            "Checkpoint preprocessing does not match the input cell count: "
            + f"{preprocessing.climatology.shape} != {expected_climatology_shape}"
        )

    if payload.get("format_version") == 1:
        LOGGER.warning(
            "Legacy checkpoint format 1 has no spatial compatibility metadata; "
            + "validated preprocessing dimensions only"
        )
        return

    contract = _validate_data_contract(
        payload, preprocessing, model_config, split_config
    )
    expected_values = {
        "target_units": data.target_units,
        "source_contract": _source_contract(data.provenance),
    }
    for key, expected in expected_values.items():
        if contract.get(key) != expected:
            raise ValueError(f"Checkpoint data contract mismatch for {key}")

    for key, expected in (
        ("cell_ids", data.cell_ids),
        ("latitude", data.latitude),
        ("longitude", data.longitude),
    ):
        if not np.array_equal(_checkpoint_array(contract, key), expected):
            raise ValueError(f"Checkpoint data contract mismatch for {key}")


def save_checkpoint(
    path: Path,
    model: SSTLSTM,
    preprocessing: Preprocessing,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    split_config: SplitConfig,
    data: ReefSSTData,
    epoch: int,
    validation_metrics: dict[str, dict[str, list[float] | list[int]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "state_dict": _cpu_state_dict(model),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "split_config": asdict(split_config),
            "feature_names": list(FEATURE_NAMES),
            "preprocessing": preprocessing.to_checkpoint(),
            "provenance": data.provenance,
            "data_contract": _data_contract(data, model_config, split_config),
            "git": _git_metadata(),
            "best_epoch": epoch,
            "validation_metrics": validation_metrics,
        },
        path,
    )


class _ModelParameters(TypedDict, total=False):
    context_days: int
    horizon: int
    input_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float


class _SplitParameters(TypedDict):
    train_end: str
    validation_end: str
    test_end: str


def _checkpoint_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint {name} must be a mapping")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError(f"Checkpoint {name} must have string keys")
    return cast(dict[str, object], mapping)


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[SSTLSTM, Preprocessing, SplitConfig, dict[str, object]]:
    raw = cast(object, torch.load(path, map_location=device, weights_only=True))
    if not isinstance(raw, dict):
        raise TypeError("Unsupported forecast checkpoint format")
    payload = _checkpoint_mapping(cast(object, raw), "payload")
    if type(payload.get("format_version")) is not int or payload.get(
        "format_version"
    ) not in {1, CHECKPOINT_FORMAT_VERSION}:
        raise ValueError("Unsupported forecast checkpoint format")
    features = payload.get("feature_names")
    if (
        not isinstance(features, list)
        or not all(isinstance(name, str) for name in cast(list[object], features))
        or features != list(FEATURE_NAMES)
    ):
        raise ValueError("Checkpoint feature order does not match this code")
    model_values = _checkpoint_mapping(payload.get("model_config"), "model_config")
    # Constructor rejects extra keys; validate() checks every value before use.
    model_config = ModelConfig(**cast(_ModelParameters, cast(object, model_values)))
    model_config.validate()
    split_values = _checkpoint_mapping(payload.get("split_config"), "split_config")
    split_dates: list[pd.Timestamp] = []
    for name in ("train_end", "validation_end", "test_end"):
        value = split_values.get(name)
        if not isinstance(value, str):
            raise TypeError(f"Checkpoint {name} must be a date string")
        date = pd.Timestamp(value)
        if pd.isna(date) or date.tzinfo is not None or date != date.normalize():
            raise ValueError(f"Checkpoint {name} must be a timezone-naive daily date")
        split_dates.append(date)
    split_config = SplitConfig(**cast(_SplitParameters, cast(object, split_values)))
    split_config.validate(pd.DatetimeIndex(split_dates))
    preprocessing = Preprocessing.from_checkpoint(
        _checkpoint_mapping(payload.get("preprocessing"), "preprocessing")
    )
    if payload["format_version"] == CHECKPOINT_FORMAT_VERSION:
        _ = _validate_data_contract(payload, preprocessing, model_config, split_config)
    state_dict = _checkpoint_mapping(payload.get("state_dict"), "state_dict")
    for name, value in state_dict.items():
        if (
            not isinstance(value, Tensor)
            or value.layout != torch.strided
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Checkpoint state_dict has invalid tensor {name}")
    model = SSTLSTM(model_config).to(device)
    _ = model.load_state_dict(cast(dict[str, Tensor], state_dict))
    _ = model.eval()
    return model, preprocessing, split_config, payload


def train_forecaster(
    data: ReefSSTData,
    split_config: SplitConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    model_config.validate()
    training_config.validate()
    split_config.validate(data.dates)
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    _ = cast(Callable[[int], torch.Generator], torch.manual_seed)(training_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_config.seed)

    preprocessing = fit_preprocessing(data, split_config.train_end)
    train_dataset = ReefSequenceDataset(
        data, preprocessing, model_config, split_config, "train"
    )
    validation_dataset = ReefSequenceDataset(
        data, preprocessing, model_config, split_config, "validation"
    )
    train_loader = make_data_loader(
        train_dataset, training_config, device, training=True
    )
    validation_loader = make_data_loader(
        validation_dataset, training_config, device, training=False
    )

    model = SSTLSTM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_validation_rmse = math.inf
    stale_epochs = 0
    best_epoch = 0
    best_validation_metrics: dict[str, dict[str, list[float] | list[int]]] | None = None

    for epoch in range(1, training_config.epochs + 1):
        _ = model.train()
        loss_sum = 0.0
        example_count = 0
        for features, target_residual, _, _, _ in cast(Iterable[Batch], train_loader):
            features = features.to(device, non_blocking=True)
            target_residual = target_residual.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predicted_residual = cast(Tensor, model(features))
                loss = nn.functional.mse_loss(predicted_residual, target_residual)
            cast(Callable[[], None], scaler.scale(loss).backward)()
            _ = scaler.step(optimizer)
            scaler.update()
            batch_size = features.shape[0]
            loss_sum += float(loss.detach()) * batch_size
            example_count += batch_size

        validation_metrics = evaluate_model(model, validation_loader, device)
        validation_rmse = float(np.mean(validation_metrics["model"]["rmse"]))
        cast(Callable[[float], None], scheduler.step)(validation_rmse)
        LOGGER.info(
            "epoch=%d train_mse=%.6f validation_rmse=%.6f",
            epoch,
            loss_sum / example_count,
            validation_rmse,
        )
        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
            stale_epochs = 0
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            save_checkpoint(
                checkpoint_path,
                model,
                preprocessing,
                model_config,
                training_config,
                split_config,
                data,
                epoch,
                validation_metrics,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break

    if best_validation_metrics is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    return {
        "best_epoch": best_epoch,
        "example_counts": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
        },
        "validation_metrics": best_validation_metrics,
    }


def evaluate_checkpoint(
    data_inputs: Sequence[str | Path],
    checkpoint_path: str | Path,
    split: Literal["validation", "test"],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, dict[str, list[float] | list[int]]]:
    model, preprocessing, split_config, payload = load_checkpoint(
        checkpoint_path, device
    )
    data = load_reef_sst(data_inputs)
    validate_checkpoint_compatibility(
        payload, preprocessing, model.config, split_config, data
    )
    dataset = ReefSequenceDataset(
        data, preprocessing, model.config, split_config, split
    )
    loader_config = TrainingConfig(
        epochs=1,
        batch_size=batch_size,
        patience=1,
        num_workers=num_workers,
    )
    loader = make_data_loader(dataset, loader_config, device, training=False)
    return evaluate_model(model, loader, device)


class _TrainingArgs(Protocol):
    data: list[str]
    train_end: str
    validation_end: str
    test_end: str
    checkpoint: Path
    context_days: int
    horizon: int
    hidden_dim: int
    num_layers: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    seed: int
    num_workers: int
    samples_per_epoch: int | None
    device: str


def _parse_args() -> _TrainingArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Train a location-safe GBR SST LSTM that predicts 28 direct daily "
            "residual corrections to anomaly persistence."
        )
    )
    _ = parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Annual reef_sst_*.nc files, glob patterns, or directories",
    )
    _ = parser.add_argument("--train-end", required=True)
    _ = parser.add_argument("--validation-end", required=True)
    _ = parser.add_argument("--test-end", required=True)
    _ = parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/gbr_sst_lstm.pt"),
    )
    _ = parser.add_argument("--context-days", type=int, default=120)
    _ = parser.add_argument("--horizon", type=int, default=28)
    _ = parser.add_argument("--hidden-dim", type=int, default=64)
    _ = parser.add_argument("--num-layers", type=int, default=1)
    _ = parser.add_argument("--dropout", type=float, default=0.0)
    _ = parser.add_argument("--epochs", type=int, default=50)
    _ = parser.add_argument("--batch-size", type=int, default=1024)
    _ = parser.add_argument("--learning-rate", type=float, default=1e-3)
    _ = parser.add_argument("--weight-decay", type=float, default=1e-4)
    _ = parser.add_argument("--patience", type=int, default=8)
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--num-workers", type=int, default=0)
    _ = parser.add_argument("--samples-per-epoch", type=int)
    _ = parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return cast(_TrainingArgs, cast(object, parser.parse_args()))


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = load_reef_sst(args.data)
    split_config = SplitConfig(
        train_end=args.train_end,
        validation_end=args.validation_end,
        test_end=args.test_end,
    )
    model_config = ModelConfig(
        context_days=args.context_days,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        samples_per_epoch=args.samples_per_epoch,
    )
    device = choose_device(args.device)
    LOGGER.info(
        "device=%s dates=%s..%s cells=%d",
        device,
        data.dates[0].date(),
        data.dates[-1].date(),
        len(data.cell_ids),
    )
    result = train_forecaster(
        data,
        split_config,
        model_config,
        training_config,
        args.checkpoint,
        device,
    )
    metrics_path = args.checkpoint.with_suffix(".metrics.json")
    _ = metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
