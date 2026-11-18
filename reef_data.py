"""Prepare observed OISST in reef-containing cells without loading all years.

Run ``uv run python reef_data.py --help``. The 2018 reef polygons are a static
all-touched mask, not reef-area weights or evidence of historical reef extent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import get_coordinates, get_parts, get_rings, make_valid
from shapely.affinity import translate
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


# Narrow signatures for the xarray operations whose upstream annotations leave
# keyword arguments unknown. These retain the library calls and lazy behavior.
class _OpenDataset(Protocol):
    def __call__(
        self, filename: str | Path, *, engine: str, chunks: dict[str, int]
    ) -> xr.Dataset: ...


class _LoadArray(Protocol):
    def __call__(self, *, scheduler: str) -> xr.DataArray: ...


class _WriteNetCDF(Protocol):
    def __call__(
        self,
        path: Path,
        *,
        engine: str = ...,
        encoding: dict[str, dict[str, str | int | float]] = ...,
    ) -> None: ...


class _OpenMultiDataset(Protocol):
    def __call__(
        self,
        paths: list[Path],
        *,
        combine: str,
        chunks: dict[str, int],
        combine_attrs: str,
        join: str,
    ) -> xr.Dataset: ...


class _Arguments(Protocol):
    sst_dir: Path
    reef_archive: Path
    reef_layer: str
    output_dir: Path
    years: list[int]
    bounds: list[float]
    product_version: str
    overwrite: bool


LOGGER = logging.getLogger(__name__)
REEF_ARCHIVE = Path("data/reef-polygons/WCMC008_CoralReefs2018_v4_1.zip")
REEF_LAYER = (
    "14_001_WCMC008_CoralReefs2018_v4_1/01_Data/WCMC008_CoralReef2018_Py_v4_1.shp"
)
REEF_CITATION = (
    "UNEP-WCMC, WorldFish Centre, WRI, TNC (2021). Global distribution of coral "
    "reefs, compiled from multiple sources including the Millennium Coral Reef "
    "Mapping Project. Version 4.1, updated by UNEP-WCMC. Includes contributions "
    "from IMaRS-USF and IRD (2005), IMaRS-USF (2005) and Spalding et al. (2001). "
    "Cambridge (UK): UN Environment Programme World Conservation Monitoring "
    "Centre. https://doi.org/10.34892/t2wk-5t34"
)
GRID_STEP = 0.25
MISSING_SEMANTICS = (
    "NaN SST is unknown, never zero or safe. sst_valid=1 only for finite "
    "observations. Missing calendar dates are inserted as NaN. Counts use "
    "the full expected calendar year, except OISST starts 1981-09-01. "
    "Unobserved future calendar slots in incomplete years are not forecasts."
)


class GeometryFailureDetail(TypedDict):
    position: int
    reason: str


class GeometryRepairReport(TypedDict):
    feature_count: int
    invalid_before: int
    repaired_count: int
    non_polygonal_remnant_features: int
    failed_feature_indices: list[int]
    failure_details: list[GeometryFailureDetail]


class SoftwareVersions(TypedDict):
    numpy: str
    pandas: str
    xarray: str
    geopandas: str


class ManifestFileRecord(TypedDict):
    source: str
    source_sha256: str
    output: str
    year: int
    cell_count: int
    expected_day_count: int
    source_day_count: int
    missing_source_dates: list[str]
    product_version_evidence: str


class ReefDataManifest(TypedDict):
    schema_version: int
    bounds_west_south_east_north: list[float] | None
    region_selection: str
    reef_archive: str
    reef_layer: str
    reef_archive_sha256: str
    geometry_report: GeometryRepairReport
    rasterization: str
    reef_license_file: str
    reef_license_sha256: str
    polygon_citation: str
    missing_data_semantics: str
    software_versions: SoftwareVersions
    files: list[ManifestFileRecord]


def normalize_longitude(values: ArrayLike) -> NDArray[np.float64]:
    """Return longitude in [-180, 180); +180 and -180 share one identity."""
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


class GeometryRepairError(ValueError):
    """Geometry failures include a machine-readable report; no silent drops."""

    def __init__(self, report: GeometryRepairReport):
        self.report: GeometryRepairReport = report
        super().__init__(f"Reef geometry repair failed: {json.dumps(report)}")


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry] if not geometry.is_empty and geometry.area > 0 else []
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        return [
            part
            for child in cast(Iterable[BaseGeometry], geometry.geoms)
            for part in _polygon_parts(child)
        ]
    return []


def _has_dateline_edges(geometry: BaseGeometry) -> bool:
    if geometry.bounds[2] - geometry.bounds[0] <= 180:
        return False
    coordinates, rings = cast(
        tuple[NDArray[np.float64], NDArray[np.intp]],
        get_coordinates(get_rings(get_parts(geometry)), return_index=True),
    )
    return bool(
        np.any(
            cast(
                NDArray[np.bool_],
                (np.abs(np.diff(coordinates[:, 0])) > 180) & (rings[1:] == rings[:-1]),
            )
        )
    )


def repair_geometries(
    reefs: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, GeometryRepairReport]:
    """Repair every feature with make_valid; retain its row and attributes.

    Polygonal parts of repaired GeometryCollections are retained. Non-area
    remnants are counted; a feature with no remaining area fails the run.
    """
    report: GeometryRepairReport = {
        "feature_count": len(reefs),
        "invalid_before": 0,
        "repaired_count": 0,
        "non_polygonal_remnant_features": 0,
        "failed_feature_indices": [],
        "failure_details": [],
    }
    geometries: list[BaseGeometry | None] = []
    # GeoSeries permits missing geometries despite its non-null iterator annotation.
    for position, geometry in enumerate(
        cast(Iterable[BaseGeometry | None], reefs.geometry)
    ):
        try:
            if geometry is None or geometry.is_empty:
                raise ValueError("null or empty geometry")
            invalid = not geometry.is_valid
            report["invalid_before"] += int(invalid)
            # Unwrap crossing shells/holes before planar repair, but report
            # invalidity in the original source representation.
            repaired = (
                wrap_geometry(geometry, _repair_report=report)
                if _has_dateline_edges(geometry)
                else make_valid(geometry)
                if invalid
                else geometry
            )
            if isinstance(repaired, GeometryCollection):
                report["non_polygonal_remnant_features"] += 1
            parts = _polygon_parts(repaired)
            if not parts:
                raise ValueError("no polygonal area remains")
            if isinstance(repaired, GeometryCollection):
                repaired = parts[0] if len(parts) == 1 else unary_union(parts)
            if not repaired.is_valid or repaired.is_empty:
                raise ValueError("repair did not produce valid polygonal area")
            geometries.append(repaired)
            report["repaired_count"] += int(invalid)
        except (ValueError, TypeError, GEOSException) as error:
            report["failed_feature_indices"].append(position)
            report["failure_details"].append(
                {"position": position, "reason": str(error)}
            )
            geometries.append(None)
    LOGGER.info("Geometry repair: %s", json.dumps(report, sort_keys=True))
    if report["failed_feature_indices"]:
        raise GeometryRepairError(report)
    result = reefs.copy()
    # All missing entries accompany reported failures, which raised above.
    result.geometry = cast(list[BaseGeometry], geometries)
    return result, report


def wrap_geometry(
    geometry: BaseGeometry, *, _repair_report: GeometryRepairReport | None = None
) -> BaseGeometry:
    """Unwrap each ring along shortest longitude edges, then split at ±180.

    Holes follow their exterior's longitude branch. Polygon parts are handled
    independently so distant islands in a MultiPolygon never become connected.
    """
    if (
        geometry.is_valid
        and not _has_dateline_edges(geometry)
        and box(-180, -90, 180, 90).covers(geometry)
    ):
        return geometry
    had_non_polygonal_remnants = False
    pieces: list[Polygon] = []
    world = box(-180, -90, 180, 90)
    for polygon in _polygon_parts(geometry):
        exterior = np.asarray(polygon.exterior.coords, dtype=float)[:, :2].copy()
        exterior[:, 0] = np.unwrap(exterior[:, 0], period=360)
        anchor = cast(np.float64, exterior[:, 0].mean())
        holes: list[NDArray[np.float64]] = []
        for ring in polygon.interiors:
            hole = np.asarray(ring.coords, dtype=float)[:, :2].copy()
            hole[:, 0] = np.unwrap(hole[:, 0], period=360)
            hole[:, 0] += 360 * round(
                (anchor - cast(np.float64, hole[:, 0].mean())) / 360
            )
            holes.append(hole)
        unwrapped = Polygon(exterior, holes)
        if not unwrapped.is_valid:
            unwrapped = make_valid(unwrapped)
            had_non_polygonal_remnants |= isinstance(unwrapped, GeometryCollection)
        for part in _polygon_parts(unwrapped):
            left, _, right, _ = part.bounds
            for offset in range(
                math.ceil((-180 - right) / 360),
                math.floor((180 - left) / 360) + 1,
            ):
                clipped = translate(part, xoff=360 * offset).intersection(world)
                pieces.extend(_polygon_parts(clipped))
    if not pieces:
        raise ValueError("Longitude wrapping left no polygonal area")
    result = unary_union(pieces)
    if not result.is_valid:
        raise ValueError("Longitude wrapping produced invalid polygonal area")
    if _repair_report is not None and had_non_polygonal_remnants:
        _repair_report["non_polygonal_remnant_features"] += 1
    return result


def read_reefs(
    archive: str | Path, layer: str = REEF_LAYER
) -> tuple[gpd.GeoDataFrame, GeometryRepairReport]:
    """Read a polygon layer directly from ZIP (or a supplied extracted SHP)."""
    archive = Path(archive)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as source:
            if layer not in source.namelist():
                raise ValueError(f"Polygon layer not present in archive: {layer}")
        uri = f"zip://{archive.resolve()}!{layer}"
    else:
        uri = str(archive)
    reefs = cast(Callable[[str], gpd.GeoDataFrame], gpd.read_file)(uri)
    if reefs.empty:
        raise ValueError("Reef polygon layer is empty")
    if reefs.crs is None or reefs.crs.to_epsg() != 4326:
        raise ValueError("Reef polygons must declare EPSG:4326")
    return repair_geometries(reefs)


def validate_grid(ds: xr.Dataset) -> None:
    """Require daily SST dimensions and unique, aligned OISST cell centers."""
    if "sst" not in ds or set(ds["sst"].dims) != {"time", "lat", "lon"}:
        raise ValueError("sst must have exactly time, lat, lon dimensions")
    for name in ("time", "lat", "lon"):
        if name not in ds.coords or ds[name].dims != (name,) or not ds.sizes[name]:
            raise ValueError(f"{name} must be a nonempty one-dimensional coordinate")
        if not ds.indexes[name].is_unique:
            raise ValueError(f"Duplicate {name} coordinates")
    for name, origin, limit in (("lat", -89.875, 89.875), ("lon", -179.875, 179.875)):
        raw = np.asarray(ds[name].values, dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError(f"{name} must be finite")
        differences = np.diff(raw)
        if differences.size and not (
            np.all(differences > 0) or np.all(differences < 0)
        ):
            raise ValueError(f"{name} must be strictly monotonic")
        if differences.size and not np.allclose(
            np.abs(differences), GRID_STEP, rtol=0, atol=1e-6
        ):
            raise ValueError(f"{name} spacing must be 0.25 degrees")
        values = normalize_longitude(raw) if name == "lon" else raw
        indices = (values - origin) / GRID_STEP
        if (
            np.any(values < origin - 1e-6)
            or np.any(values > limit + 1e-6)
            or not np.allclose(indices, np.rint(indices), rtol=0, atol=1e-5)
            or len(np.unique(values)) != len(values)
        ):
            raise ValueError(f"{name} coordinates are not aligned OISST centers")


def _canonical_grid(ds: xr.Dataset) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.sort(cast(NDArray[np.float64], ds["lat"].values)),
        np.sort(normalize_longitude(ds["lon"].values)),
    )


def build_reef_cells(
    ds: xr.Dataset,
    reefs: gpd.GeoDataFrame,
    bounds: Sequence[float] | None = None,
) -> xr.Dataset:
    """All-touched cells with center-based regional selection and global IDs.

    bounds=(west,south,east,north) uses [-180,180] longitude; west>east crosses
    the dateline. Boolean membership is NOT fractional reef area.
    """
    validate_grid(ds)
    latitude, longitude = _canonical_grid(ds)
    # Rasterize in descending latitude with cell edges half a step beyond
    # centers. A complete longitude axis handles subsets straddling ±180.
    full_longitude = np.arange(-179.875, 180, GRID_STEP)
    transform = from_origin(
        -180, cast(np.float64, latitude[-1]) + GRID_STEP / 2, GRID_STEP, GRID_STEP
    )
    shapes = [(wrap_geometry(geometry), 1) for geometry in reefs.geometry]
    mask = rasterize(
        shapes,
        out_shape=(len(latitude), len(full_longitude)),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )[::-1]
    lon_columns = np.rint((longitude + 179.875) / GRID_STEP).astype(int)
    mask = mask[:, lon_columns].astype(bool)
    if bounds is not None:
        west, south, east, north = bounds
        if not (
            -180 <= west <= 180
            and -180 <= east <= 180
            and -90 <= south < north <= 90
            and west != east
        ):
            raise ValueError("Invalid region bounds (west,south,east,north)")
        lon_inside = (
            (longitude >= west) & (longitude <= east)
            if west < east
            else (longitude >= west) | (longitude <= east)
        )
        mask &= ((latitude >= south) & (latitude <= north))[:, None]
        mask &= lon_inside[None, :]
    row, column = np.nonzero(mask)
    if not len(row):
        raise ValueError("No reef cells intersect the selected grid/region")
    lat, lon = latitude[row], longitude[column]
    lat_index = np.rint((lat + 89.875) / GRID_STEP).astype("int32")
    lon_index = np.rint((lon + 179.875) / GRID_STEP).astype("int32")
    cells = xr.Dataset(
        coords={
            "cell": lat_index * 1440 + lon_index,
            "lat": ("cell", lat),
            "lon": ("cell", lon),
        },
        data_vars={
            "lat_index": ("cell", lat_index),
            "lon_index": ("cell", lon_index),
        },
    )
    cells["source_lat_index"] = (
        "cell",
        ds.indexes["lat"].get_indexer(pd.Index(lat)).astype("int32"),
    )
    cells["source_lon_index"] = (
        "cell",
        pd.Index(normalize_longitude(ds["lon"].values))
        .get_indexer(pd.Index(lon))
        .astype("int32"),
    )
    cells["source_lon"] = ("cell", ds["lon"].values[cells["source_lon_index"].values])
    cells["cell"].attrs["definition"] = (
        "cell_id = lat_index * 1440 + lon_index; canonical 0.25-degree "
        "centers start at lat=-89.875, lon=-179.875, longitude in [-180,180)"
    )
    return cells


def _expected_dates(year: int) -> pd.DatetimeIndex:
    if not 1981 <= year <= 9998:
        raise ValueError("OISST year must be at least 1981")
    start = f"{year}-01-01" if year != 1981 else "1981-09-01"
    return pd.date_range(start, f"{year}-12-31", freq="D")


def _validate_source(
    ds: xr.Dataset, year: int, product_version: str
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, object]:
    validate_grid(ds)
    if product_version != "2.1":
        raise ValueError("This pipeline requires declared OISST product version 2.1")
    metadata = cast(Mapping[str, object], ds.attrs)
    title = " ".join(
        str(metadata.get(key, ""))
        for key in ("title", "dataset_title", "source", "institution")
    ).lower()
    if not (
        "noaa" in title
        and ("optimum interpolation" in title or "oisst" in title or "blended" in title)
    ):
        raise ValueError("Source metadata must identify NOAA OISST")
    explicit = metadata.get("product_version", metadata.get("version"))
    if explicit is not None and not re.fullmatch(
        r"(?:version\s*|v)?2\.1", str(explicit).strip(), flags=re.IGNORECASE
    ):
        raise ValueError(f"Source product version conflicts with 2.1: {explicit}")
    sst_metadata = cast(Mapping[str, object], ds["sst"].attrs)
    units = str(sst_metadata.get("units", "")).lower()
    if units not in {
        "degc",
        "degree_celsius",
        "degrees_celsius",
        "celsius",
        "degree_c",
    }:
        raise ValueError(f"SST units must be degrees Celsius, not {units!r}")
    dates = pd.DatetimeIndex(ds["time"].values)
    if dates.hasnans or not dates.is_unique:
        raise ValueError("Missing or duplicate dates in time coordinate")
    if not dates.is_monotonic_increasing:
        raise ValueError("Daily time coordinates must be monotonically increasing")
    if not dates.equals(dates.normalize()):
        raise ValueError("Daily time coordinates must be at midnight")
    expected = _expected_dates(year)
    if not dates.isin(expected).all():
        raise ValueError(f"Source dates fall outside expected OISST year {year}")
    return dates, expected, explicit


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        # Metadata arrays can contain arbitrary Python objects, including scalars
        # for zero-dimensional arrays; json applies this default recursively.
        return cast(Callable[[], object], value.tolist)()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def extract_reef_sst(
    ds: xr.Dataset,
    cells: xr.Dataset,
    year: int,
    source_file: str = "fixture.nc",
    product_version: str = "2.1",
) -> xr.Dataset:
    """Validate and materialize only one year's selected cells, preserving NaN."""
    dates, expected, explicit_version = _validate_source(ds, year, product_version)
    lat_indices = cast(
        Callable[[ArrayLike], NDArray[np.intp]], ds.indexes["lat"].get_indexer
    )(cells["lat"].values)
    normalized_lons = pd.Index(normalize_longitude(ds["lon"].values))
    lon_indices = cast(
        Callable[[ArrayLike], NDArray[np.intp]], normalized_lons.get_indexer
    )(cells["lon"].values)
    if (lat_indices < 0).any() or (lon_indices < 0).any():
        raise ValueError("Reef cells do not align with this source grid")
    selected = (
        ds["sst"]
        .isel(
            lat=xr.DataArray(lat_indices, dims="cell"),
            lon=xr.DataArray(lon_indices, dims="cell"),
        )
        .transpose("time", "cell")
    )
    # Only the extracted annual cell array is materialized. Single-threaded
    # evaluation bounds concurrent global-source chunk decompression.
    selected = cast(_LoadArray, selected.reindex(time=expected).load)(
        scheduler="single-threaded"
    )
    values = cast(NDArray[np.float32], selected.values.astype("float32"))
    valid: NDArray[np.bool_] = np.isfinite(values)
    values[~valid] = np.nan
    counts = cast(NDArray[np.int64], valid.sum(axis=0)).astype("int32")
    output = cells.copy(deep=True)
    output = cast(Callable[[Mapping[str, object]], xr.Dataset], output.assign_coords)(
        {"time": expected}
    )
    output["sst"] = (("time", "cell"), values)
    output["sst"].attrs = {"units": "degC", "long_name": "Observed NOAA OISST SST"}
    output["sst_valid"] = (("time", "cell"), valid.astype("uint8"))
    output["source_time_present"] = ("time", expected.isin(dates).astype("uint8"))
    output["valid_observation_count"] = ("cell", counts)
    output["expected_observation_count"] = (
        "cell",
        np.full(len(counts), len(expected), dtype="int32"),
    )
    output["valid_coverage_fraction"] = ("cell", counts / len(expected))
    output["source_lat_index"] = ("cell", lat_indices.astype("int32"))
    output["source_lon_index"] = ("cell", lon_indices.astype("int32"))
    output["source_lon"] = ("cell", ds["lon"].values[lon_indices])
    output["source_file"] = ("time", np.repeat(str(source_file), len(expected)))
    output["source_year"] = ("time", np.full(len(expected), year, dtype="int32"))
    for name in ("sst_valid", "source_time_present"):
        output[name].attrs.update(
            flag_values=np.array([0, 1], dtype="uint8"),
            flag_meanings="unavailable available",
        )
    output["lat"].attrs.update(units="degrees_north", standard_name="latitude")
    output["lon"].attrs.update(units="degrees_east", standard_name="longitude")
    output["valid_coverage_fraction"].attrs["denominator"] = (
        "expected_observation_count: full expected OISST calendar period"
    )
    date_values = dates.to_numpy()
    expected_values = expected.to_numpy()
    missing_date_values = expected.difference(dates).to_numpy()
    output.attrs.update(
        Conventions="CF-1.8",
        data_kind="observations; missing calendar slots are not forecasts",
        sst_product="NOAA OISST",
        declared_product_version=product_version,
        product_version_evidence=(
            "source metadata explicitly declares 2.1"
            if explicit_version is not None
            else "user-declared 2.1; source metadata does not explicitly verify version"
        ),
        source_metadata=json.dumps(ds.attrs, default=_json_default, sort_keys=True),
        source_sst_metadata=json.dumps(
            ds["sst"].attrs, default=_json_default, sort_keys=True
        ),
        source_file=str(source_file),
        source_year=year,
        source_time_start=np.datetime_as_string(
            cast(np.datetime64, date_values[0]), unit="s"
        ),
        source_time_end=np.datetime_as_string(
            cast(np.datetime64, date_values[-1]), unit="s"
        ),
        expected_time_start=np.datetime_as_string(
            cast(np.datetime64, expected_values[0]), unit="s"
        ),
        expected_time_end=np.datetime_as_string(
            cast(np.datetime64, expected_values[-1]), unit="s"
        ),
        missing_source_dates=json.dumps(
            [np.datetime_as_string(value, unit="D") for value in missing_date_values]
        ),
        missing_data_semantics=MISSING_SEMANTICS,
        rasterization="all_touched boolean; no reef-area weighting",
        spatial_resolution_degrees=GRID_STEP,
        reef_extent_caveat=(
            "Static WCMC008 v4.1 compilation with heterogeneous source dates; "
            "historical/future reef extent is not inferred. SST is cell-scale, "
            "not colony temperature."
        ),
    )
    return output


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_reef_data(
    sst_files: Iterable[str | Path],
    reef_archive: str | Path,
    reef_layer: str,
    output_dir: str | Path,
    bounds: Sequence[float] | None,
    product_version: str = "2.1",
    overwrite: bool = False,
) -> ReefDataManifest:
    """Write one independently usable NetCDF per year and a provenance manifest."""
    files = sorted(Path(path) for path in sst_files)
    if not files:
        raise ValueError("No SST files selected")
    years: list[int] = []
    for path in files:
        match = re.fullmatch(r"sst\.day\.mean\.(\d{4})\.nc", path.name)
        if match is None:
            raise ValueError(f"Not an annual OISST source filename: {path}")
        years.append(int(match[1]))
    if len(set(years)) != len(years):
        raise ValueError("More than one source file selected for the same year")
    output_dir = Path(output_dir)
    outputs = [output_dir / f"reef_sst_{year}.nc" for year in years]
    if not overwrite and any(
        path.exists() for path in outputs + [output_dir / "manifest.json"]
    ):
        raise FileExistsError(
            "Output exists; choose a new directory or use --overwrite"
        )
    reefs, geometry_report = read_reefs(reef_archive, reef_layer)
    reef_hash = _sha256(reef_archive)
    license_bytes = (
        b"See the source polygon license and obtain redistribution permission."
    )
    if Path(reef_archive).suffix.lower() == ".zip":
        with zipfile.ZipFile(reef_archive) as source:
            licenses = [
                name for name in source.namelist() if name.endswith("/LICENSE.txt")
            ]
            if licenses:
                license_bytes = source.read(licenses[0])
    file_records: list[ManifestFileRecord] = []
    manifest: ReefDataManifest = {
        "schema_version": 1,
        "bounds_west_south_east_north": list(bounds) if bounds is not None else None,
        "region_selection": "grid-cell centers within bounds; west>east crosses dateline",
        "reef_archive": str(reef_archive),
        "reef_layer": reef_layer,
        "reef_archive_sha256": reef_hash,
        "geometry_report": geometry_report,
        "rasterization": "all_touched boolean; not fractional reef coverage",
        "reef_license_file": "reef_source_LICENSE.txt",
        "reef_license_sha256": hashlib.sha256(license_bytes).hexdigest(),
        "polygon_citation": REEF_CITATION,
        "missing_data_semantics": MISSING_SEMANTICS,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "geopandas": gpd.__version__,
        },
        "files": file_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _ = (output_dir / "reef_source_LICENSE.txt").write_bytes(license_bytes)
    reference_grid = None
    cells: xr.Dataset | None = None
    for path, year, destination in zip(files, years, outputs, strict=True):
        LOGGER.info("Processing %s", path)
        # Source chunks are bounded in both space and time; only one input is open.
        with open_nc_dataset(path) as ds:
            validate_grid(ds)
            current_grid = _canonical_grid(ds)
            if cells is None:
                reference_grid = current_grid
                cells = build_reef_cells(ds, reefs, bounds)
            else:
                assert reference_grid is not None
                if any(
                    not np.array_equal(current, reference)
                    for current, reference in zip(
                        current_grid, reference_grid, strict=True
                    )
                ):
                    raise ValueError(
                        f"Inconsistent grid alignment across years: {path}"
                    )
            output = extract_reef_sst(ds, cells, year, str(path), product_version)
            checksum = _sha256(path)
            output.attrs.update(
                source_sha256=checksum,
                reef_archive=str(reef_archive),
                reef_layer=reef_layer,
                reef_archive_sha256=reef_hash,
                geometry_repair_report=json.dumps(geometry_report, sort_keys=True),
                region_bounds=json.dumps(manifest["bounds_west_south_east_north"]),
                polygon_source="UNEP-WCMC WCMC008 Coral Reefs 2018 v4.1",
                polygon_citation=REEF_CITATION,
                polygon_license=(
                    "UNEP-WCMC General Data License (excluding WDPA), 2014-07-21; "
                    "commercial use and redistribution require permission. "
                    "See reef_source_LICENSE.txt copied from archive."
                ),
            )
            temporary = destination.with_suffix(".nc.tmp")
            try:
                cast(_WriteNetCDF, output.to_netcdf)(
                    temporary,
                    engine="netcdf4",
                    encoding={
                        "sst": {"zlib": True, "complevel": 4, "_FillValue": np.nan}
                    },
                )
                _ = temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            file_records.append(
                {
                    "source": str(path),
                    "source_sha256": checksum,
                    "output": destination.name,
                    "year": year,
                    "cell_count": output.sizes["cell"],
                    "expected_day_count": output.sizes["time"],
                    "source_day_count": int(output["source_time_present"].sum()),
                    "missing_source_dates": json.loads(
                        cast(str, output.attrs["missing_source_dates"])
                    ),
                    "product_version_evidence": output.attrs[
                        "product_version_evidence"
                    ],
                }
            )
            LOGGER.info(
                "Wrote %s: %d cells, %d expected days; %s",
                destination,
                output.sizes["cell"],
                output.sizes["time"],
                cast(str, output.attrs["product_version_evidence"]),
            )
    manifest_path = output_dir / "manifest.json"
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return manifest


def open_nc_dataset(filename: str | Path) -> xr.Dataset:
    """Open one source lazily; explicit paths no longer depend on import-time env."""
    return cast(_OpenDataset, xr.open_dataset)(
        filename, engine="netcdf4", chunks={"time": 31, "lat": 90, "lon": 180}
    )


def add_reef_mask(
    filename: str | Path | None, reef_only_return: bool, ds: xr.Dataset | None = None
) -> xr.Dataset | xr.DataArray:
    """Research entrypoint: preserve missing cells; never select by SST validity."""
    # A filename is needed only when the caller has not supplied a dataset.
    source = open_nc_dataset(cast(str | Path, filename)) if ds is None else ds
    reefs, _ = read_reefs(REEF_ARCHIVE)
    cells = build_reef_cells(source, reefs)
    if reef_only_return:
        selected = source["sst"].isel(
            lat=xr.DataArray(
                cast(
                    Callable[[ArrayLike], NDArray[np.intp]],
                    source.indexes["lat"].get_indexer,
                )(cells["lat"].values),
                dims="points",
            ),
            lon=xr.DataArray(
                cast(
                    Callable[[ArrayLike], NDArray[np.intp]],
                    pd.Index(normalize_longitude(source["lon"].values)).get_indexer,
                )(cells["lon"].values),
                dims="points",
            ),
        )
        selected = cast(
            Callable[[Mapping[str, object]], xr.DataArray], selected.assign_coords
        )({"points": cells["cell"].values, "lon": ("points", cells["lon"].values)})
        if ds is None:
            selected.set_close(source.close)
        return selected
    normalized = cast(
        Callable[[Mapping[str, object]], xr.Dataset], source.assign_coords
    )({"lon": normalize_longitude(source["lon"])}).sortby("lon")
    mask = xr.DataArray(
        np.zeros((source.sizes["lat"], source.sizes["lon"]), dtype=bool),
        dims=("lat", "lon"),
        coords={"lat": source["lat"], "lon": normalized["lon"]},
    )
    row = cast(
        Callable[[ArrayLike], NDArray[np.intp]], normalized.indexes["lat"].get_indexer
    )(cells["lat"].values)
    column = cast(
        Callable[[ArrayLike], NDArray[np.intp]], normalized.indexes["lon"].get_indexer
    )(cells["lon"].values)
    mask.values[row, column] = True
    result = normalized.assign(reef_points=mask)
    if ds is None:
        result.set_close(source.close)
    return result


def merge_ds(path: str | Path = "reef_sst", dim: str = "points") -> xr.Dataset:
    """Research-only lazy merge; exclude previously generated merged outputs."""
    _ = dim
    files = sorted(
        candidate
        for candidate in Path(path).glob("*.nc")
        if not candidate.name.startswith("merged")
    )
    if not files:
        raise ValueError(f"No source NetCDF files in {path}")
    return cast(_OpenMultiDataset, xr.open_mfdataset)(
        files,
        combine="by_coords",
        chunks={"time": 31},
        combine_attrs="override",
        join="exact",
    )


def save(
    ds: xr.Dataset,
    year: int | None,
    name: str | None = None,
    path: str | Path = "reef_sst",
) -> None:
    """Save a research dataset; year-based names do not require a spare argument."""
    if year is not None:
        filename = f"reef_sst_{year}.nc"
    elif name:
        filename = name
    else:
        raise ValueError("name is required when year is None")
    if "points" in ds.indexes and isinstance(ds.indexes["points"], pd.MultiIndex):
        ds = ds.reset_index("points")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    cast(_WriteNetCDF, ds.to_netcdf)(destination / filename)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--sst-dir", type=Path, default=Path("data/sst"))
    _ = parser.add_argument("--reef-archive", type=Path, default=REEF_ARCHIVE)
    _ = parser.add_argument("--reef-layer", default=REEF_LAYER)
    _ = parser.add_argument("--output-dir", type=Path, default=Path("data/reef-sst"))
    _ = parser.add_argument("--years", type=int, nargs="+", required=True)
    _ = parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Regional cell-center bounds in [-180,180]; west>east crosses dateline",
    )
    _ = parser.add_argument("--product-version", default="2.1")
    _ = parser.add_argument("--overwrite", action="store_true")
    # argparse creates these attributes from the declarations above. Namespace
    # does not expose them statically, so bridge to that precise parsed contract.
    arguments = cast(_Arguments, cast(object, parser.parse_args(argv)))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = prepare_reef_data(
        [arguments.sst_dir / f"sst.day.mean.{year}.nc" for year in arguments.years],
        arguments.reef_archive,
        arguments.reef_layer,
        arguments.output_dir,
        arguments.bounds,
        arguments.product_version,
        arguments.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
