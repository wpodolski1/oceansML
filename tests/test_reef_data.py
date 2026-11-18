"""Small, synthetic scientific regression tests; no external data or network."""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, cast
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.typing import ArrayLike
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from reef_data import (
    GeometryRepairError,
    build_reef_cells,
    extract_reef_sst,
    normalize_longitude,
    prepare_reef_data,
    repair_geometries,
    validate_grid,
    wrap_geometry,
)


class _SelectArray(Protocol):
    def __call__(self, *, cell: float = ..., time: str = ...) -> xr.DataArray: ...


def _numeric_item(array: xr.DataArray) -> float:
    """Numeric fixture variables yield Python bool, int or float scalars."""
    return cast(Callable[[], float], array.item)()


def _text_item(array: xr.DataArray) -> str:
    """The source-file provenance variable contains string scalars."""
    return cast(Callable[[], str], array.item)()


def make_sst(
    *,
    lat: ArrayLike | None = None,
    lon: ArrayLike | None = None,
    times: list[str] | pd.DatetimeIndex | None = None,
    values: ArrayLike | None = None,
) -> xr.Dataset:
    """Canonical quarter-degree centers with recognizable location histories."""
    lat = np.asarray([-0.125, 0.125] if lat is None else lat, dtype=float)
    lon = np.asarray([10.125, 10.375] if lon is None else lon, dtype=float)
    times = pd.to_datetime(
        ["2000-01-01", "2000-01-02", "2000-01-03"] if times is None else times
    )
    if values is None:
        values = (
            np.arange(len(times))[:, None, None]
            + np.arange(len(lat))[None, :, None] * 10
            + np.arange(len(lon))[None, None, :] * 100
        ).astype(np.float32)
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.asarray(values, dtype=np.float32))},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs={
            "title": "NOAA Daily Optimum Interpolation SST",
            "product_version": "2.1",
        },
    )
    ds["sst"].attrs["units"] = "degC"
    return ds


def reefs_for(*geometries: BaseGeometry | None) -> gpd.GeoDataFrame:
    # Object arrays preserve null fixtures and match GeoPandas' nullable input API.
    return gpd.GeoDataFrame(
        geometry=np.asarray(geometries, dtype=object), crs="EPSG:4326"
    )


def all_cells(ds: xr.Dataset) -> xr.Dataset:
    return build_reef_cells(ds, reefs_for(box(10.01, -0.24, 10.49, 0.24)))


def cell_id(lat: float, lon: float) -> int:
    return round((lat + 89.875) / 0.25) * 1440 + round((lon + 179.875) / 0.25)


def write_reef_zip(tmp_path: Path) -> tuple[Path, str]:
    shape_dir = tmp_path / "shapes"
    shape_dir.mkdir()
    layer = shape_dir / "reefs.shp"
    cast(Callable[[Path], None], reefs_for(box(10.01, -0.24, 10.49, 0.24)).to_file)(
        layer
    )
    archive = tmp_path / "reefs.zip"
    with ZipFile(archive, "w") as zipped:
        for component in shape_dir.iterdir():
            zipped.write(component, arcname=f"nested/{component.name}")
        zipped.writestr("nested/LICENSE.txt", b"\x93No redistribution\x94\n")
    return archive, "nested/reefs.shp"


def test_repair_bowtie_preserves_feature_and_polygonal_area():
    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    source = reefs_for(bowtie, box(5, 5, 6, 6))
    repaired, report = repair_geometries(source)
    assert report["feature_count"] == 2
    assert report["invalid_before"] == 1
    assert report["repaired_count"] == 1
    assert report["failed_feature_indices"] == []
    assert len(repaired) == 2
    assert repaired.geometry.is_valid.all()
    # GeoPandas scalar indexing inherits pandas' broader scalar typing.
    repaired_bowtie = repaired.geometry.iloc[0]
    repaired_box = repaired.geometry.iloc[1]
    original_bowtie = source.geometry.iloc[0]
    assert isinstance(repaired_bowtie, BaseGeometry)
    assert isinstance(repaired_box, BaseGeometry)
    assert isinstance(original_bowtie, BaseGeometry)
    assert repaired_bowtie.area == pytest.approx(2)
    original_box = source.geometry.iloc[1]
    assert isinstance(original_box, BaseGeometry)
    assert repaired_box.equals(original_box)
    assert not original_bowtie.is_valid


@pytest.mark.parametrize("geometry", [None, Polygon()])
def test_unrepairable_geometry_reports_failure_instead_of_dropping(
    geometry: BaseGeometry | None,
) -> None:
    source = reefs_for(box(0, 0, 1, 1), geometry)
    with pytest.raises(GeometryRepairError) as caught:
        _ = repair_geometries(source)
    assert caught.value.report["feature_count"] == 2
    assert caught.value.report["failed_feature_indices"] == [1]


def test_wrapping_dateline_and_disconnected_islands_preserves_area():
    crossing = Polygon([(179, -1), (-179, -1), (-179, 1), (179, 1)])
    wrapped = wrap_geometry(MultiPolygon([crossing, box(10, 10, 11, 11)]))
    assert wrapped.is_valid
    assert wrapped.area == pytest.approx(5)
    assert wrapped.bounds == pytest.approx((-180, -1, 180, 11))
    assert wrapped.covers(box(179.1, -0.9, 179.9, 0.9))
    assert wrapped.covers(box(-179.9, -0.9, -179.1, 0.9))
    assert wrapped.covers(box(10.1, 10.1, 10.9, 10.9))
    assert wrapped.intersection(box(-170, -0.9, 170, 0.9)).is_empty


@pytest.mark.parametrize("descending", [False, True])
def test_raster_uses_half_cell_edges_and_all_touched(descending: bool) -> None:
    ds = make_sst(lat=[0.125, -0.125] if descending else [-0.125, 0.125])
    # Inside the upper-right cell, but deliberately far from its center.
    cells = build_reef_cells(ds, reefs_for(box(10.26, 0.01, 10.27, 0.02)))
    assert cells.sizes["cell"] == 1
    assert _numeric_item(cells["cell"]) == cell_id(0.125, 10.375)
    assert _numeric_item(cells["lat"]) == pytest.approx(0.125)
    assert _numeric_item(cells["lon"]) == pytest.approx(10.375)
    assert _numeric_item(cells["source_lat_index"]) == (0 if descending else 1)
    assert _numeric_item(cells["source_lon_index"]) == 1


def test_normalize_longitude_and_native_conventions_share_cell_identity():
    np.testing.assert_allclose(
        normalize_longitude(np.array([0, 180, 359.875, -180, -0.125])),
        [0, -180, -0.125, -180, -0.125],
    )
    reefs = reefs_for(box(-0.49, -0.24, -0.01, 0.24))
    west = build_reef_cells(make_sst(lon=[-0.375, -0.125]), reefs)
    east = build_reef_cells(make_sst(lon=[359.625, 359.875]), reefs)
    np.testing.assert_array_equal(np.sort(west["cell"]), np.sort(east["cell"]))
    for identity in cast(Iterable[np.int32], west["cell"].values):
        a, b = west.sel(cell=identity), east.sel(cell=identity)
        assert _numeric_item(a["lon"]) == _numeric_item(b["lon"])
        assert _numeric_item(a["lat_index"]) == _numeric_item(b["lat_index"])
        assert _numeric_item(a["lon_index"]) == _numeric_item(b["lon_index"])
        assert _numeric_item(b["source_lon"]) == pytest.approx(
            _numeric_item(b["lon"]) + 360
        )


def test_ids_survive_native_axis_order_and_center_based_regional_cut():
    forward = make_sst()
    reverse = forward.isel(lat=slice(None, None, -1), lon=slice(None, None, -1))
    cells = all_cells(forward)
    reversed_cells = all_cells(reverse)
    np.testing.assert_array_equal(
        np.sort(cells["cell"]), np.sort(reversed_cells["cell"])
    )
    selected = build_reef_cells(
        reverse,
        reefs_for(box(10.01, -0.24, 10.49, 0.24)),
        bounds=(10.3, 0.0, 10.4, 0.2),
    )
    assert selected["cell"].values.tolist() == [cell_id(0.125, 10.375)]
    identity = _numeric_item(selected["cell"])
    assert _numeric_item(selected["source_lat_index"]) == 0
    assert _numeric_item(selected["source_lon_index"]) == 0
    assert _numeric_item(selected["lat_index"]) == _numeric_item(
        cells.sel(cell=identity)["lat_index"]
    )
    assert _numeric_item(selected["lon_index"]) == _numeric_item(
        cells.sel(cell=identity)["lon_index"]
    )


def test_dateline_bounds_select_both_sides_without_middle_cells():
    ds = make_sst(lon=[179.625, 179.875, 180.125, 180.375])
    reefs = reefs_for(
        box(-179.99, -0.24, -179.51, 0.24),
        box(179.51, -0.24, 179.99, 0.24),
    )
    cells = build_reef_cells(ds, reefs, bounds=(179.8, -0.2, -179.8, 0.2))
    assert cells.sizes["cell"] == 4
    assert set(cells["lon"].values) == {-179.875, 179.875}


def test_missing_sst_keeps_cells_and_records_counts_and_coverage():
    ds = make_sst()
    ds["sst"].values[:, 0, 0] = np.nan
    ds["sst"].values[1, 1, 1] = np.nan
    result = extract_reef_sst(ds, all_cells(ds), 2000, source_file="tiny.nc")
    assert result.sizes == {"time": 366, "cell": 4}
    assert result["sst"].dtype == np.dtype("float32")
    assert result["sst_valid"].dtype == np.dtype("uint8")
    missing = result.sel(cell=cell_id(-0.125, 10.125))
    partial = result.sel(cell=cell_id(0.125, 10.375))
    assert _numeric_item(missing["sst"].isnull().all())
    assert _numeric_item(missing["valid_observation_count"]) == 0
    assert _numeric_item(missing["valid_coverage_fraction"]) == 0
    assert _numeric_item(partial["valid_observation_count"]) == 2
    assert _numeric_item(partial["expected_observation_count"]) == 366
    assert _numeric_item(partial["valid_coverage_fraction"]) == pytest.approx(2 / 366)
    np.testing.assert_array_equal(partial["sst_valid"][:3], [1, 0, 1])
    np.testing.assert_array_equal(result["source_time_present"][:4], [1, 1, 1, 0])
    np.testing.assert_array_equal(
        result["sst_valid"],
        cast(Callable[[str], xr.DataArray], result["sst"].notnull().astype)("uint8"),
    )


def test_gap_reindex_distinguishes_missing_day_from_missing_observation():
    ds = make_sst(times=["2001-01-01", "2001-01-03"])
    result = extract_reef_sst(ds, all_cells(ds), 2001)
    assert result.sizes["time"] == 365
    np.testing.assert_array_equal(result["source_time_present"][:3], [1, 0, 1])
    assert _numeric_item(
        cast(_SelectArray, result["sst"].sel)(time="2001-01-02").isnull().all()
    )
    np.testing.assert_allclose(
        cast(_SelectArray, result["sst"].sel)(cell=cell_id(-0.125, 10.125))[:3],
        [0, np.nan, 1],
    )


@pytest.mark.parametrize(
    "times",
    [
        ["2000-01-01", "2000-01-01"],
        ["2000-01-02", "2000-01-01"],
        ["2000-01-01T12:00", "2000-01-02T12:00"],
    ],
)
def test_invalid_time_axis_is_rejected(times: list[str]) -> None:
    ds = make_sst(times=times)
    with pytest.raises(ValueError):
        _ = extract_reef_sst(ds, all_cells(ds), 2000)


@pytest.mark.parametrize(
    ("year", "first", "count", "present"),
    [
        (1981, "1981-09-01", 122, "1981-09-02"),
        (2000, "2000-01-01", 366, "2000-02-29"),
        (2001, "2001-01-01", 365, "2001-02-28"),
    ],
)
def test_expected_calendar_handles_leap_year_and_1981_start(
    year: int, first: str, count: int, present: str
) -> None:
    ds = make_sst(times=[present])
    result = extract_reef_sst(ds, all_cells(ds), year)
    assert result.sizes["time"] == count
    assert pd.Timestamp(cast(np.datetime64, result["time"].values[0])) == pd.Timestamp(
        first
    )
    assert pd.Timestamp(cast(np.datetime64, result["time"].values[-1])) == pd.Timestamp(
        year, 12, 31
    )
    assert (
        _numeric_item(
            cast(_SelectArray, result["source_time_present"].sel)(time=present)
        )
        == 1
    )
    assert _numeric_item(result["source_time_present"].sum()) == 1
    assert _numeric_item(result["expected_observation_count"].min()) == count


@pytest.mark.parametrize(
    ("axis", "coordinates"),
    [
        ("lon", [10.125, 10.125]),
        ("lat", [-0.125, -0.125]),
        ("lon", [10.125, 10.625]),
        ("lat", [-0.125, 0.375]),
        ("lon", [10.0, 10.25]),
        ("lat", [-0.1, 0.15]),
    ],
)
def test_grid_rejects_duplicate_nonquarter_degree_and_offcenter_axes(
    axis: str, coordinates: list[float]
) -> None:
    ds = cast(Callable[[Mapping[str, object]], xr.Dataset], make_sst().assign_coords)(
        {axis: coordinates}
    )
    with pytest.raises(ValueError):
        validate_grid(ds)


@pytest.mark.parametrize(
    "attribute,value", [("units", "K"), ("product_version", "2.0")]
)
def test_explicit_product_or_units_mismatch_is_rejected(
    attribute: str, value: str
) -> None:
    ds = make_sst()
    if attribute == "units":
        ds["sst"].attrs[attribute] = value
    else:
        ds.attrs[attribute] = value
    with pytest.raises(ValueError):
        _ = extract_reef_sst(ds, all_cells(ds), 2000, product_version="2.1")


def test_cellwise_consecutive_histories_never_mix_locations():
    ds = make_sst()
    output = extract_reef_sst(ds, all_cells(ds), 2000)
    for lat_i, lat in enumerate(cast(Iterable[float], ds["lat"].values)):
        for lon_i, lon in enumerate(cast(Iterable[float], ds["lon"].values)):
            history = cast(_SelectArray, output["sst"].sel)(
                cell=cell_id(lat, lon)
            ).isel(time=slice(0, 3))
            np.testing.assert_array_equal(
                history, np.arange(3) + lat_i * 10 + lon_i * 100
            )
            assert history.dims == ("time",)


def test_zip_shapefile_and_netcdf_pipeline_preserves_identity_across_years(
    tmp_path: Path,
) -> None:
    archive, layer = write_reef_zip(tmp_path)
    paths: list[Path] = []
    for year in [2000, 2001]:
        ds = make_sst(times=pd.date_range(f"{year}-01-01", periods=3))
        ds["sst"].values += year - 2000
        if year == 2000:
            ds["sst"].values[:, 0, 0] = np.nan
        path = tmp_path / f"sst.day.mean.{year}.nc"
        cast(Callable[[Path], None], ds.to_netcdf)(path)
        paths.append(path)
    output_dir = tmp_path / "clean"
    manifest = prepare_reef_data(paths, archive, layer, output_dir, bounds=None)
    assert isinstance(manifest, dict)
    assert (output_dir / "reef_source_LICENSE.txt").read_bytes() == (
        b"\x93No redistribution\x94\n"
    )
    with (
        cast(Callable[[Path], xr.Dataset], xr.open_dataset)(
            output_dir / "reef_sst_2000.nc"
        ) as first,
        cast(Callable[[Path], xr.Dataset], xr.open_dataset)(
            output_dir / "reef_sst_2001.nc"
        ) as second,
    ):
        np.testing.assert_array_equal(first["cell"], second["cell"])
        np.testing.assert_array_equal(first["lat"], second["lat"])
        np.testing.assert_array_equal(first["lon"], second["lon"])
        assert first.sizes == {"time": 366, "cell": 4}
        assert second.sizes == {"time": 365, "cell": 4}
        identity = cell_id(0.125, 10.375)
        np.testing.assert_array_equal(
            cast(_SelectArray, first["sst"].sel)(cell=identity)[:3], [110, 111, 112]
        )
        np.testing.assert_array_equal(
            cast(_SelectArray, second["sst"].sel)(cell=identity)[:3], [111, 112, 113]
        )
        assert (
            _numeric_item(
                cast(_SelectArray, first["source_year"].sel)(time="2000-01-01")
            )
            == 2000
        )
        assert (
            Path(
                str(
                    _text_item(
                        cast(_SelectArray, first["source_file"].sel)(time="2000-01-01")
                    )
                )
            ).name
            == paths[0].name
        )
        unavailable = first.sel(cell=cell_id(-0.125, 10.125))
        available = second.sel(cell=cell_id(-0.125, 10.125))
        assert _numeric_item(unavailable["sst"].isnull().all())
        assert _numeric_item(unavailable["sst_valid"].sum()) == 0
        assert _numeric_item(unavailable["valid_observation_count"]) == 0
        assert _numeric_item(unavailable["expected_observation_count"]) == 366
        assert _numeric_item(unavailable["valid_coverage_fraction"]) == 0
        assert _numeric_item(available["valid_observation_count"]) == 3
        assert _numeric_item(available["expected_observation_count"]) == 365
        assert _numeric_item(available["valid_coverage_fraction"]) == pytest.approx(
            3 / 365
        )
        np.testing.assert_array_equal(available["sst_valid"][:4], [1, 1, 1, 0])
        np.testing.assert_array_equal(second["source_time_present"][:4], [1, 1, 1, 0])


def test_pipeline_rejects_changed_spatial_grid_between_years(tmp_path: Path) -> None:
    archive, layer = write_reef_zip(tmp_path)
    paths: list[Path] = []
    for year, lon in [(2000, [10.125, 10.375]), (2001, [10.375, 10.625])]:
        ds = make_sst(lon=lon, times=[f"{year}-01-01"])
        path = tmp_path / f"sst.day.mean.{year}.nc"
        cast(Callable[[Path], None], ds.to_netcdf)(path)
        paths.append(path)
    with pytest.raises(ValueError):
        _ = prepare_reef_data(paths, archive, layer, tmp_path / "clean", bounds=None)


def test_dateline_hole_is_unwrapped_before_planar_repair():
    geometry = Polygon(
        [(179, -2), (-179, -2), (-179, 2), (179, 2)],
        [[(179.5, -1), (-179.5, -1), (-179.5, 1), (179.5, 1)]],
    )
    assert not geometry.is_valid
    repaired, report = repair_geometries(reefs_for(geometry))
    repaired_geometry = repaired.geometry.iloc[0]
    assert isinstance(repaired_geometry, BaseGeometry)
    result = wrap_geometry(repaired_geometry)
    assert report["invalid_before"] == report["repaired_count"] == 1
    assert result.is_valid
    assert result.area == pytest.approx(6)
    assert result.covers(box(179.1, -0.5, 179.4, 0.5))
    assert not result.covers(box(179.6, -0.5, 179.9, 0.5))


@pytest.mark.parametrize("version", ["Version 2.1", "v2.1", "2.1"])
def test_official_product_version_spellings_preserve_observations(version: str) -> None:
    ds = make_sst()
    _ = cast(str, ds.attrs.pop("product_version"))
    ds.attrs["version"] = version
    result = extract_reef_sst(ds, all_cells(ds), 2000)
    np.testing.assert_array_equal(
        cast(_SelectArray, result["sst"].sel)(cell=cell_id(0.125, 10.375))[:3],
        [110, 111, 112],
    )
    assert result.attrs["product_version_evidence"] == (
        "source metadata explicitly declares 2.1"
    )
