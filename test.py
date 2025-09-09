#!/usr/bin/env python
"""
Coral bleaching stress classification pipeline.

Computes HotSpot, Degree Heating Weeks (DHW), and NOAA-like bleaching stress
classes from daily SST fields, optionally masking to reef points.

Optimizations:
- Chunking chosen so time chunk >= rolling window (84 days)
- Vectorized classification via xr.where (no Python loops in apply_ufunc)
- Baseline climatology kept minimal (can extend to multi-year later)
- Longitude normalization applied consistently
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Tuple

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# Local imports (assumed available)
from reef_data import add_reef_mask
from helper_functions import open_ncs_dataset_enriched, open_nc_dataset  # if still needed
from css import css_formating


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROLLING_WINDOW_DAYS = 84          # DHW window length (12 weeks)
HOTSPOT_THRESHOLD_C = 1.0         # Hotspot anomaly threshold
DHW_DIVISOR = 7                   # Convert degree heating days sum to weeks
DEFAULT_CHUNKS = {"time": 90, "lat": 180, "lon": 360}  # Adjust per memory
BLEACH_CLASS_DTYPE = np.int8


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalize_lon(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Normalize longitude coordinates to [-180, 180) if currently 0..360.
    Sorts by 'lon'.
    """
    if "lon" not in ds.coords:
        return ds
    lon = ds.lon
    if lon.max() > 180:
        ds = ds.assign_coords(lon=((lon + 180) % 360) - 180).sortby("lon")
    return ds


def open_sst_dataset(path: str,
                     chunks: Optional[Dict[str, int]] = None,
                     decode_times: bool = True) -> xr.Dataset:
    """
    Open daily SST NetCDF dataset with optional chunking.
    """
    ds = xr.open_dataset(path, chunks=chunks, decode_times=decode_times)
    ds = normalize_lon(ds)
    return ds


def compute_baseline_climatology(baseline_ds: xr.Dataset) -> xr.DataArray:
    """
    Compute maximum monthly mean (MMM) climatology from a baseline dataset.

    NOTE: If baseline_ds contains multiple years, this collapses them
    via groupby("time.month").mean("time") then takes max across months.
    """
    sst_base = baseline_ds["sst"]
    monthly = sst_base.groupby("time.month").mean("time")
    max_monthly_mean = monthly.max("month")
    return max_monthly_mean


def compute_hotspot_and_dhw(
    sst_target: xr.DataArray,
    max_monthly_mean_baseline: xr.DataArray,
    time_chunk_aligned: bool = True
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute HotSpot and DHW arrays.

    DHW = rolling sum (over 84 days) of positive HotSpot exceeding 1°C threshold,
          divided by 7 (days/week).
    """
    hotspot = sst_target - max_monthly_mean_baseline
    hotspot_thresh = hotspot.where(hotspot > HOTSPOT_THRESHOLD_C, 0)

    # Ensure appropriate chunking for time dimension
    if not time_chunk_aligned and hasattr(hotspot_thresh.data, "chunk"):
        # Rechunk so rolling sees full window within a chunk
        hotspot_thresh = hotspot_thresh.chunk({"time": ROLLING_WINDOW_DAYS})

    # Rolling window (set min_periods = full window for DHW validity mask)
    rolling_sum = (
        hotspot_thresh
        .rolling(time=ROLLING_WINDOW_DAYS, min_periods=1)
        .sum()
    ) / DHW_DIVISOR

    full_window_mask = (
        hotspot_thresh
        .rolling(time=ROLLING_WINDOW_DAYS, min_periods=ROLLING_WINDOW_DAYS)
        .count() == ROLLING_WINDOW_DAYS
    )

    dhw = rolling_sum.where(full_window_mask, 0)
    return hotspot, dhw


def classify_bleaching(hotspot: xr.DataArray,
                       dhw: xr.DataArray) -> xr.DataArray:
    """
    NOAA-like bleaching stress classes:

    0: No stress (default)
    1: Watch            (0 < HotSpot < 1)  (though some definitions start at >= 1; your original kept 0<HS<1)
    2: Warning          (HotSpot >=1, DHW < 4)
    3: Alert Level 1    (DHW 4–8)
    4: Alert Level 2?   (DHW 8–12)   (Your original increments)
    5: 12–16
    6: 16–20
    7: >=20
    """
    hs = hotspot
    d = dhw

    # Start with zeros
    bleach = xr.zeros_like(hs, dtype=BLEACH_CLASS_DTYPE)

    # Class 1: watch (0 < hotspot < 1)
    bleach = xr.where((hs > 0) & (hs < 1), 1, bleach)

    # For hotspot >= 1, base becomes 2 (Warning)
    mask_hot = hs >= 1
    bleach = xr.where(mask_hot & (d < 4), 2, bleach)

    bleach = xr.where(mask_hot & (d >= 4)  & (d < 8), 3, bleach)
    bleach = xr.where(mask_hot & (d >= 8)  & (d < 12), 4, bleach)
    bleach = xr.where(mask_hot & (d >= 12) & (d < 16), 5, bleach)
    bleach = xr.where(mask_hot & (d >= 16) & (d < 20), 6, bleach)
    bleach = xr.where(mask_hot & (d >= 20), 7, bleach)

    bleach.name = "bleaching_stress_class"
    return bleach


def add_bleaching_variables(
    target_ds: xr.Dataset,
    baseline_ds: xr.Dataset
) -> xr.Dataset:
    """
    Enrich target_ds with:
      - hotspot
      - dhw
      - bleaching_stress_class
    """
    max_monthly_mean = compute_baseline_climatology(baseline_ds).persist()

    hotspot, dhw = compute_hotspot_and_dhw(
        target_ds["sst"],
        max_monthly_mean,
        time_chunk_aligned=True
    )
    bleaching = classify_bleaching(hotspot, dhw)

    target_ds["hotspot"] = hotspot
    target_ds["dhw"] = dhw
    target_ds["bleaching_stress_class"] = bleaching
    return target_ds


def extract_reef_only(ds: xr.Dataset,
                      bleaching_var: str = "bleaching_stress_class") -> xr.DataArray:
    """
    Return bleaching classes only at reef points.

    Assumes add_reef_mask added a boolean variable 'reef_points' (True at reefs).
    """
    if "reef_points" not in ds:
        raise KeyError("Dataset missing 'reef_points' mask.")
    return ds[bleaching_var].where(ds["reef_points"] == True)


def plot_bleaching(
    ds: xr.Dataset,
    time_index: int = -1,
    var: str = "bleaching_stress_class",
    reef_only: bool = False,
    figsize: Tuple[int, int] = (12, 5),
    cmap: str = "turbo",
    vmin: int = 0,
    vmax: int = 7
):
    """
    Simple pcolormesh plot for a selected time slice.
    """
    if reef_only:
        data = extract_reef_only(ds, bleaching_var=var).isel(time=time_index)
    else:
        data = ds[var].isel(time=time_index)

    plt.figure(figsize=figsize)
    mesh = plt.pcolormesh(
        data["lon"],
        data["lat"],
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    cbar = plt.colorbar(mesh, label=var.replace("_", " ").title())
    plt.title(f"{var.replace('_',' ').title()} (time index {time_index})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------

def run_pipeline(
    target_sst_path: str = "sst.day.mean.2020.nc",
    baseline_sst_path: str = "sst_data/sst.day.mean.1985.nc",
    apply_reef_mask: bool = True,
    reef_only_return: bool = False,
    chunks: Dict[str, int] = None,
    compute_now: bool = False,
    do_plot: bool = True
):
    """
    Execute the bleaching classification pipeline.

    Parameters
    ----------
    target_sst_path : str
        Daily SST file for the analysis year.
    baseline_sst_path : str
        Baseline reference period file (can be single or multi-year).
    apply_reef_mask : bool
        Whether to add reef mask (uses add_reef_mask).
    reef_only_return : bool
        If True, only keep reef cells in final dataset (drops others).
    chunks : dict
        Dask chunk mapping. Defaults to DEFAULT_CHUNKS.
    compute_now : bool
        If True, triggers immediate computation (.compute()) of key variables.
    do_plot : bool
        If True, plot final bleaching class for last timestep.
    """
    if chunks is None:
        chunks = DEFAULT_CHUNKS

    # Load datasets
    baseline_ds = open_sst_dataset(baseline_sst_path, chunks=chunks)
    if apply_reef_mask:
        # Assume add_reef_mask internally uses xr.open_dataset; adapt to pass chunks if possible.
        target_ds = add_reef_mask(filename=target_sst_path,
                                  reef_only_return=False)  # modify add_reef_mask to accept chunks if needed
        # Ensure chunking on target
        for var in target_ds.data_vars:
            if hasattr(target_ds[var].data, "chunks"):
                continue
        # If not chunked, optionally chunk here (soft fallback)
        target_ds = target_ds.chunk(chunks)
    else:
        target_ds = open_sst_dataset(target_sst_path, chunks=chunks)

    # Normalize lon alignment
    target_ds = normalize_lon(target_ds)

    # Add bleaching vars
    target_ds = add_bleaching_variables(target_ds, baseline_ds)

    # Optionally reduce to reefs only
    if reef_only_return:
        reef_mask = target_ds["reef_points"] == True
        target_ds = target_ds.where(reef_mask)

    if compute_now:
        # Trigger realization of core variables
        target_ds[["hotspot", "dhw", "bleaching_stress_class"]].load()

    if do_plot:
        plot_bleaching(target_ds, time_index=-1, reef_only=False)

    return target_ds


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------

def main():
    load_dotenv(".env.dev")
    css_formating()

    ds = run_pipeline(
        target_sst_path="sst.day.mean.2020.nc",
        baseline_sst_path="sst_data/sst.day.mean.1985.nc",
        apply_reef_mask=True,
        reef_only_return=False,
        chunks=DEFAULT_CHUNKS,
        compute_now=False,   # Set True to force immediate computation
        do_plot=True
    )

    print(ds)
    print("Variables:", list(ds.data_vars))

    # Example: compute reef-only bleaching subset
    reef_bleaching = extract_reef_only(ds)
    print("Reef bleaching dask graph size:",
          getattr(reef_bleaching.data, "__dask_graph__", lambda: {})().__len__()
          if hasattr(reef_bleaching.data, "__dask_graph__") else "Not dask")


if __name__ == "__main__":
    main()