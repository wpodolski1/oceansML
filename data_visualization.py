import os
from dotenv import load_dotenv
import xarray as xr
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import glob
import plotly.express as px
import re
import pandas as pd
from css import css_formating
from helper_functions import open_ncs_dataset_enriched, open_nc_dataset
import geopandas as gpd
from shapely.ops import transform
from rasterio import features
from affine import Affine
from matplotlib.patches import Patch
from reef_data import add_reef_mask

# Load environment and apply CSS formatting
load_dotenv(".env.dev")
css_formating()

def normalize_lon(ds):
    """
    Re-center longitude coordinates from [0, 360) to [-180, 180).
    """
    if ds.lon.max() > 180:
        ds = ds.assign_coords(
            lon=(((ds.lon + 180) % 360) - 180)
        ).sortby("lon")
    return ds

# Pre-load and preprocess baseline climatology once


def calculate_bleaching_bins(ds):
    """
    Calculate hotspot, DHW, and bleaching stress classes on an SST dataset.
    """
    _ds0 = (
    xr.open_dataset("sst_data/sst.day.mean.1985.nc")
      .pipe(normalize_lon)
    )
    sst0 = _ds0["sst"]
    monthly_mean_baseline = sst0.groupby("time.month").mean("time")
    max_mmb = monthly_mean_baseline.max(dim="month")
        
    ds = normalize_lon(ds)
    sst = ds["sst"]

    # Hotspot anomaly above max monthly mean
    hotspot = sst - max_mmb
    hotspot = hotspot.chunk({"time": 30})

    # Degree Heating Weeks (DHW) with full 84-day coverage requirement
    coverage = hotspot.rolling(time=84, min_periods=84).count()
    dhw = (
        hotspot.where(hotspot > 0)
               .rolling(time=84)
               .sum()
        / 7
    ).where(coverage == 84, 0)

    # Build stress class via chained where() calls
    cls = xr.zeros_like(hotspot, dtype=int)
    cls = xr.where((hotspot > 0) & (hotspot < 1), 1, cls)
    cls = xr.where((hotspot >= 1) & (dhw < 4), 2, cls)
    cls = xr.where((hotspot >= 1) & (dhw >= 4) & (dhw < 8), 3, cls)
    cls = xr.where((hotspot >= 1) & (dhw >= 8) & (dhw < 12), 4, cls)
    cls = xr.where((hotspot >= 1) & (dhw >= 12) & (dhw < 16), 5, cls)
    cls = xr.where((hotspot >= 1) & (dhw >= 16) & (dhw < 20), 6, cls)
    cls = xr.where((hotspot >= 1) & (dhw >= 20), 7, cls)
    
    return ds.assign(
        hotspot=hotspot,
        dhw=dhw,
        bleaching_stress_class=cls
    )


def bleaching_bins(ds):
    
    data_array = ds.to_numpy()
    data_array = data_array[~np.isnan(ds)]
    data_array = data_array.astype(int)
    
    ds_bins = np.bincount(data_array.ravel(), minlength=8)
    
    total = ds_bins.sum().item()
    
    for i in range(len(ds_bins)):
        prct = (ds_bins[i] / total) * 100
        print(f"Class {i}: {prct:.2f}%")

def unique_bleaching_bins(ds):
    
    data_array = ds.to_numpy()
    

    lat, lon = data_array.shape[1], data_array.shape[2]
    
    num_classes=8
    
    # Initialize boolean array to mark if class appeared at each point
    appeared = np.zeros((num_classes, lat, lon), dtype=bool)

    for c in range(num_classes):
        # Check if class c appeared at each point over time
        appeared[c] = np.any(data_array == c, axis=0)

    # Count points per class
    counts = appeared.sum(axis=(1, 2))
    total_points = counts.sum()  # total points where any class appeared
    percentages = (counts / total_points) * 100

    # Print results
    for i, prct in enumerate(percentages):
        print(f"Class {i}: {prct:.2f}% of points saw this class at least once")

    
def check_check(ds):
    
    ds = ds.max(dim='time')
    data_array = ds.to_numpy()
    data_array = data_array.astype(int)
    
    ds_bins = np.bincount(data_array.ravel(), minlength=8)

    total = ds_bins.sum().item()
    
    for i in range(len(ds_bins)):
        prct = (ds_bins[i] / total) * 100
        print(f"Class {i}: {prct:.2f}%")
    
    
if __name__ == "__main__":
    reef_ds = add_reef_mask(
        filename="sst.day.mean.2017.nc",
        reef_only_return=False
    )
    reef_ds = calculate_bleaching_bins(ds=reef_ds)
    bleaching_on_reefs = reef_ds["bleaching_stress_class"].where(reef_ds["reef_points"] == True)
    
    check_check(bleaching_on_reefs)
    
    # bleaching_bins(bleaching_on_reefs)
    # unique_bleaching_bins(bleaching_on_reefs)
        
        

    world = gpd.read_file("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson")
    
    fig, ax = plt.subplots(figsize=(12,10))
    world.plot(ax=ax, color='lightgray', edgecolor='lightgray')
    bleaching_on_reefs.mean(dim='time').plot(
        x='lon', y='lat',
        cmap='turbo',
        vmin=0, vmax=7,
        cbar_kwargs={'label': 'Bleaching Stress Class'}
        
    )
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Map of Bleaching Stress Class on Coral Reefs')
    plt.show()
    
    
    
