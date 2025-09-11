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
from reef_data import merge_ds
from reef_data import save
from dask.distributed import Client


#### CONFIG ####

# For the DHW calculation logic you need to have two SST datasets in yearly order. Calculations will run for the younger filename (FILENAME_2)

FILENAME_1 = "sst_data/sst.day.mean.2018.nc"   # Support filename used to calculate DHW for the first 84 days of main filename
FILENAME_2 = "sst_data/sst.day.mean.2019.nc"   # Main filename 

# Dask config
# Minimal Requirements to run below script in memory: 32GB RAM

N_WORKERS = 8                    #  Worker per physical core
THREADS_PER_WORKER = 2           #  Threads per worker
MEMORY_LIMIT = "4GB"             #  Divide RAM among workers
PROCESSES = True                 #  Separate processes for workers




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

def extract_84_days_of_ds(filename=FILENAME_1):
    """
    Extract 84 days of dataset that are used to calculate
    dhw for the first 84 days of the next year's dataset
    """
    
    ds = xr.open_dataset(filename, chunks={"lat": 1440, "lon": 720, "time": -1})
    
    last_day = ds.time.max().item()
    last_day = pd.to_datetime(last_day, unit="ns")
    start_day = pd.Timestamp(last_day) - pd.Timedelta(days=83)
    ds = ds.sel(time=slice(start_day, last_day))
    
    return ds

def combine_datasets_in_memory():
    
    ds_previous_year = extract_84_days_of_ds()
    ds_year = xr.open_dataset(FILENAME_2, chunks={"lat": 1440, "lon":720, "time": -1})
    files = [ds_previous_year, ds_year]
    ds_merged = xr.combine_by_coords(files, combine_attrs="override")
    
    return ds_merged


def calculate_bleaching_bins(ds):
    """
    Calculate hotspot, DHW, and bleaching stress classes on SST dataset.
    """
    ds_baseline = (
    xr.open_dataset("baseline_datasets/merged_baseline.nc", chunks={"time": -1, "lat": 1440, "lon": 720})
      .pipe(normalize_lon)
    )
    sst_baseline = ds_baseline["sst"]
    # Monthly climatology (mean for each calendar month)
    monthly_mean_baseline = sst_baseline.groupby('time.month').mean(dim='time')
    # max of the 12 monthly means
    max_monthly_mean_baseline = monthly_mean_baseline.max(dim='month')
        
        
    ds = normalize_lon(ds)
    sst = ds["sst"]

    # Hotspot anomaly above max monthly mean
    hotspot = sst - max_monthly_mean_baseline
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

    cls = xr.where(hotspot <= 0, 0, cls)
    cls = xr.where((hotspot > 0) & (hotspot < 1), 1, cls)
    cls = xr.where((hotspot >= 1) & (dhw > 0) & (dhw < 4), 2, cls)
    cls = xr.where((hotspot >= 1) & (dhw >= 4) & (dhw <= 8), 3, cls)
    cls = xr.where((hotspot >= 1) & (dhw > 8), 4, cls)
    
    ds = ds.sel(time="2018") # ///// change it later to variable 
    return ds.assign(
        hotspot=hotspot,
        dhw=dhw,
        bleaching_stress_class=cls
    )


def bleaching_bins(ds, nclasses=5):
    """
    Calculate class value for lat x lon x time dims
    """
    data_array = ds.to_numpy()
    data_array = data_array[~np.isnan(ds)]
    data_array = data_array.astype(int)
    
    ds_bins = np.bincount(data_array.ravel(), minlength=5)
    
    total = ds_bins.sum().item()
    
    for i in range(nclasses):
        prct = (ds_bins[i] / total) * 100
        print(f"Class {i}: {prct:.2f}%")

def unique_bleaching_bins(ds, nclasses=5):
    """
    Calculate the maximum class value of each pixel over time and plot results
    as histogram
    """
    
    ds = ds.max(dim='time')
    data_array = ds.to_numpy().astype(int)
    data_array = data_array[data_array >= 0]
    
    ds_bins = np.bincount(data_array, minlength=nclasses)
    
    total = ds_bins.sum().item()
    percentages = (ds_bins / total) * 100

    # Print percentages for each class
    for i, prct in enumerate(percentages):
        print(f"Class {i}: {prct:.2f}%")
        
    plt.bar(range(nclasses), percentages, color='skyblue', edgecolor='black')
    plt.xlabel('Class')
    plt.ylabel('Percentage of Pixels')
    plt.title('Percentage of max class reached during the year at unique point')
    plt.show()

def main():
    
    try: client.close()
    except: pass

    client = Client(
            n_workers=N_WORKERS,                
            threads_per_worker=THREADS_PER_WORKER,       
            memory_limit=MEMORY_LIMIT,         
            processes=PROCESSES
            )
    
    print(client)
    
    


if __name__ == "__main__":
    #main()
    
    ds_baseline = merge_ds(path="baseline_datasets",
                           dim='time')
    
    save(ds=ds_baseline,
         year=None,
         name="merged_baseline.nc",
         path="baseline_datasets")
    del ds_baseline
    
    ds_merged = combine_datasets_in_memory()
    
    reef_ds = add_reef_mask(
        filename=None,
        reef_only_return=False,
        ds=ds_merged
        ).chunk({"lat": 1440, "lon": 720, "time":-1})
    del ds_merged
    reef_ds = calculate_bleaching_bins(ds=reef_ds)
    bleaching_on_reefs = reef_ds["bleaching_stress_class"].where(reef_ds["reef_points"] == True)
    #del reef_ds
    
    
    
    bleaching_bins(bleaching_on_reefs)
    unique_bleaching_bins(bleaching_on_reefs)
        
        

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
    
    
    
