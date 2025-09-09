"""
reef_data.py

This module provides functions to enrich SST (Sea Surface Temperature) dataset from NOAA (National Oceanid and Atmospheric Administration)
with:
coral reef gridded dataset from:
bleaching dataset from:
and extracts reef points for further analysis.

Functions:
- add_reef_mask(filename): Adds a coral reef mask to the SST dataset and saves the result.
- add_bleaching_and_create_reefpoints(filename): Adds bleaching event data and extracts reef points as an XArray DataSet.

Author: Wojciech Podolski
Date: 2025-08-28
"""

from dotenv import load_dotenv
import xarray as xr
import numpy as np
import re
from helper_functions import open_nc_dataset
import geopandas as gpd
from rasterio import features
from affine import Affine
from shapely.geometry import box
from pathlib import Path
import os
import pandas as pd
import gc

gc.collect()
load_dotenv(".env.dev")

 
DATA_PATH_SST = Path(os.getenv("DATA_PATH_SST"))

def open_nc_dataset(filename):
    '''create secure way to snatch raw data'''

    my_file = os.path.join(DATA_PATH_SST, filename)
    ds = xr.open_dataset(my_file, engine='netcdf4')
    return ds


def add_reef_mask(filename, reef_only_return: bool):
    """
    Adds a coral reef mask to the SST NetCDF file and saves the augmented dataset.

    Parameters:
        filename (str): Path to the input NetCDF SST file.

    Returns:
        None. The augmented dataset is saved as 'reef_sst_{year}.nc' in the 'reef_sst' directory.
    """
    
    # Open SST dataset
    ds_tmps = open_nc_dataset(filename)
    # Open coral reef file (polygons)
    reefs = gpd.read_file("raw_data/01_Data/WCMC008_CoralReef2018_Py_v4_1.shp")
        
    # Normalize longitude to (-180, 180]
    lons = ds_tmps['lon']
    lons = ((lons + 180) % 360) - 180
    ds_tmps['lon'] = lons

    # Sort to align properly
    ds_tmps = ds_tmps.sortby('lat')
    ds_tmps = ds_tmps.sortby('lon')

    # Use SST dataset's lat/lon directly
    lon = ds_tmps['lon'].values
    lat = ds_tmps['lat'].values

    # Check orientation — rasterio expects top-left first
    if lat[0] < lat[-1]:
        lat_raster = lat[::-1]  # reverse so top (max lat) is first
    else:
        lat_raster = lat

    # Affine transform: pixel size in degrees, origin at top-left
    transform = Affine(
        lon[1] - lon[0],  # pixel width
        0,
        lon.min(),
        0,
        lat_raster[1] - lat_raster[0],  # pixel height (negative if lat is descending)
        lat_raster.max()
    )

    # Prepare shapes from reef polygons (
    reefs_projected = reefs.to_crs(epsg=4326)  # ensure WGS84
    shapes = [(geom, 1) for geom in reefs_projected.geometry]

    # Rasterize directly to SST grid shape
    reef_mask_array = features.rasterize(
        shapes=shapes,
        out_shape=(len(lat_raster), len(lon)),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='uint8'
    )

    # If lat was reversed for rasterization, reverse back
    if lat[0] < lat[-1]:
        reef_mask_array = reef_mask_array[::-1, :]

    # Create DataArray aligned to SST grid
    reef_mask_da = xr.DataArray(
        reef_mask_array,
        coords={'lat': ds_tmps['lat'], 'lon': ds_tmps['lon']},
        dims=('lat', 'lon'),
        name='reef_mask'
    )
    
    reef_points_mask = xr.DataArray(
    reef_mask_da.astype(bool).values,  # convert to boolean
    coords={'lat': ds_tmps['lat'], 'lon': ds_tmps['lon']},
    dims=('lat', 'lon'),
    name='reef_points'
    )
    # Assign reef mask as feature into SST 
    
    reef_only = ds_tmps['sst'].where(reef_mask_da).stack(points=("lat", "lon")).dropna("points")
    
    if reef_only_return == True:
        return reef_only
    else:  
        ds_out = ds_tmps.copy()
        ds_out['reef_points'] = reef_points_mask
        return ds_out


def merge_ds(path="reef_sst", dim='points'):
    """
    Merge nc datasets into one file
    """
    
    PATH = Path(path)
    files = [PATH / f for f in os.listdir(PATH)]
    
    datasets = []
    for f in files:
        ds = xr.open_dataset(f)
        datasets.append(ds)
    
    merged = xr.combine_by_coords(datasets, combine_attrs="override")
    
    return merged
    
def save(ds, year):
    """
    Save to local dir
    """
    
    if 'points' in ds.dims and isinstance(ds.indexes['points'], pd.MultiIndex):
        ds = ds.reset_index('points')
    
    # Define path
    DATA_PATH = Path("reef_sst")
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    DATA_NAME = f"reef_sst_{year}.nc"
    DATA_FULL_PATH = DATA_PATH / DATA_NAME
    
    # Save ds to netcdf 
    ds.to_netcdf(DATA_FULL_PATH)
        
    


if __name__ == "__main__":
   
    files = sorted(DATA_PATH_SST.glob("*.nc"))
    
    for f in files:
        
        
        filename = str(f) 
        # Regex year extraction
        match = re.search(r'(\d{4})', filename)
        if match:
            year = match.group(1)
        else:
            raise ValueError("Year not found in filename.")
        
        if not (DATA_PATH_SST / f"reef_sst_{year}.nc").exists():
            ds = add_reef_mask(filename, reef_only_return=True)
            
            print(ds)
            
            save(ds, year)
    
      
    merged = merge_ds(path="reef_sst",
                        dim="time")
    
    merged.to_netcdf("raw_data/merged_reef_data.nc")

        

    