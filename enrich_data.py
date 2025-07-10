import os
from dotenv import load_dotenv
import xarray as xr
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import glob
import plotly.express as px
import re

load_dotenv(".env.dev")


def open_nc_dataset(filename):
    '''create secure way to snatch raw data'''
    
    raw_data = os.getenv("DATA_PATH")  
    my_file = os.path.join(raw_data, filename)
    ds = xr.open_dataset(my_file)
    return ds

def open_ncs_dataset_enriched(filename):
    '''create secure way to snatch enriched data'''
    
    enriched_data = os.getenv("DATA_PATH_ENRICHED")
    my_file = os.path.join(enriched_data, filename)
    ds = xr.open_dataset(my_file)
    return ds

def open_emissions_file(filename):
    ''' so much emissions that separate function is needed'''
    
    emissions_data = os.getenv("DATA_PATH_EMISSIONS")
    my_file = os.path.join(emissions_data, filename)
    ds = xr.open_dataset(my_file)
    return ds

ds = open_nc_dataset("NOAAGlobalTemp_v6.0.0_gridded_s185001_e202505_c20250608T185757.nc")

# Shift longitude from 0–360 to -180–180 for sampling with the GeoTIFF
lons = ds['lon'].values
lons = ((lons + 180) % 360) - 180

lats = ds['lat'].values
lon2d, lat2d = np.meshgrid(lons, lats)

with rio.open('opentopodata/data/etopo1/ETOPO1_Ice_g_geotiff.tif') as src:
    coords = [(float(lon), float(lat)) for lon, lat in zip(lon2d.flatten(), lat2d.flatten())]
    elevation = np.array([v[0] for v in src.sample(coords)])  # v is a 1-element array per band
    elevations = elevation.reshape(lat2d.shape)

ds['elevation'] = (('lat', 'lon'), elevations)
ds.to_netcdf('enriched_NOAAGlobalTemp.nc', mode='w', format='NETCDF4')


ds = open_ncs_dataset_enriched('enriched_NOAAGlobalTemp.nc')


# Sort the dataset by latitude and longitude to ensure proper plotting
ds = ds.sortby(['lat', 'lon'])

#check with plot if data looks correct
elev = ds['elevation']
elev.plot(cmap='terrain', x='lon', y='lat', figsize=(12, 6))
plt.show()

emission_files = sorted(glob.glob("raw_data/TOTALS_emi_nc/EDGAR_2024_GHG_CO2_*_TOTALS_emi.nc"))

datasets =[]
for f in emission_files:
    # Extract the 4-digit year using regex
    match = re.search(r'EDGAR_2024_GHG_CO2_(\d{4})_TOTALS_emi\.nc', f)
    year = match.group(1)
    print(year) #debug
    ds = xr.open_dataset(f)
    # Add a 'time' coordinate for this file (year)
    ds = ds.expand_dims(time=[year])
    datasets.append(ds)

# Concat everything collected in datasets into one file and save
ds_emissions = xr.concat(datasets, dim='time')
ds_emissions.to_netcdf('emissions_combined_1970-2023.nc') 


