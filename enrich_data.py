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


load_dotenv(".env.dev")
css_formating()

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

def check_lons_lats_min_max_values(filename, variable):
    '''explore dataset before concating'''
    
    ref_ds = xr.open_dataset(filename[0])
    ref_lat = ref_ds['lat'].values
    ref_lon = ref_ds['lon'].values

    for f in filename:
        ds = xr.open_dataset(f)
        lat_equal = np.array_equal(ds['lat'].values, ref_lat)
        lon_equal = np.array_equal(ds['lon'].values, ref_lon)
        print(f, 'lat_equal', lat_equal, 'lon_equal', lon_equal)
        
    min_max = []
    for f in filename:
        ds = xr.open_dataset(f)
        values = ds[variable].values
        min_max.append(values.flatten())
        
    min_max = np.concatenate(min_max)
        
    print(f"Max val: {np.nanmax(min_max)}, Min val: {np.nanmin(min_max)}")
        

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
ds.to_netcdf('enriched_data/enriched_NOAAGlobalTemp.nc', mode='w', format='NETCDF4')

# open enriched datasets
ds = open_ncs_dataset_enriched('enriched_NOAAGlobalTemp.nc')


# Sort the dataset by latitude and longitude to ensure proper plotting
ds = ds.sortby(['lat', 'lon'])

#check with plot if data looks correct
elev = ds['elevation']
elev.plot(cmap='terrain', x='lon', y='lat', figsize=(12, 6))
plt.show()

# open emission raw datasets
emission_files = sorted(glob.glob("raw_data/TOTALS_emi_nc/EDGAR_2024_GHG_CO2_*_TOTALS_emi.nc"))


# check if lats and longs align in all files
check_lons_lats_min_max_values(emission_files, 'emissions')

# concat emission data
datasets =[]
for f in emission_files:
    # Extract the 4-digit year using regex
    match = re.search(r'EDGAR_2024_GHG_CO2_(\d{4})_TOTALS_emi\.nc', f)
    year = match.group(1)
    print(year) #debug
    ds = xr.open_dataset(f)
    # Add a 'time' coordinate for this file (year)
    ds = ds.expand_dims(time=[pd.Timestamp(f"{year}-01-01")])
    datasets.append(ds)
    
ref_lat = datasets[0]['lat']
ref_lon = datasets[0]['lon']
aligned_datasets = [ds.assign_coords(lat=ref_lat, lon=ref_lon) for ds in datasets]

# Concat everything collected in datasets into one file and save
ds_emissions = xr.concat(aligned_datasets, dim='time')
ds_emissions.to_netcdf('enriched_data/emissions_combined_1970-2023.nc')

# open raw temp datasets
temp_files = sorted(glob.glob("raw_data/cru_ts4.09.*.*.tmp.dat.nc"))

# check if lats and longs align in all files
check_lons_lats_min_max_values(temp_files,'tmp')


datasets = []
for f in temp_files:
   ds = xr.open_dataset(f)
   datasets.append(ds)
   
ds_temps = xr.concat(datasets, dim='time')
ds_temps.to_netcdf('temps_combined_1970-2024')


#idk doesnt work yet