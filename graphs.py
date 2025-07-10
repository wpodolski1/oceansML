import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd

ds = xr.open_dataset('NOAAGlobalTemp_v6.0.0_gridded_s185001_e202505_c20250608T185757.nc')
print(ds)
anom = ds['anom'].isel(z=0)
print(anom)

slice_df = anom.isel(time=0).to_dataframe().reset_index()
print(slice_df.head())

latest_anom = anom.isel(time=-1)
plt.figure(figsize=(12,6))
latest_anom.plot(cmap='coolwarm')
plt.title('Global temp anomaly')
plt.show()

df = anom.stack().to_dataframe().reset_index()
df = df.dropna()  # Remove any NaN values

df['time'] = pd.to_datetime(df['time'])

#df_decadal = df[df['time'].dt.year <1980] # decadal data, too big to animate in memory
#df_yearly = df[df['time'].dt.year >= 1980] # up from 1980 yearly data to accurately show changes

df['time'] = df['time'].dt.year
df_yearly_means = df.groupby(['time', 'lat', 'lon'])['anom'].mean().reset_index()

max_anom = np.max(df_yearly_means['anom'])
min_anom = np.min(df_yearly_means['anom'])

fig = px.scatter_geo(
    df_yearly_means,
    lat='lat', lon='lon',
    color='anom',
    animation_frame='time',
    color_continuous_scale='RdBu_r',
    projection='natural earth',
    title='Global Surface Temperature Anomalies Over Time',
    range_color=(min_anom, max_anom),
    height=600
)
fig.show()

