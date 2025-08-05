import xarray as xr
from css import css_formating
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


css_formating()

dataset = xr.open_dataset('enriched_data/dataset_temps_elev_emis_1971-2023')    

dataset['time'] = pd.to_datetime(dataset['time'])

dataframe = dataset.to_dataframe().reset_index()
dataframe.dropna(inplace=True)


