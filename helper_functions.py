import os
import xarray as xr
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd 


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

def load_globals(env=".env.dev"):
    """
    Load environment variables from a .env file 
    and return the correct torch device.
    """
    load_dotenv(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device
    