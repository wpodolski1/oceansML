import os
from collections.abc import Callable
from typing import cast

import torch
import xarray as xr
from dotenv import load_dotenv

# xarray accepts filesystem paths; its broader backend signature is untyped.
_open_dataset = cast(Callable[[str], xr.Dataset], xr.open_dataset)


def open_nc_dataset(filename: str) -> xr.Dataset:
    """Open raw data; raise KeyError if DATA_PATH is unset."""
    raw_data = os.environ["DATA_PATH"]
    my_file = os.path.join(raw_data, filename)
    ds = _open_dataset(my_file)
    return ds


def open_ncs_dataset_enriched(filename: str) -> xr.Dataset:
    """Open enriched data; raise KeyError if DATA_PATH_ENRICHED is unset."""
    enriched_data = os.environ["DATA_PATH_ENRICHED"]
    my_file = os.path.join(enriched_data, filename)
    ds = _open_dataset(my_file)
    return ds


def open_emissions_file(filename: str) -> xr.Dataset:
    """Open emissions data; raise KeyError if DATA_PATH_EMISSIONS is unset."""
    emissions_data = os.environ["DATA_PATH_EMISSIONS"]
    my_file = os.path.join(emissions_data, filename)
    ds = _open_dataset(my_file)
    return ds


def load_globals(env: str = ".env.dev") -> torch.device:
    """
    Load environment variables from a .env file
    and return the correct torch device.
    """
    _ = load_dotenv(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device
