"""Rasterio 1.4 in-memory uint8 rasterization used by reef_data.

The published types-rasterio distribution requires Python >=3.12 and Rasterio
>=1.5. This partial interface follows Rasterio 1.4's features.rasterize contract;
file output and other dtypes are intentionally not declared.
"""

from collections.abc import Iterable
from typing import Literal

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

def rasterize(
    shapes: Iterable[tuple[BaseGeometry, int]],
    out_shape: tuple[int, int],
    fill: float = 0,
    *,
    transform: Affine = ...,
    all_touched: bool = False,
    dtype: Literal["uint8"],
) -> NDArray[np.uint8]: ...
