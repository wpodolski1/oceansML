"""Rasterio 1.4 affine construction interface used by reef_data."""

from affine import Affine

def from_origin(west: float, north: float, xsize: float, ysize: float) -> Affine: ...
