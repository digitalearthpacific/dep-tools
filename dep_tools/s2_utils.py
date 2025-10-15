"""This module contains useful functions for working with Sentinel-2 data."""

import datetime
from typing import Iterable, Tuple

from odc.algo import erase_bad, mask_cleanup
from xarray import DataArray, concat


def mask_clouds(
    xr: DataArray,
    filters: Iterable[Tuple[str, int]] | None = None,
    keep_ints: bool = False,
    return_mask: bool = False,
) -> DataArray:
    """Mask Sentinel-2 data using the `"SCL"` band, with optional filters.

    The following classes are masked:
        - `"SATURATED_OR_DEFECTIVE"` (SCL value 1)
        - `"CLOUD_SHADOWS"` (3)
        - `"CLOUD_MEDIUM_PROBABILITY"` (8)
        - `"CLOUD_HIGH_PROBABILITY"` (9)
        - `"THIN_CIRRUS"` (10)

    Args:
        xr: Input Sentinel-2 data.
        filters: Filters to apply, passed to :func:`odc.algo.mask_cleanup`.
        keep_ints: If True, data is kept as input (typically integer) data
            type, and masking is performed using :func:`odc.algo.erase_bad`.
        return_mask: Whether to return the mask itself along with the data.

    Returns:
        If `return_mask` is `False`, the input data is returned, with the
        specified masking applied, If `True`, then a tuple of `(<data>, <mask>)`.
    """
    # NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    # DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    # VEGETATION = 4
    # NOT_VEGETATED = 5
    # WATER = 6
    # UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    # SNOW = 11

    cloud_mask = xr.scl.isin(
        [
            SATURATED_OR_DEFECTIVE,
            CLOUD_SHADOWS,
            CLOUD_MEDIUM_PROBABILITY,
            CLOUD_HIGH_PROBABILITY,
            THIN_CIRRUS,
        ]
    )

    if filters is not None:
        cloud_mask = mask_cleanup(cloud_mask, filters)

    if keep_ints:
        masked = erase_bad(xr, cloud_mask)
    else:
        masked = xr.where(~cloud_mask)

    if return_mask:
        return masked, cloud_mask
    else:
        return masked


def harmonize_to_old(data: DataArray) -> DataArray:
    """
    Harmonize new Sentinel-2 data to the old baseline.
    Inspired by https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change

    Parameters
    ----------
    data: xarray.DataArray
        A DataArray with four dimensions: time, band, y, x

    Returns
    -------
    harmonized: xarray.DataArray
        A DataArray with all values harmonized to the old
        processing baseline.

    """
    CUTOFF = datetime.datetime(2022, 1, 25)
    OFFSET = 1000

    # Do nothing if the data is all before the CUTOFF
    latest_datetime = data.time.max().values.astype("M8[ms]").astype("O")
    if latest_datetime < CUTOFF:
        return data

    # Check if the data has the band dimension
    data_has_band_dim = "band" in data.dims

    # Handle the case where the data has the band dimension
    if data_has_band_dim:
        # Remove the bands dimension
        data = data.to_dataset(dim="band").drop_dims("band")

    # Store the old data, it doesn't need offsetting
    old = data.sel(time=slice(CUTOFF))
    # Get the data after the CUTOFF
    new_unoffset = data.sel(time=slice(CUTOFF, None))

    # Handle SCL being there or not
    new_scl = None
    if "SCL" in new_unoffset:
        new_scl = new_unoffset.SCL

    new = new_unoffset.drop_vars("SCL", errors="ignore")

    # Now clip to 1000, so any values lower than that are lost
    # (otherwise we get integer overflows) and then offset
    new = new.clip(OFFSET)
    new -= OFFSET

    # Combine with the SCL band if it was there
    if new_scl is not None:
        new["SCL"] = new_scl

    out_data = concat([old, new], dim="time")

    if data_has_band_dim:
        out_data = out_data.to_array(dim="band")

    return out_data
