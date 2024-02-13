import datetime
from typing import Iterable, Tuple

from odc.algo import erase_bad, mask_cleanup
from xarray import DataArray, concat


def mask_clouds(
    xr: DataArray,
    filters: Iterable[Tuple[str, int]] | None = None,
    keep_ints: bool = False,
) -> DataArray:
    # NO_DATA = 0
    # SATURATED_OR_DEFECTIVE = 1
    # DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    # VEGETATION = 4
    # NOT_VEGETATED = 5
    # WATER = 6
    # UNCLASSIFIED = 7
    # CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    # THIN_CIRRUS = 10
    # SNOW = 11

    bitmask = 0
    for field in [CLOUD_SHADOWS, CLOUD_HIGH_PROBABILITY]:
        bitmask |= 1 << field

    try:
        cloud_mask = xr.sel(band="SCL").astype("uint16") & bitmask != 0
    except KeyError:
        cloud_mask = xr.SCL.astype("uint16") & bitmask != 0

    if filters is not None:
        cloud_mask = mask_cleanup(cloud_mask, filters)

    if keep_ints:
        return erase_bad(xr, cloud_mask)
    else:
        return xr.where(~cloud_mask)


def harmonize_to_old(data: DataArray) -> DataArray:
    """
    Harmonize new Sentinel-2 data to the old baseline. Taken from https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change

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
    cutoff = datetime.datetime(2022, 1, 25)

    # Do nothing if the data is all before the cutoff
    latest_datetime = data.time.max().values.astype("M8[ms]").astype("O")
    if latest_datetime < cutoff:
        return data

    offset = 1000
    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]

    # Check if the data has the band dimension
    data_has_band_dim = "band" in data.dims

    # Store the old data, it doesn't need offsetting
    old = data.sel(time=slice(cutoff))
    # Get the data after the cutoff
    new = data.sel(time=slice(cutoff, None))

    if data_has_band_dim:
        to_process = list(set(bands) & set(data.band.data.tolist()))
        new = new.drop_sel(band=to_process)

        new_harmonized = data.sel(time=slice(cutoff, None), band=to_process).clip(
            offset
        )
        new_harmonized -= offset

        new = concat([new, new_harmonized], "band").sel(band=data.band.data.tolist())
    else:
        # Handle SCL being there or not
        new_scl = None
        if "SCL" in new:
            new_scl = new.SCL

        new_data = new.drop_vars("SCL", errors="ignore")

        # Now clip to 1000, so any values lower than that are lost, and then offset
        new_data = new_data.clip(offset)
        new_data -= offset

        # Combine
        if new_scl is not None:
            new_data["SCL"] = new_scl

        new = new_data

    out_data = concat([old, new], dim="time")

    return out_data
