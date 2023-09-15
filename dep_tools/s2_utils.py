import datetime

from xarray import DataArray, concat


def scale_and_offset_s2(da: DataArray) -> DataArray:
    return harmonize_to_old(da) * 0.0001


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

    old = data.sel(time=slice(cutoff))

    to_process = list(set(bands) & set(data.band.data.tolist()))
    new = data.sel(time=slice(cutoff, None)).drop_sel(band=to_process)

    new_harmonized = data.sel(time=slice(cutoff, None), band=to_process).clip(offset)
    new_harmonized -= offset

    new = concat([new, new_harmonized], "band").sel(band=data.band.data.tolist())
    return concat([old, new], dim="time")
