import datetime

from odc.algo import erase_bad, mask_cleanup
from xarray import DataArray, concat

from .processors import Processor


class S2Processor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = True,
        mask_clouds: bool = True,
        dilate_mask: bool = False,
        keep_ints: bool = False,
    ) -> None:
        super().__init__(send_area_to_processor)
        self.scale_and_offset = scale_and_offset
        self.mask_clouds = mask_clouds
        self.dilate_mask = dilate_mask
        self.keep_ints = keep_ints

    def process(self, xr: DataArray) -> DataArray:
        if self.mask_clouds:
            xr = mask_clouds(xr, dilate=self.dilate_mask, keep_ints=self.keep_ints)

        if self.scale_and_offset:
            xr = scale_and_offset_s2(xr)

        return xr


def scale_and_offset_s2(da: DataArray) -> DataArray:
    # TODO: don't scale SCL
    return harmonize_to_old(da) * 0.0001


def mask_clouds(
    xr: DataArray, dilate: bool = True, keep_ints: bool = False
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

    if dilate:
        # From Alex @ https://gist.github.com/alexgleith/d9ea655d4e55162e64fe2c9db84284e5
        cloud_mask = mask_cleanup(cloud_mask, [("opening", 2), ("dilation", 3)])

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
