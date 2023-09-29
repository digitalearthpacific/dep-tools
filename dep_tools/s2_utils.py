import datetime

from odc.algo import mask_cleanup
from xarray import DataArray, concat

from .processors import Processor


class S2Processor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = True,
        mask_clouds: bool = True,
        dilate_mask: bool = False,
    ) -> None:
        super().__init__(send_area_to_processor)
        self.scale_and_offset = scale_and_offset
        self.mask_clouds = mask_clouds
        self.dilate_mask = dilate_mask

    def process(self, xr: DataArray) -> DataArray:
        if self.mask_clouds:
            xr = xr.where(~clouds(xr.sel(band="SCL"), self.dilate_mask))

        if self.scale_and_offset:
            xr = scale_and_offset_s2(xr)

        return xr


def scale_and_offset_s2(da: DataArray) -> DataArray:
    # TODO: don't scale SCL
    return harmonize_to_old(da) * 0.0001


def clouds(scl: DataArray, dilate: bool = True) -> DataArray:
    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW = 11

    clouds = (scl == CLOUD_SHADOWS) | (scl == CLOUD_HIGH_PROBABILITY)

    if dilate:
        clouds = mask_cleanup(clouds, [("opening", 2), ("dilation", 3)])

    return clouds


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
