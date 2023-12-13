from typing import Dict, Iterable, Tuple

import geopandas as gpd
from odc.algo import erase_bad, mask_cleanup
from xarray import DataArray, Dataset


def mask_clouds(
    xr: DataArray | Dataset,
    filters: Iterable[Tuple[str, int]] | None = None,
    keep_ints: bool = False,
) -> DataArray | Dataset:
    """
    Mask clouds in Landsat data.

    Args:
        xr: DataArray containing Landsat data including the `qa_pixel` band.
        filters: List of filters to apply to the cloud mask. Each filter is a tuple of
            (filter name, filter size). Valid filter names are 'opening' and 'dilation'.
            If None, no filters will be applied.
            For example: [("closing", 10),("opening", 2),("dilation", 2)]
        keep_ints: If True, return the masked data as integers. Otherwise, return
            the masked data as floats.
    """
    CLOUD = 3
    CLOUD_SHADOW = 4

    bitmask = 0
    for field in [CLOUD, CLOUD_SHADOW]:
        bitmask |= 1 << field

    try:
        cloud_mask = xr.sel(band="qa_pixel").astype("uint16") & bitmask != 0
    except KeyError:
        cloud_mask = xr.qa_pixel.astype("uint16") & bitmask != 0

    if filters is not None:
        cloud_mask = mask_cleanup(cloud_mask, filters)

    if keep_ints:
        return erase_bad(xr, cloud_mask)
    else:
        return xr.where(~cloud_mask)


def pathrow_with_greatest_area(shapes: gpd.GeoDataFrame) -> Tuple[str, str]:
    pathrows = gpd.read_file(
        "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
    )
    intersection = shapes.overlay(pathrows, how="intersection")
    row_with_greatest_area = intersection.iloc[[intersection.geometry.area.idxmax()]]
    return (row_with_greatest_area.PATH.item(), row_with_greatest_area.ROW.item())
