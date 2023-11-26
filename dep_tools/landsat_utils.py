from typing import Dict, Iterable, Tuple

import planetary_computer
import pystac_client
from odc.algo import erase_bad, mask_cleanup
from pystac import ItemCollection
from retry import retry
from xarray import DataArray


def mask_clouds(
    xr: DataArray, filters: Iterable[Tuple[str, int]] | None = None, keep_ints: bool = False
) -> DataArray:
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


@retry(tries=10, delay=1)
def item_collection_for_pathrow(
    path: int, row: int, search_args: Dict
) -> ItemCollection:
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    return catalog.search(
        **search_args,
        query=[
            #           f"landsat:wrs_path={path:03d}",
            #           f"landsat:wrs_row={row:03d}",
        ],
    ).item_collection()
