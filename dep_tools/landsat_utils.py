from typing import Dict, Tuple

import planetary_computer
import pystac_client
from odc.algo import erase_bad, mask_cleanup
from pystac import ItemCollection
from retry import retry
from xarray import DataArray


def mask_clouds(
    xr: DataArray, dilate: Tuple[int, int] | None = None, keep_ints: bool = False
) -> DataArray:
    # dilated cloud, cirrus, cloud, cloud shadow
    mask_bitfields = [1, 2, 3, 4]
    bitmask = 0
    for field in mask_bitfields:
        bitmask |= 1 << field

    try:
        cloud_mask = xr.sel(band="qa_pixel").astype("uint16") & bitmask != 0
    except KeyError:
        cloud_mask = xr.qa_pixel.astype("uint16") & bitmask != 0

    if dilate is not None:
        opening = dilate[0]
        dilation = dilate[1]
        cloud_mask = mask_cleanup(cloud_mask, [("opening", opening), ("dilation", dilation)])

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
