from typing import Dict, List

import geopandas as gpd
import planetary_computer
import pystac_client
from pystac import ItemCollection
import xarray as xr
from xarray import DataArray


def mask_clouds(xr: DataArray) -> DataArray:
    # dilated cloud, cirrus, cloud, cloud shadow
    mask_bitfields = [1, 2, 3, 4]
    bitmask = 0
    for field in mask_bitfields:
        bitmask |= 1 << field

    qa = xr.sel(band="qa_pixel").astype("uint16")
    return xr.where(qa & bitmask == 0)


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
            f"landsat:wrs_path={path:03d}",
            f"landsat:wrs_row={row:03d}",
        ],
    ).item_collection()
