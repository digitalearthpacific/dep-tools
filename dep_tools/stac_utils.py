from datetime import datetime
import json
from pathlib import Path

import numpy as np
from pandas import DataFrame
from pystac import Asset, Item, MediaType, read_dict
import pystac_client
import rasterio
from rio_stac.stac import create_stac_item, get_raster_info
from xarray import DataArray, Dataset


from .aws import object_exists
from .namers import GenericItemPath, S3ItemPath
from .processors import Processor


class StacCreator(Processor):
    def __init__(
        self,
        itempath: GenericItemPath,
        remote: bool = True,
        collection_url_root: str = "https://stac.staging.digitalearthpacific.io/collections",
        **kwargs,
    ):
        self._itempath = itempath
        self._remote = remote
        self._collection_url_root = collection_url_root
        self._kwargs = kwargs

    def process(
        self,
        data: DataArray | Dataset,
        item_id: str,
    ) -> Item | str:
        return get_stac_item(
            itempath=self._itempath,
            item_id=item_id,
            data=data,
            remote=self._remote,
            collection_url_root=self._collection_url_root,
            **self._kwargs,
        )


def get_stac_item(
    itempath: GenericItemPath,
    item_id: str,
    data: DataArray | Dataset,
    collection_url_root: str = "https://stac.staging.digitalearthpacific.org/collections",
    **kwargs,
) -> Item | str:

    properties = {}
    if "stac_properties" in data.attrs:
        properties = (
            json.loads(data.attrs["stac_properties"].replace("'", '"'))
            if isinstance(data.attrs["stac_properties"], str)
            else data.attrs["stac_properties"]
        )

    assets = {}
    for variable in data:
        raster_info = {}
        full_path = itempath.path(item_id, variable, absolute=True)
        if kwargs.get("with_raster"):
            with rasterio.open(full_path) as src_dst:
                raster_info = {"raster:bands": get_raster_info(src_dst, max_size=1024)}

        assets[variable] = Asset(
            media_type=MediaType.COG,
            href=full_path,
            roles=["data"],
            extra_fields={**raster_info},
        )
    stac_id = itempath.basename(item_id)
    collection = itempath.item_prefix
    collection_url = f"{collection_url_root}/{collection}"

    input_datetime = properties.get("datetime", None)
    if input_datetime is not None:
        format_string = (
            "%Y-%m-%dT%H:%M:%S.%fZ" if "." in input_datetime else "%Y-%m-%dT%H:%M:%SZ"
        )
        input_datetime = datetime.strptime(input_datetime, format_string)

    an_href = next(iter(assets.values())).href
    item = create_stac_item(
        an_href,
        id=stac_id,
        input_datetime=input_datetime,
        assets=assets,
        with_proj=True,
        properties=properties,
        collection_url=collection_url,
        collection=collection,
        **kwargs,
    )

    item.set_self_href(itempath.stac_path(item_id, absolute=True))

    return item


def write_stac_local(item: Item, stac_path: str) -> None:
    with open(stac_path, "w") as f:
        json.dump(item.to_dict(), f, indent=4)


def existing_stac_items(possible_ids: list, itempath: S3ItemPath) -> list:
    """Returns only those ids which have an existing stac item."""
    return [
        id
        for id in possible_ids
        if object_exists(itempath.bucket, itempath.stac_path(id))
    ]


def remove_items_with_existing_stac(grid: DataFrame, itempath: S3ItemPath) -> DataFrame:
    """Filter a dataframe to only include items which don't have an existing stac output.
    The dataframe must have an index which corresponds to ids for the given itempath.
    """
    return grid[~grid.index.isin(existing_stac_items(list(grid.index), itempath))]


def set_stac_properties(
    input_xr: DataArray | Dataset, output_xr: DataArray | Dataset
) -> Dataset | DataArray:
    """Sets an attribute called "stac_properties" in the output which is a
    dictionary containing the following properties for use in stac writing:
    "start_datetime", "end_datetime", "datetime", and "created". These are
    set from the input_xr.time coordinate. Typically, `input_xr` would be
    an array of EO data (e.g. Landsat) containing data over a range of
    dates (such as a year).
    """
    start_datetime = np.datetime_as_string(
        np.datetime64(input_xr.time.min().values, "Y"), unit="ms", timezone="UTC"
    )

    end_datetime = np.datetime_as_string(
        np.datetime64(input_xr.time.max().values, "Y")
        + np.timedelta64(1, "Y")
        - np.timedelta64(1, "s"),
        timezone="UTC",
    )
    output_xr.attrs["stac_properties"] = dict(
        start_datetime=start_datetime,
        datetime=start_datetime,
        end_datetime=end_datetime,
        created=np.datetime_as_string(np.datetime64(datetime.now()), timezone="UTC"),
    )

    return output_xr


def copy_stac_properties(item: Item, ds: Dataset) -> Dataset:
    """Copy item's properties to ds["stac_properties"], merging/updating any existing
    values. Also sets `ds.attrs["stac_properties"]["start_datetime"]` and "end_datetime"
    to item.properties["datetime"].
    """
    ds.attrs["stac_properties"] = (
        item.properties.copy()
        if "stac_properties" not in ds.attrs
        else {
            **ds.attrs["stac_properties"],
            **item.properties,
        }
    )
    ds.attrs["stac_properties"]["start_datetime"] = ds.attrs["stac_properties"][
        "datetime"
    ]
    ds.attrs["stac_properties"]["end_datetime"] = ds.attrs["stac_properties"][
        "datetime"
    ]
    return ds


def use_alternate_s3_href(modifiable: pystac_client.Modifiable) -> None:
    """Fixes the urls in the official USGS Landsat Stac Server
    (https://landsatlook.usgs.gov/stac-server) so the alternate (s3)
    urls are used. Can be used like

    client = pystac_client.Client.open(
        "https://landsatlook.usgs.gov/stac-server",
        modifier=use_alternate_s3_href,
    )

    Modifies in place, according to best practices from
    https://pystac-client.readthedocs.io/en/stable/usage.html#automatically-modifying-results.
    """
    if isinstance(modifiable, dict):
        if modifiable["type"] == "FeatureCollection":
            new_features = list()
            for item_dict in modifiable["features"]:
                use_alternate_s3_href(item_dict)
                new_features.append(item_dict)
            modifiable["features"] = new_features
        else:
            stac_object = read_dict(modifiable)
            use_alternate_s3_href(stac_object)
            modifiable.update(stac_object.to_dict())
    else:
        for _, asset in modifiable.assets.items():
            asset_dict = asset.to_dict()
            if "alternate" in asset_dict.keys():
                asset.href = asset.to_dict()["alternate"]["s3"]["href"]
