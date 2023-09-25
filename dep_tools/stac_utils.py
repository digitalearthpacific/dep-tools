import datetime
import json
from urlpath import URL
from typing import Union

from azure.storage.blob import ContentSettings
import numpy as np
from pystac import Item
from rio_stac.stac import create_stac_item
from xarray import DataArray, Dataset

from .utils import write_to_blob_storage, write_to_local_storage


def write_stac(
    xr: Union[DataArray, Dataset],
    path: str,
    stac_url,
    writer=write_to_blob_storage,
    **kwargs,
) -> None:
    item = _get_stac_item(xr, path, **kwargs)
    item_json = json.dumps(item.to_dict(), indent=4)
    writer(
        item_json,
        stac_url,
        write_args=dict(
            content_settings=ContentSettings(content_type="application/json")
        ),
    )


def write_stac_local(xr: Union[DataArray, Dataset], path: str, stac_url, **kwargs):
    write_stac(
        xr, path, stac_url, writer=write_to_local_storage, remote=False, **kwargs
    )


write_stac_blob_storage = write_stac


def _get_stac_item(
    xr: Union[DataArray, Dataset],
    path: str,
    collection: str,
    remote: bool = True,
    **kwargs,
) -> Item:
    az_prefix = URL("https://deppcpublicstorage.blob.core.windows.net/output")
    blob_url = az_prefix / path if remote else path
    properties = {}
    if "stac_properties" in xr.attrs:
        properties = (
            json.loads(xr.attrs["stac_properties"].replace("'", '"'))
            if isinstance(xr.attrs["stac_properties"], str)
            else xr.attrs["stac_properties"]
        )

    collection_url = (
        f"https://stac.staging.digitalearthpacific.org/collections/{collection}"
    )
    return create_stac_item(
        str(blob_url),
        asset_roles=["data"],
        with_proj=True,
        properties=properties,
        collection_url=collection_url,
        collection=collection,
        **kwargs,
    )


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
        created=np.datetime_as_string(
            np.datetime64(datetime.datetime.now()), timezone="UTC"
        ),
    )

    return output_xr
