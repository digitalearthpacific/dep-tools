from datetime import datetime
import json
from pathlib import Path

from azure.storage.blob import ContentSettings
import numpy as np
from pystac import Asset, Item, MediaType
from rio_stac.stac import create_stac_item
from urlpath import URL
from xarray import DataArray, Dataset


from .namers import DepItemPath
from .processors import Processor
from .azure import write_to_blob_storage


class StacCreator(Processor):
    def __init__(
        self,
        itempath: DepItemPath,
        remote: bool = True,
        collection_url_root: str = "https://stac.staging.digitalearthpacific.org/collections",
        bucket: str | None = None,
    ):
        self._itempath = itempath
        self._remote = remote
        self._collection_url_root = collection_url_root
        self._bucket = bucket

    def process(
        self,
        data: DataArray | Dataset,
        item_id: str,
        **kwargs,
    ) -> Item:
        return get_stac_item(
            self._itempath,
            item_id,
            data,
            self._collection,
            self._remote,
            self._collection_url_root,
            self._bucket,
            **kwargs,
        )


def get_stac_item(
    itempath: DepItemPath,
    item_id: str,
    data: DataArray | Dataset,
    collection: str,
    remote: bool = True,
    collection_url_root: str = "https://stac.staging.digitalearthpacific.org/collections",
    bucket: str | None = None,
    **kwargs,
) -> Item:
    prefix = Path("./")
    # Remote means not local
    # TODO: neaten local file writing up
    if remote:
        if bucket is not None:
            # Writing to S3
            prefix = URL(f"s3://{bucket}")
        else:
            # Default to Azure
            prefix = URL("https://deppcpublicstorage.blob.core.windows.net/output")

    properties = {}
    if "stac_properties" in data.attrs:
        properties = (
            json.loads(data.attrs["stac_properties"].replace("'", '"'))
            if isinstance(data.attrs["stac_properties"], str)
            else data.attrs["stac_properties"]
        )

    paths = [itempath.path(item_id, variable) for variable in data]

    assets = {
        variable: Asset(
            media_type=MediaType.COG,
            href=str(prefix / path),
            roles=["data"],
        )
        for variable, path in zip(data, paths)
    }
    stac_id = itempath.basename(item_id)
    collection = itempath.item_prefix
    collection_url = f"{collection_url_root}/{collection}"

    input_datetime = properties.get("datetime", None)
    if input_datetime is not None:
        input_datetime = datetime.strptime(input_datetime, "%Y-%m-%dT%H:%M:%S.000Z")

    item = create_stac_item(
        str(prefix / paths[0]),
        id=stac_id,
        input_datetime=input_datetime,
        assets=assets,
        with_proj=True,
        properties=properties,
        collection_url=collection_url,
        collection=collection,
        **kwargs,
    )

    stac_url = str(prefix / itempath.stac_path(item_id))
    item.set_self_href(stac_url)

    return item


def write_stac_blob_storage(
    item: Item,
    stac_path: str,
    **kwargs,
) -> str | None:
    item_json = json.dumps(item.to_dict(), indent=4)
    write_to_blob_storage(
        item_json,
        stac_path,
        content_settings=ContentSettings(content_type="application/json"),
        **kwargs,
    )
    return stac_path


def write_stac_local(item: Item, stac_path: str, **kwargs) -> None:
    with open(stac_path, "w") as f:
        json.dump(item.to_dict(), f, indent=4)


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
