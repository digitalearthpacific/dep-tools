import datetime
import json

import numpy as np
from pystac import Item
from xarray import DataArray, Dataset

from azure.storage.blob import ContentSettings

from .utils import write_to_blob_storage


def write_stac_blob_storage(
    item: Item,
    stac_path: str,
    **kwargs,
) -> None:
    item_json = json.dumps(item.to_dict(), indent=4)
    write_to_blob_storage(
        item_json,
        stac_path,
        content_settings=ContentSettings(content_type="application/json"),
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
        created=np.datetime_as_string(
            np.datetime64(datetime.datetime.now()), timezone="UTC"
        ),
    )

    return output_xr
