import json
from io import BytesIO
from pathlib import Path
from typing import IO, Union

import boto3
from botocore.client import BaseClient
from fiona.io import MemoryFile
from geopandas import GeoDataFrame
from odc.geo.xr import to_cog
from xarray import DataArray, Dataset

from pystac import Item


def object_exists(bucket: str, key: str, client: BaseClient | None = None) -> bool:
    """Check if a key exists in a bucket."""
    if client is None:
        client = boto3.client("s3")

    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError:
        return False


def s3_dump(
    data: Union[bytes, str, IO], bucket: str, key: str, client: BaseClient, **kwargs
) -> bool:
    """Write data to s3 object."""

    r = client.put_object(Bucket=bucket, Key=key, Body=data, **kwargs)
    code = r["ResponseMetadata"]["HTTPStatusCode"]
    return 200 <= code < 300


def write_to_s3(
    d: Union[DataArray, Dataset, GeoDataFrame, Item, str],
    path: Union[str, Path],
    bucket: str,
    overwrite: bool = True,
    use_odc_writer: bool = True,
    client: BaseClient | None = None,
    s3_dump_kwargs=dict(),
    **kwargs,
):
    if client is None:
        client = boto3.client("s3")

    key = str(path).lstrip("/")

    if not overwrite and object_exists(bucket, key, client):
        return

    if isinstance(d, (DataArray, Dataset)):
        if use_odc_writer:
            if "driver" in kwargs:
                del kwargs["driver"]
            binary_data = to_cog(d, **kwargs)
            s3_dump(
                binary_data,
                bucket,
                key,
                client,
                ContentType="image/tiff",
                **s3_dump_kwargs,
            )

        else:
            with BytesIO() as binary_data:
                d.rio.to_raster(binary_data, driver="COG", **kwargs)
                binary_data.seek(0)
                s3_dump(
                    binary_data,
                    bucket,
                    key,
                    client,
                    ContentType="image/tiff",
                    **s3_dump_kwargs,
                )

    elif isinstance(d, GeoDataFrame):
        with MemoryFile() as buffer:
            d.to_file(buffer, **kwargs)
            buffer.seek(0)
            s3_dump(buffer.read(), bucket, key, client, **s3_dump_kwargs)
    elif isinstance(d, Item):
        s3_dump(
            json.dumps(d.to_dict(), indent=4),
            bucket,
            key,
            client,
            ContentType="application/json",
            **s3_dump_kwargs,
        )
    elif isinstance(d, str):
        s3_dump(d, bucket, key, client, **s3_dump_kwargs)
    else:
        raise ValueError(
            "You can only write an Xarray DataArray or Dataset, Geopandas GeoDataFrame, Pystac Item, or string"
        )


def write_stac_s3(
    item: Item,
    stac_path: str,
    bucket: str,
    **kwargs,
) -> None:
    item_string = json.dumps(item.to_dict(), indent=4)
    write_to_s3(
        item_string, stac_path, bucket=bucket, ContentType="application/json", **kwargs
    )

    return stac_path
