"""Utility / helper functions for using Amazon S3 Storage."""

import json
from io import BytesIO
from pathlib import Path
from typing import IO, Union

import boto3
from botocore.client import BaseClient
from fiona.io import MemoryFile
from geopandas import GeoDataFrame
from odc.geo.xr import to_cog
from pystac import Item
from xarray import DataArray, Dataset


def object_exists(bucket: str, key: str, client: BaseClient | None = None) -> bool:
    """Check if a given object exists in a bucket.

    Args:
        bucket: The name of the bucket.
        key: The object key.
        client: An optional client. If none is specified the default client is used.

    Returns:
        True if the object is in the bucket, otherwise False.

    """
    if client is None:
        client: BaseClient = boto3.client("s3")

    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError:
        return False


def s3_dump(
    data: Union[bytes, str, IO], bucket: str, key: str, client: BaseClient, **kwargs
) -> bool:
    """Write data to an s3 bucket.

    This is a wrapper around :py:func:`boto3.client.put_object`.

    Args:
        data: The data to write.
        bucket: The name of the bucket.
        key: The key in the bucket where the data should go.
        client: A client.
        **kwargs: Additional arguments to :py:func:`boto3.client.put_object`

    Returns:
        True if the operation was successful, otherwise False.

    """

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
    s3_dump_kwargs: dict = dict(),
    **kwargs,
):
    """Writes a given object to s3.

    This is a specialized version of :py:func:`s3_dump`. If the object is a GeoDataFrame,
    :py:func:`geopandas.to_file` is used. If an xarray DataArray or Dataset,
    :py:func:`odc.geo.xr.to_cog` is used if `use_odc_writer` is `True`, otherwise
    :py:func:`rioxarray.to.raster` is used. If a :py:class:`pystac.Item`, `d` is first
    dumped to json.

    Args:
        d: The data or object to write.
        path: The path (key) to write the object.
        bucket: The bucket where the object should go.
        overwrite: Whether existing objects with the same key should be overwritten.
        use_odc_writer: Whether :py:func:`odc.geo.xr.to_cog` should be used to write
            the object. If `False` :py:func:`rioxarray.to_raster` is used. Only
            matters for xarray objects (DataArrays or Datasets).
        client: The s3 client. If not set the default client is used.
        s3_dump_kwargs (): Additional arguments to :py:func:`s3_dump`.
        **kwargs: Additional arguments to the writing function, if the object is
            a :py:class:`geopandas.GeoDataFrame` or :py:class:`xarray.DataArray` /
            :py:class:`xarray.Dataset`.

    Raises:
        ValueError: If `d` is not of the allowed types.
    """
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
) -> str:
    """Writes a STAC item to s3.

    Args:
        item: A STAC item.
        stac_path: The path where the item should go.
        bucket: The bucket where the item should go.
        **kwargs: Additional arguments to :py:func:`write_to_s3`.

    Returns:
        The stac_path.

    """
    item_string = json.dumps(item.to_dict(), indent=4)
    write_to_s3(
        item_string, stac_path, bucket=bucket, ContentType="application/json", **kwargs
    )

    return stac_path


def get_s3_bucket_region(bucket_name: str, client: BaseClient | None = None) -> str:
    """Return the AWS region for a given bucket.

    Args:
        bucket_name: The name of the bucket

    Returns:
        The name of the region.

    """
    if client is None:
        client: BaseClient = boto3.client("s3")
    response = client.head_bucket(Bucket=bucket_name)
    return response["BucketRegion"]
