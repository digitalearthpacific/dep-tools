import io
from pathlib import Path
from typing import Dict, Union

import boto3
from geopandas import GeoDataFrame
from odc.geo.xr import write_cog
from xarray import DataArray, Dataset


def write_to_s3(
    d: Union[DataArray, Dataset, str],
    path: Union[str, Path],
    write_args: Dict = dict(),
    overwrite: bool = True,
    use_odc_writer: bool = True,
    **kwargs,
) -> None:
    path_str = str(path)
    # Path should be like `s3://bucket/path/to/file.tif`
    bucket, path = path_str.split("/", 3)[2:]

    if isinstance(d, (DataArray, Dataset)):
        if use_odc_writer:
            if "driver" in write_args:
                del write_args["driver"]
            write_cog(d, path_str, overwrite=overwrite, **write_args)
        else:
            with io.BytesIO() as buffer:
                d.rio.to_raster(buffer, **write_args)
                buffer.seek(0)

                # Write with boto3 directly
                s3 = boto3.client("s3")
                s3.upload_fileobj(
                    buffer,
                    bucket,
                    path,
                    ExtraArgs={"ACL": "public-read"},
                )
    elif isinstance(d, GeoDataFrame):
        raise NotImplementedError("GeoDataFrame writing not yet implemented")
    elif isinstance(d, str):
        # Write a string to the blob storage
        s3 = boto3.resource("s3")
        s3.Object(bucket, path).put(Body=d)
    else:
        raise ValueError(
            "You can only write an Xarray DataArray or Dataset, or Geopandas GeoDataFrame"
        )
