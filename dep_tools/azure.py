from functools import partial
import io
from multiprocessing.dummy import Pool as ThreadPool
import os
from pathlib import Path
from typing import Union, Generator

import azure.storage.blob
from azure.storage.blob import ContainerClient
import fiona
from geopandas import GeoDataFrame
from odc.geo.xr import to_cog
from osgeo import gdal
from xarray import DataArray, Dataset


def get_container_client(
    storage_account: str = os.environ.get("AZURE_STORAGE_ACCOUNT"),
    container_name: str = "output",
    credential: str = os.environ.get("AZURE_STORAGE_SAS_TOKEN"),
) -> ContainerClient:
    if storage_account is None:
        raise ValueError(
            "'None' is not a valid value for 'storage_account'. Pass a valid name or set the 'AZURE_STORAGE_ACCOUNT' environment variable"
        )

    if credential is None:
        raise ValueError(
            "'None' is not a valid value for 'credential'. Pass a valid name or set the 'AZURE_STORAGE_SAS_TOKEN' environment variable"
        )

    return azure.storage.blob.ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )


def blob_exists(
    path: Union[str, Path], container_client: ContainerClient | None = None, **kwargs
):
    if container_client is None:
        container_client = get_container_client(**kwargs)
    blob_client = container_client.get_blob_client(str(path))
    return blob_client.exists()


def copy_to_blob_storage(
    container_client: ContainerClient,
    local_path: Path,
    remote_path: Path,
) -> None:
    with open(local_path, "rb") as src:
        blob_client = container_client.get_blob_client(str(remote_path))
        blob_client.upload_blob(src, overwrite=True)


def download_blob(container_client: ContainerClient, file: str):
    blob_client = container_client.get_blob_client(file)
    return blob_client.download_blob().readall()


def download_blobs(
    container_client: ContainerClient, blob_list: list[str], n_workers: int = 20
) -> list:
    pool = ThreadPool(n_workers)
    blobs = pool.map(partial(download_blob, container_client), blob_list)
    pool.close()
    pool.join()
    return blobs


def list_blob_container(
    container_client: ContainerClient, prefix: str, suffix: str = ".stac-item.json"
) -> Generator:
    for blob_record in container_client.list_blobs(name_starts_with=prefix):
        blob_name = blob_record["name"]
        if blob_name.endswith(suffix):
            yield blob_name


def build_vrt(
    bounds: list,
    prefix: str = "",
    suffix: str = "",
) -> Path:
    blobs = [
        f"/vsiaz/output/{blob.name}"
        for blob in get_container_client().list_blobs(name_starts_with=prefix)
        if blob.name.endswith(suffix)
    ]

    local_prefix = Path(prefix).stem
    vrt_file = f"data/{local_prefix}.vrt"
    gdal.BuildVRT(vrt_file, blobs, outputBounds=bounds)
    return Path(vrt_file)


def write_to_blob_storage(
    d: Union[DataArray, Dataset, GeoDataFrame, str],
    path: Union[str, Path],
    overwrite: bool = True,
    use_odc_writer: bool = False,
    client: ContainerClient | None = None,
    **kwargs,
) -> None:
    # Allowing for a shared container client, which might be
    # more efficient. If not provided, get one.
    if client is None:
        client = get_container_client()
    blob_client = client.get_blob_client(str(path))
    if not overwrite and blob_client.exists():
        return

    if isinstance(d, (DataArray, Dataset)):
        if use_odc_writer:
            if "driver" in kwargs:
                del kwargs["driver"]
            binary_data = to_cog(d, **kwargs)
            blob_client.upload_blob(
                binary_data, overwrite=overwrite, connection_timeout=TIMEOUT_SECONDS
            )
        else:
            with io.BytesIO() as buffer:
                # This is needed or rioxarray doesn't know what type it is writing
                if "driver" not in kwargs:
                    kwargs["driver"] = "COG"
                d.rio.to_raster(buffer, **kwargs)
                buffer.seek(0)
                blob_client.upload_blob(
                    buffer, overwrite=overwrite, connection_timeout=TIMEOUT_SECONDS
                )

    elif isinstance(d, GeoDataFrame):
        with fiona.io.MemoryFile() as buffer:
            d.to_file(buffer, **kwargs)
            buffer.seek(0)
            blob_client.upload_blob(buffer, overwrite=overwrite)
    elif isinstance(d, str):
        blob_client.upload_blob(d, overwrite=overwrite, **kwargs)
    else:
        raise ValueError(
            "You can only write an Xarray DataArray or Dataset, or Geopandas GeoDataFrame"
        )
