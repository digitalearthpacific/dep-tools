import os
from pathlib import Path
from typing import Union, Generator

import azure.storage.blob
from azure.storage.blob import ContainerClient
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool


def get_container_client(
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    container_name: str = "output",
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
) -> ContainerClient:
    return azure.storage.blob.ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )


def blob_exists(path: Union[str, Path], **kwargs):
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
