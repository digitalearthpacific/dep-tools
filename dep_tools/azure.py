from azure.storage.blob import ContainerClient
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool


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


def list_blob_container(container_client: ContainerClient, prefix: str, suffix: str = ".stac-item.json") -> list:
    for blob_record in container_client.list_blobs(name_starts_with=prefix):
        blob_name = blob_record["name"]
        if blob_name.endswith(suffix):
            yield blob_name
