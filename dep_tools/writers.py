from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Hashable, List

from xarray import Dataset

from .aws import write_to_s3, write_stac_s3
from .namers import DepItemPath, S3ItemPath
from .utils import write_to_local_storage


class Writer(ABC):
    """The base abstract class for writing."""

    @abstractmethod
    def write(self, data, item_id) -> str | list[str]:
        pass


class DsCogWriter(Writer):
    def __init__(
        self,
        itempath: DepItemPath,
        write_multithreaded: bool = False,
        load_before_write: bool = False,
        write_function: Callable = write_to_s3,
        **kwargs,
    ):
        self._itempath = itempath
        self._write_multithreaded = write_multithreaded
        self._load_before_write = load_before_write
        self._write_function = write_function
        self._kwargs = kwargs

    def write(self, xr: Dataset, item_id: str) -> str | List:
        if self._load_before_write:
            xr.load()

        paths = []

        def get_write_partial(variable: Hashable) -> Callable:
            output_da = xr[variable].squeeze()
            path = self._itempath.path(item_id, variable)
            paths.append(path)

            return partial(
                self._write_function,
                output_da,
                path=path,
                **self._kwargs,
            )

        if self._write_multithreaded:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(get_write_partial(variable)) for variable in xr
                ]
                for future in futures:
                    future.result()
        else:
            for variable in xr:
                get_write_partial(variable)()
        return paths


class AwsDsCogWriter(DsCogWriter):
    def __init__(
        self,
        itempath: S3ItemPath,
        write_multithreaded: bool = False,
        load_before_write: bool = False,
        write_function: Callable = write_to_s3,
        **kwargs,
    ):
        super().__init__(
            itempath=itempath,
            write_multithreaded=write_multithreaded,
            load_before_write=load_before_write,
            write_function=write_function,
            bucket=itempath.bucket,
            **kwargs,
        )


class LocalDsCogWriter(DsCogWriter):
    def __init__(self, **kwargs):
        super().__init__(
            write_function=write_to_local_storage,
            **kwargs,
        )


class StacWriter(Writer):
    def __init__(
        self,
        itempath: DepItemPath,
        write_stac_function: Callable = write_stac_s3,
        **kwargs,
    ):
        self._itempath = itempath
        self._write_stac_function = write_stac_function
        self._kwargs = kwargs

    def write(self, item, item_id) -> str:
        stac_path = self._itempath.stac_path(item_id)
        self._write_stac_function(item, stac_path, **self._kwargs)

        return item.self_href


class AwsStacWriter(StacWriter):
    def __init__(
        self,
        itempath: S3ItemPath,
        **kwargs,
    ):
        super().__init__(
            itempath=itempath,
            write_stac_function=write_to_s3,
            bucket=itempath.bucket,
            **kwargs,
        )


class LocalStacWriter(StacWriter):
    def __init__(
        self,
        itempath: S3ItemPath,
        **kwargs,
    ):
        super().__init__(
            itempath=itempath,
            write_stac_function=write_to_local_storage,
            **kwargs,
        )
