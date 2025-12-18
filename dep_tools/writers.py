"""This module contains the definition and various implementations of
:class:`Writer` objects, which are used to write files locally or to the
cloud."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Hashable, List

from pystac import Item
from xarray import Dataset

from .aws import write_to_s3, write_stac_s3
from .namers import GenericItemPath, ItemPath, S3ItemPath
from .utils import write_to_local_storage


class Writer(ABC):
    """The base abstract class for writing."""

    @abstractmethod
    def write(self, *args, **kwargs) -> str | list[str]:
        pass


class DsCogWriter(Writer):

    def __init__(
        self,
        itempath: ItemPath,
        write_multithreaded: bool = False,
        load_before_write: bool = False,
        write_function: Callable = write_to_s3,
        **kwargs,
    ):
        """A :class:`Writer` object which writes an :class:`xarray.Dataset`
        to a cloud-optimized GeoTIFF (COG) file or files. A COG is written
        for each Dataset variable.

        Args:
            itempath: The :class:`ItemPath` used to define the output path.
            write_multithreaded: Whether to use multiple threads (one for
                each Dataset variable) when writing.
            load_before_write: Whether to load the Dataset before writing. This
                can be useful to prevent re-reading source data from
                when multiple variables depend on the same source.
                data.
            write_function: The function actually used to write the data. It
                should take an :class:`xarray.DataArray` as the first parameter
                and the output path (as a string or :class:`Path`) as a named
                parameter.
            **kwargs: Additional arguments to :func:`write_function`.
        """
        self._itempath = itempath
        self._write_multithreaded = write_multithreaded
        self._load_before_write = load_before_write
        self._write_function = write_function
        self._kwargs = kwargs

    def write(self, xr: Dataset, item_id: str) -> str | List:
        """Write a :class:`xarray.Dataset`.

        Args:
            xr: The Dataset.
            item_id: An item id. It is passed to the :class:`ItemPath` provided
                on initialization to determine the output file path.

        Returns: The output path(s) as a string or list of strings.
        """
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
        """A :class:`DsCogWriter` which writes to S3 storage. The only difference
        is that the ItemPath is an :class:`S3ItemPath`; `itempath.bucket` is passed
        as the `bucket` keyword argument to `write_function`.
        """
        super().__init__(
            itempath=itempath,
            write_multithreaded=write_multithreaded,
            load_before_write=load_before_write,
            write_function=write_function,
            bucket=itempath.bucket,
            **kwargs,
        )


class LocalDsCogWriter(DsCogWriter):
    """A :class:`DsCogWriter` which writes to local storage using
    :func:`dep_tools.utils.write_to_local_storage`."""

    def __init__(self, **kwargs):
        super().__init__(
            write_function=write_to_local_storage,
            **kwargs,
        )


class StacWriter(Writer):
    def __init__(
        self,
        itempath: GenericItemPath,
        write_stac_function: Callable = write_stac_s3,
        **kwargs,
    ):
        """A :class:`Write` which writes spatio-temporal asset catalog (STAC)
        Items.

        Args:
            itempath: The :class:`ItemPath` used to define the output path.
            write_stac_function: The function used to write a STAC item. It should
                take the Item as its first argument and the output path as its
                second.
            **kwargs: Additional arguments to :func:`write_stac_function`.
        """
        self._itempath = itempath
        self._write_stac_function = write_stac_function
        self._kwargs = kwargs

    def write(self, item: Item, item_id: str) -> str:
        """Write a STAC Item to an output file.

        Args:
            item: The STAC Item.
            item_id: The item id, passed to :func:`ItemPath.stac_path` to
                define the output path.

        Returns: The self reference of the item (`item.self_href`).
        """
        stac_path = self._itempath.stac_path(item_id)
        self._write_stac_function(item, stac_path, **self._kwargs)

        return item.self_href


class AwsStacWriter(StacWriter):
    def __init__(
        self,
        itempath: S3ItemPath,
        **kwargs,
    ):
        """A :class:`StacWriter` to write to Amazon S3 Storage.

        Args:
            itempath: The itempath used to define the output path. `itempath.bucket`
                is passed as the `bucket` keyword argument to
                :func:`dep_tools.aws.write_to_s3`.
            **kwargs: Additional arguments to :func:`StacWriter.__init__`.
        """
        super().__init__(
            itempath=itempath,
            write_stac_function=write_to_s3,
            bucket=itempath.bucket,
            **kwargs,
        )


class LocalStacWriter(StacWriter):
    def __init__(
        self,
        itempath: GenericItemPath,
        **kwargs,
    ):
        """A :class:`StacWriter` to write to a local file.

        This class uses :func:`dep_tools.utils.write_to_local_storage` to
        write the data. It is typically used for testing only.

        Args:
            itempath: The itempath used to define the output path.
            **kwargs: Additional arguments to :func:`StacWriter.__init__`.
        """
        super().__init__(
            itempath=itempath,
            write_stac_function=write_to_local_storage,
            **kwargs,
        )
