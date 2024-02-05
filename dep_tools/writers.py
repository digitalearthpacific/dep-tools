from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Hashable, List

import numpy as np
from pystac import Asset
from urlpath import URL
from xarray import Dataset

from .azure import blob_exists
from .namers import DepItemPath
from .processors import Processor, XrPostProcessor
from .stac_utils import write_stac_blob_storage, write_stac_local
from .utils import write_to_blob_storage, write_to_local_storage


class Writer(ABC):
    @abstractmethod
    def write(self, data, item_id) -> str:
        pass


class DepWriter(Writer):
    def __init__(
        self,
        itempath: DepItemPath,
        pre_processor: Processor,
        cog_writer: Writer,
        stac_writer: Writer | None,
    ):
        self._itempath = itempath
        self._pre_processor = pre_processor
        self._cog_writer = cog_writer
        self._stac_writer = stac_writer

    def _stac_exists(self, item_id):
        return blob_exists(self._itempath.stac_path(item_id))

    def _all_paths(self, ds: Dataset, item_id: str):
        # Only problem here is if xr has variables that aren't in the stac
        # so maybe we need to read stac?
        paths = [self._itempath.path(item_id, variable) for variable in ds]
        paths += self._itempath.stac_path(item_id)
        return paths

    def write(self, xr, item_id):
        if self._stac_exists(item_id):
            return self._all_paths(xr, item_id)
        output = self._pre_processor.process(xr)
        self._cog_writer.write(output, item_id)
        if self._stac_writer is not None:
            self._stac_writer.write(output, item_id)


class DsWriter(DepWriter):
    def __init__(
        self,
        itempath: DepItemPath,
        convert_to_int16: bool = True,
        output_value_multiplier: int = 10000,
        scale_int16s: bool = False,
        output_nodata: int = -32767,
        extra_attrs: dict = {},
        write_function: Callable = write_to_blob_storage,
        write_multithreaded: bool = True,
        write_stac: bool = True,
        write_stac_function=write_stac_blob_storage,
        load_before_write: bool = False,
        **kwargs
    ):
        pre_processor = XrPostProcessor(
            convert_to_int16,
            output_value_multiplier,
            scale_int16s,
            output_nodata,
            extra_attrs,
        )
        cog_writer = DsCogWriter(
            itempath, write_multithreaded, load_before_write, write_function, **kwargs
        )
        stac_writer = StacWriter(itempath, write_stac_function) if write_stac else None
        super().__init__(itempath, pre_processor, cog_writer, stac_writer)


class StacWriter(Writer):
    def __init__(
        self,
        itempath: DepItemPath,
        write_stac_function: Callable = write_stac_blob_storage,
    ):
        self._itempath = itempath
        self._write_stac_function = write_stac_function

    def write(self, ds: Dataset, item_id):
        paths = [self._itempath.path(item_id, variable) for variable in ds]
        assets = {
            variable: Asset(
                media_type="image/tiff; application=geotiff; profile=cloud-optimized",
                href=str(
                    URL("https://deppcpublicstorage.blob.core.windows.net/output")
                    / path
                ),
                roles=["data"],
            )
            for variable, path in zip(ds, paths)
        }
        stac_id = self._itempath.basename(item_id)
        collection = self._itempath.item_prefix
        # has to be a datetime datetime object
        datetime = np.datetime64(ds.attrs["stac_properties"]["datetime"]).item()
        stac_path = self._write_stac_function(
            ds,
            paths[0],
            stac_url=self._itempath.path(item_id, ext=".stac-item.json"),
            input_datetime=datetime,
            assets=assets,
            collection=collection,
            id=stac_id,
        )
        paths.append(stac_path)

        return paths


class DsCogWriter(Writer):
    def __init__(
        self,
        itempath: DepItemPath,
        write_multithreaded: bool = False,
        load_before_write: bool = False,
        write_function: Callable = write_to_blob_storage,
        **kwargs
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


class LocalDsCogWriter(DsWriter):
    def __init__(self, **kwargs):
        super().__init__(
            write_function=write_to_local_storage,
            write_stac_function=write_stac_local,
            **kwargs,
        )


class AzureDsWriter(DsWriter):
    pass
