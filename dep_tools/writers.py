from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Hashable, List

from botocore.client import BaseClient
from xarray import Dataset

from .azure import blob_exists, write_to_blob_storage, write_stac_blob_storage
from .aws import write_to_s3, object_exists, write_stac_aws
from .namers import DepItemPath, S3ItemPath
from .processors import Processor, XrPostProcessor
from .stac_utils import write_stac_local
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
        write_function: Callable = write_to_blob_storage,
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


# This should be the AwsDsCogWriter as the one below subclasses DsWriter
class RealAwsDsCogWriter(DsCogWriter):
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


class StacWriter(Writer):
    def __init__(
        self,
        itempath: DepItemPath,
        write_stac_function: Callable = write_stac_blob_storage,
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


# Everything below here should be candidate for deletion
class DepWriter(Writer):
    """A writer which writes xarray Datasets to cloud-optimized GeoTiffs.
    It requires a pre-processor and optionally creates and writes a STAC
    item."""

    def __init__(
        self,
        itempath: DepItemPath,
        pre_processor: Processor,
        cog_writer: Writer,
        stac_writer: Writer | None,
        overwrite: bool = False,
    ):
        self._itempath = itempath
        self._pre_processor = pre_processor
        self._cog_writer = cog_writer
        self._stac_writer = stac_writer
        self._overwrite = overwrite

    def _stac_exists(self, item_id) -> bool:
        return blob_exists(self._itempath.stac_path(item_id))

    def _all_paths(self, ds: Dataset, item_id: str) -> list:
        # Only problem here is if xr has variables that aren't in the stac
        # so maybe we need to read stac?
        paths = [self._itempath.path(item_id, variable) for variable in ds]
        paths.append(self._itempath.stac_path(item_id))
        return paths

    def write(self, ds: Dataset, item_id) -> list | list[str]:
        # Check if the STAC doc exists, and quit early if it does
        if not self._overwrite and self._stac_exists(item_id):
            return self._all_paths(ds, item_id)

        # Process the data and write it to COGs
        output = self._pre_processor.process(ds)
        paths = self._cog_writer.write(output, item_id)

        # If a STAC writer is provided, write the STAC doc
        if self._stac_writer is not None:
            stac_path = self._stac_writer.write(output, item_id)
            if isinstance(paths, str):
                paths = [paths]
            assert isinstance(stac_path, str)
            paths.append(stac_path)

        return paths


class DsWriter(DepWriter):
    """A DepWriter with common parameters for DEP assets. If you
    need more fine-grained control over the individual pieces, then use
    a DepWriter."""

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
        overwrite: bool = False,
        **kwargs,
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
        stac_writer = (
            StacWriter(itempath, write_stac_function=write_stac_function)
            if write_stac
            else None
        )
        super().__init__(itempath, pre_processor, cog_writer, stac_writer, overwrite)


class LocalDsCogWriter(DsWriter):
    def __init__(self, **kwargs):
        super().__init__(
            write_function=write_to_local_storage,
            write_stac_function=write_stac_local,
            **kwargs,
        )


class AwsDsCogWriter(DsWriter):
    def __init__(self, bucket: str, client: BaseClient | None = None, **kwargs):
        super().__init__(
            write_function=write_to_s3,
            write_stac_function=write_stac_aws,
            bucket=bucket,
            client=client,
            **kwargs,
        )
        self._bucket = bucket

    def _stac_exists(self, item_id):
        return object_exists(self._bucket, self._itempath.stac_path(item_id))


class AzureDsWriter(DsWriter):
    pass
