from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Hashable, List

from pystac import Asset, MediaType
from urlpath import URL
from xarray import Dataset, DataArray

from pathlib import Path

from datetime import datetime

from .azure import blob_exists
from .namers import DepItemPath
from .processors import Processor, XrPostProcessor
from .stac_utils import write_stac_blob_storage, write_stac_local
from .utils import write_to_blob_storage, write_to_local_storage
from .aws import write_to_s3, object_exists, write_stac_aws
from botocore.client import BaseClient
from rio_stac.stac import create_stac_item
import json
from pystac import Item


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
        overwrite: bool = False,
        **kwargs,
    ):
        self._itempath = itempath
        self._pre_processor = pre_processor
        self._cog_writer = cog_writer
        self._stac_writer = stac_writer
        self._overwrite = overwrite

    def _stac_exists(self, item_id):
        return blob_exists(self._itempath.stac_path(item_id))

    def _all_paths(self, ds: Dataset, item_id: str):
        # Only problem here is if xr has variables that aren't in the stac
        # so maybe we need to read stac?
        paths = [self._itempath.path(item_id, variable) for variable in ds]
        paths.append(self._itempath.stac_path(item_id))
        return paths

    def write(self, xr, item_id):
        # Check if the STAC doc exists, and quit early if it does
        if not self._overwrite and self._stac_exists(item_id):
            return self._all_paths(xr, item_id)

        # Process the data and write it to COGs
        output = self._pre_processor.process(xr)
        paths = self._cog_writer.write(output, item_id)

        # If a STAC writer is provided, write the STAC doc
        if self._stac_writer is not None:
            stac_path = self._stac_writer.write(output, item_id)
            paths.append(stac_path)

        return paths


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
            StacWriter(itempath, write_stac_function, **kwargs) if write_stac else None
        )
        super().__init__(itempath, pre_processor, cog_writer, stac_writer, **kwargs)


class StacWriter(Writer):
    def __init__(
        self,
        itempath: DepItemPath,
        write_stac_function: Callable = write_stac_blob_storage,
        bucket: str | None = None,
        overwrite: bool = True,
        client: BaseClient | None = None,
        **kwargs,
    ):
        self._itempath = itempath
        self._write_stac_function = write_stac_function
        self._bucket = bucket
        self._overwrite = overwrite
        self._client = client
        self._kwargs = kwargs

    def get_stac_item(
        self,
        item_id: str,
        data: DataArray | Dataset,
        collection: str,
        remote: bool = True,
        collection_url_root: str = "https://stac.staging.digitalearthpacific.org/collections",
        bucket: str = None,
        **kwargs,
    ) -> Item:
        prefix = Path("./")
        # Remote means not local
        # TODO: neaten local file writing up
        if remote:
            if bucket is not None:
                # Writing to S3
                prefix = URL(f"s3://{bucket}")
            else:
                # Default to Azure
                prefix = URL("https://deppcpublicstorage.blob.core.windows.net/output")

        properties = {}
        if "stac_properties" in data.attrs:
            properties = (
                json.loads(data.attrs["stac_properties"].replace("'", '"'))
                if isinstance(data.attrs["stac_properties"], str)
                else data.attrs["stac_properties"]
            )

        paths = [self._itempath.path(item_id, variable) for variable in data]

        assets = {
            variable: Asset(
                media_type=MediaType.COG,
                href=str(prefix / path),
                roles=["data"],
            )
            for variable, path in zip(data, paths)
        }
        stac_id = self._itempath.basename(item_id)
        collection = self._itempath.item_prefix
        collection_url = f"{collection_url_root}/{collection}"

        input_datetime = properties.get("datetime", None)
        if input_datetime is not None:
            input_datetime = datetime.strptime(input_datetime, "%Y-%m-%dT%H:%M:%S.000Z")

        item = create_stac_item(
            str(prefix / paths[0]),
            id=stac_id,
            input_datetime=input_datetime,
            assets=assets,
            with_proj=True,
            properties=properties,
            collection_url=collection_url,
            collection=collection,
            **kwargs,
        )

        stac_url = str(prefix / self._itempath.stac_path(item_id))
        item.set_self_href(stac_url)

        return item

    def write(self, ds: Dataset, item_id, **kwargs):
        item = self.get_stac_item(
            item_id, ds, self._itempath.item_prefix, bucket=self._bucket
        )
        stac_path = self._itempath.stac_path(item_id)
        self._write_stac_function(item, stac_path, bucket=self._bucket, **kwargs)

        return item.self_href


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


class LocalDsCogWriter(DsWriter):
    def __init__(self, **kwargs):
        super().__init__(
            write_function=write_to_local_storage,
            write_stac_function=write_stac_local,
            **kwargs,
        )


class AwsDsCogWriter(DsWriter):
    # This really runs a DepWriter
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
    # This really runs a DepWriter
    pass
