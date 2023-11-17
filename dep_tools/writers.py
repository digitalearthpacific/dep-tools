from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union

import numpy as np
from pystac import Asset
from urlpath import URL
from xarray import DataArray, Dataset

from .azure import get_container_client
from .namers import DepItemPath
from .stac_utils import write_stac_blob_storage, write_stac_local
from .utils import scale_to_int16, write_to_blob_storage, write_to_local_storage


class Writer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def write(self, data, id) -> str:
        pass


@dataclass
class XrWriterMixin(object):
    itempath: DepItemPath
    overwrite: bool = False
    convert_to_int16: bool = True
    output_value_multiplier: int = 10000
    scale_int16s: bool = False
    output_nodata: int = -32767
    extra_attrs: Dict = field(default_factory=dict)
    use_odc_writer: bool = True

    def prep(self, xr: Union[DataArray, Dataset]):
        xr.attrs.update(self.extra_attrs)
        if self.convert_to_int16:
            xr = scale_to_int16(
                xr,
                output_multiplier=self.output_value_multiplier,
                output_nodata=self.output_nodata,
                scale_int16s=self.scale_int16s,
            )
        return xr


@dataclass
class DsWriter(XrWriterMixin, Writer):
    write_function: Callable = write_to_blob_storage
    write_stac_function: Callable = write_stac_blob_storage
    write_stac: bool = True

    def write(self, xr: Dataset, item_id: str) -> Union[str, List]:
        xr = super().prep(xr)
        paths = []
        assets = {}
        client = get_container_client()

        # Use a threadpool to write all at once
        with ThreadPoolExecutor() as executor:
            futures = []
            for variable in xr:
                output_da = xr[variable].squeeze()
                path = self.itempath.path(item_id, variable)
                paths.append(path)
                futures.append(
                    executor.submit(
                        self.write_function,
                        output_da,
                        path=path,
                        write_args=dict(
                            driver="COG", nodata=output_da.attrs.get("nodata", None)
                        ),
                        overwrite=self.overwrite,
                        use_odc_writer=self.use_odc_writer,
                        client=client,
                    )
                )
            for future in futures:
                future.result()

        if self.write_stac:
            assets = {
                variable: Asset(
                    media_type="image/tiff; application=geotiff; profile=cloud-optimized",
                    href=str(URL("https://deppcpublicstorage.blob.core.windows.net/output") / path),
                    roles=["data"],
                )
                for variable, path in zip(xr, paths)
            }
            stac_id = self.itempath.basename(item_id)  # , variable)
            collection = self.itempath.item_prefix
            # has to be a datetime datetime object
            datetime = np.datetime64(xr.attrs["stac_properties"]["datetime"]).item()
            self.write_stac_function(
                xr,
                paths[0],
                stac_url=self.itempath.path(item_id, ext=".stac-item.json"),
                input_datetime=datetime,
                assets=assets,
                collection=collection,
                id=stac_id,
            )

        return paths


class LocalDsWriter(DsWriter):
    def __init__(self, use_odc_writer: bool = False, **kwargs):
        super().__init__(
            write_function=write_to_local_storage,
            write_stac_function=write_stac_local,
            use_odc_writer=use_odc_writer,
            **kwargs,
        )


class AzureDsWriter(DsWriter):
    pass
