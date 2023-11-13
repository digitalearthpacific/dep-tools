from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urlpath import URL
from typing import Callable, Dict, List, Union

import numpy as np
from pystac import Asset

from .namers import DepItemPath
from .utils import scale_to_int16, write_to_blob_storage, write_to_local_storage
from .stac_utils import write_stac_local, write_stac_blob_storage
from xarray import DataArray, Dataset


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
        for variable in xr:
            output_da = xr[variable].squeeze()
            path = self.itempath.path(item_id, variable)
            paths.append(path)
            self.write_function(
                output_da,
                path=path,
                write_args=dict(driver="COG"),
                overwrite=self.overwrite,
            )
            # TODO: This stuff should be moved, just iterate again in write_stac_function
            # TODO: This is invalid for local file writing...
            az_prefix = URL("https://deppcpublicstorage.blob.core.windows.net/output")
            asset = Asset(
                media_type="image/tiff; application=geotiff; profile=cloud-optimized",
                href=str(az_prefix / path),
                roles=["data"],
            )
            assets[variable] = asset
        if self.write_stac:
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
    def __init__(self, **kwargs):
        super().__init__(
            write_function=write_to_local_storage,
            write_stac_function=write_stac_local,
            **kwargs,
        )


class AzureDsWriter(DsWriter):
    pass
