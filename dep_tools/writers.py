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


# This is _only_ here because it's still used in dep-coastlines. I just need
# some time to migrate over there. (If i get a chance I'll just move it)
@dataclass
class AzureXrWriter(XrWriterMixin, Writer):
    write_stac: bool = True

    def write(
        self, xr: Union[DataArray, Dataset], item_id: Union[str, List]
    ) -> Union[str, List]:
        xr = super().prep(xr)
        # If we want to create an output for each variable, split or further
        # split results into a list of da/ds for each variable
        # or variable x year. This is behaves a little better than splitting
        # by year above, since individual variables often use the same data,
        # requiring only one pull. However writing multiple files does create
        # overhead. Perhaps only useful / required when the output dataset
        # with multiple variables doesn't fit into memory (since as of this
        # writing write_to_blob_storage must first write to an in memory
        # object before shipping to azure bs.
        if isinstance(xr, Dataset):
            xr = [xr.to_array().sel(variable=var) for var in xr]

        if isinstance(xr, List):
            paths = []
            for this_xr in xr:
                variable = this_xr.coords["variable"].values
                path = self.itempath.path(item_id, variable)
                paths.append(path)
                write_to_blob_storage(
                    this_xr,
                    path=path,
                    write_args=dict(driver="COG", compress="LZW"),
                    overwrite=self.overwrite,
                )
            return paths
        else:
            # We cannot write outputs with > 2 dimensions using rio.to_raster,
            # so we create new variables for each year x variable combination
            # Note this requires time to represent year, so we should consider
            # doing that here as well (rather than earlier).
            if not isinstance(xr, DataArray) and len(xr.dims.keys()) > 2:
                xr = (
                    xr.to_array(dim="variables")
                    .stack(z=["time", "variables"])
                    .to_dataset(dim="z")
                    .pipe(
                        lambda ds: ds.rename_vars(
                            {name: "_".join(name) for name in ds.data_vars}
                        )
                    )
                    .drop_vars(["variables", "time"])
                )
            path = self.itempath.path(item_id, ".tif")
            write_to_blob_storage(
                # Squeeze here in case we have e.g. a single time reading
                xr.squeeze(),
                path=path,
                write_args=dict(driver="COG", compress="LZW"),
                overwrite=self.overwrite,
            )
            if self.write_stac:
                print("This should fail, right? Where's the function?")
                _write_stac(xr.squeeze(), path)

            return path
