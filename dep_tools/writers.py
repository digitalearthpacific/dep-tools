from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Union

from azure.storage.blob import ContentSettings
from rio_stac.stac import create_stac_item

from .utils import scale_to_int16, write_to_blob_storage
from xarray import DataArray, Dataset


class Writer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def write(self, data, id) -> str:
        pass


@dataclass
class XrWriterMixin(object):
    dataset_id: str
    year: Union[str, None] = None
    prefix: Union[str, None] = None
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

    def _get_path(
        self, item_id, time: Union[str, None] = None, variable: Union[str, None] = None
    ) -> str:
        if variable is None:
            variable = self.dataset_id
        if time is None:
            time = self.year

        prefix = f"{self.prefix}/" if self.prefix is not None else ""
        time = time.replace("/", "_") if time is not None else time
        suffix = "_".join([str(i) for i in item_id])
        return (
            f"{prefix}{self.dataset_id}/{time}/{variable}_{time}_{suffix}.tif"
            if time is not None
            else f"{prefix}{self.dataset_id}/{variable}_{suffix}.tif"
        )


class LocalXrWriter(XrWriterMixin, Writer):
    def __init__(self, write_kwargs=dict(), **kwargs):
        super().__init__(**kwargs)
        self.write_kwargs = write_kwargs

    def write(self, xr: Union[DataArray, Dataset], item_id: str) -> str:
        xr = super().prep(xr)
        path = self._get_path(item_id)
        xr.squeeze().rio.to_raster(
            raster_path=path,
            compress="LZW",
            overwrite=self.overwrite,
            **self.write_kwargs,
        )
        return path


@dataclass
class AzureXrWriter(XrWriterMixin, Writer):
    split_by_variable: bool = False
    split_by_time: bool = False
    write_stac: bool = False

    def write(
        self, xr: Union[DataArray, Dataset], item_id: Union[str, List]
    ) -> Union[str, List]:
        xr = super().prep(xr)
        # If we want to create an output for each year, split results
        # into a list of da/ds. Useful in theory but in practice it often
        # made dask jobs unwieldy and caused slowdowns. My current
        # recommendation is to one run process for each year's work of data,
        # particularly for larger areas.
        if self.split_by_time:
            xr = [xr.sel(time=time) for time in xr.coords["time"]]

        # If we want to create an output for each variable, split or further
        # split results into a list of da/ds for each variable
        # or variable x year. This is behaves a little better than splitting
        # by year above, since individual variables often use the same data,
        # requiring only one pull. However writing multiple files does create
        # overhead. Perhaps only useful / required when the output dataset
        # with multiple variables doesn't fit into memory (since as of this
        # writing write_to_blob_storage must first write to an in memory
        # object before shipping to azure bs.
        if self.split_by_variable:
            xr = (
                [xr.to_array().sel(variable=var) for xr in xr for var in xr]
                if self.split_by_time
                else [xr.to_array().sel(variable=var) for var in xr]
            )

        if isinstance(xr, List):
            paths = []
            for this_xr in xr:
                time = (
                    this_xr.time.values.tolist() if "time" in this_xr.coords else None
                )
                variable = (
                    this_xr.coords["variable"].values
                    if "variable" in this_xr.coords
                    else None
                )

                path = self._get_path(item_id, time, variable)
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
            path = self._get_path(item_id)
            write_to_blob_storage(
                # Squeeze here in case we have e.g. a single time reading
                xr.squeeze(),
                path=path,
                write_args=dict(driver="COG", compress="LZW"),
                overwrite=self.overwrite,
            )
            if self.write_stac:
                _write_stac(path)

            return path


def _write_stac(path) -> None:
    az_prefix = Path("https://deppcpublicstorage.blob.core.windows.net/output")
    blob_url = az_prefix / path
    item = create_stac_item(blob_url, with_proj=True)
    item_json = json.dumps(item.to_dict(), indent=4)
    stac_url = Path(path).with_suffix(".stac-item.json")
    write_to_blob_storage(
        item_json,
        stac_url,
        write_args=dict(
            content_settings=ContentSettings(content_type="application/json")
        ),
    )
