"""ABC definition and implementation of Loader objects."""

from abc import ABC, abstractmethod
from typing import Any, Iterable
import warnings

from geopandas import GeoDataFrame
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.stac import load as stac_load
from pystac import Item
from rasterio.errors import RasterioError, RasterioIOError
import rioxarray
from stackstac import stack
from xarray import DataArray, Dataset, concat


class Loader(ABC):
    """A base class for something that loads data based on input areas."""

    def __init__(self):
        pass

    @abstractmethod
    def load(self, areas):
        pass


class StacLoader(Loader):
    """A loader which loads data based on (STAC) items and areas."""
    @abstractmethod
    def load(self, items, areas) -> Any:
        pass


class OdcLoader(StacLoader):
    """A wrapper around :func:`odc.stac.load`.

    In addition to allowing conformance to the :class:`Loader` form, this
    class offers a number of convenience operations which compliment
    :func:`odc.stac.load`:

        - If the data is loaded as floating point, any nodata values (as defined
          by the `"nodata"` attribute of the loaded data) are set to NaN, and 
          the attribute itself is reset to NaN.
        - The nodata value is also set to be accessed via the 
          rioxarray accessor (`.rio.nodata`).

    Args:
        load_as_dataset: If False, load as a DataArray with each variable in the
            `band` dimension.
        clip_to_areas: If True, loaded data is clipped to the given areas using
            :func:`odc.geo.mask`.
        **kwargs: Additional arguments to :func:`odc.stac.load`.
    """

    def __init__(
        self,
        load_as_dataset: bool = True,
        clip_to_areas: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._kwargs = kwargs
        self._clip_to_area = clip_to_areas
        self._load_as_dataset = load_as_dataset

    def load(
        self, items: Iterable[Item], areas: GeoDataFrame | GeoBox | None = None
    ) -> Dataset | DataArray:
        """Load STAC Items into an xarray object.

        If `nodata` is passed as a kwarg on initialization, or the stac item
        contains the nodata value, `xr[variable].nodata` will be set on load.

        Args:
            items: The items to load.
            areas: If `clip_to_areas` is `True`, the output is clipped to these areas.

        Returns:
            If `load_to_dataset` is True on initialization, a :class:`xarray.Dataset`
            is returned. Otherwise a :class:`xarray:DataArray` is returned, with
            variables set on the `"band"` dimension.


        Raises:
            ValueError: If `areas` is a GeoBox and `clip_to_areas` is set to `True`.
        """
        if isinstance(areas, GeoDataFrame):
            load_geometry = dict(geopolygons=areas)
        elif isinstance(areas, GeoBox):
            load_geometry = dict(geobox=areas)
        else:
            load_geometry = dict()

        ds = stac_load(
            items,
            **load_geometry,
            **self._kwargs,
        )

        for name in ds:
            # Since nan is more-or-less universally accepted as a nodata value,
            # if the dtype of a band is some sort of floating point, then recode
            # existing values that are equal to the value set on load to nan
            if ds[name].dtype.kind == "f":
                # Should I make this an option?
                if "nodata" in ds[name].attrs.keys():
                    ds[name] = ds[name].where(ds[name] != ds[name].nodata, float("nan"))
                ds[name].attrs["nodata"] = float("nan")
            # To be helpful, set the nodata for rioxarray accessor
            ds[name].rio.write_nodata(ds[name].attrs.get("nodata"), inplace=True)

        if self._clip_to_area:
            if isinstance(areas, GeoBox):
                raise ValueError(
                    "Clip not supported for GeoBox (nor should it be needed)"
                )

            if areas is None:
                warnings.warn("clip_to_area is True but areas is None, ignoring")

            geom = Geometry(areas.geometry.unary_union, crs=areas.crs)
            ds = ds.odc.mask(geom)

        if not self._load_as_dataset:
            return (
                ds.to_array("band")
                .rename(band="data")
                .rio.write_crs(ds[list(ds.data_vars)[0]].odc.crs)
            )

        return ds


class StackStacLoader(StacLoader):
    def __init__(
        self, stack_kwargs=dict(resolution=30), resamplers_and_assets=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.stack_kwargs = stack_kwargs
        self.resamplers_and_assets = resamplers_and_assets

    def load(
        self,
        items,
        areas: GeoDataFrame,
    ) -> DataArray:
        areas_proj = areas.to_crs(self._current_epsg)
        if self.resamplers_and_assets is not None:
            s = concat(
                [
                    stack(
                        items,
                        chunksize=self.dask_chunksize,
                        epsg=self._current_epsg,
                        errors_as_nodata=(RasterioError(".*"),),
                        assets=resampler_and_assets["assets"],
                        resampling=resampler_and_assets["resampler"],
                        bounds=areas_proj.total_bounds.tolist(),
                        band_coords=False,  # needed or some coords are often missing
                        # from qa pixel and we get an error. Make sure it doesn't
                        # mess anything up else where (e.g. rio.crs)
                        **self.stack_kwargs,
                    )
                    for resampler_and_assets in self.resamplers_and_assets
                ],
                dim="band",
            )
        else:
            s = stack(
                items,
                chunksize=self.dask_chunksize,
                epsg=self._current_epsg,
                errors_as_nodata=(RasterioIOError(".*"),),
                **self.stack_kwargs,
            )

        return s.rio.write_crs(self._current_epsg).rio.clip(
            areas_proj.geometry,
            all_touched=True,
            from_disk=True,
        )
