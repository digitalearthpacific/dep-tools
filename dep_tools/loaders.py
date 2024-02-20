from abc import ABC, abstractmethod

from geopandas import GeoDataFrame
import odc.stac
from rasterio.errors import RasterioError, RasterioIOError
from stackstac import stack
from xarray import DataArray, Dataset, concat

from dep_tools.searchers import Searcher


class Loader(ABC):
    """A loader loads data."""

    def __init__(self):
        pass

    @abstractmethod
    def load(self, area):
        pass


class StacLoader(Loader):
    @abstractmethod
    def load(self, items, area):
        pass


class SearchLoader(Loader):
    def __init__(self, searcher: Searcher, loader: StacLoader):
        self.searcher = searcher
        self.loader = loader

    def load(self, area):
        return self.loader.load(self.searcher.search(area), area)


class OdcLoader(StacLoader):
    def __init__(
        self,
        load_as_dataset: bool = True,
        clip_to_area: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._kwargs = kwargs
        self._clip_to_area = clip_to_area
        self._load_as_dataset = load_as_dataset

    def load(self, items, areas: GeoDataFrame) -> Dataset | DataArray:
        # If `nodata` is passed as an arg, or the stac item contains the nodata
        # value, xr[variable].nodata will be set on load.
        xr = odc.stac.load(
            items,
            geopolygon=areas,
            **self._kwargs,
        )

        # TODO: need to handle cases where nodata is _not_ set on load. (see
        # landsat qr_radsat band)
        for name in xr:
            # Since nan is more-or-less universally accepted as a nodata value,
            # if the dtype of a band is some sort of floating point, then recode
            # existing values that are equal to the value set on load to nan
            if xr[name].dtype.kind == "f":
                # Should I make this an option?
                if "nodata" in xr[name].attrs.keys():
                    xr[name] = xr[name].where(xr[name] != xr[name].nodata, float("nan"))
                xr[name].attrs["nodata"] = float("nan")
            # To be helpful, set the nodata so rioxarray can understand it too.
            xr[name].rio.write_nodata(xr[name].nodata, inplace=True)

        if self._clip_to_area:
            xr = xr.rio.clip(
                areas.to_crs(xr.odc.crs).geometry, all_touched=True, from_disk=True
            )
            # Clip loses this, so re-set.
            for name in xr:
                xr[name].attrs["nodata"] = xr[name].rio.nodata

        if not self._load_as_dataset:
            xr = xr.to_array("band").rename("data").rio.write_crs(xr.odc.crs)

        return xr


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
