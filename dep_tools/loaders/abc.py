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
        load_as_dataset: bool = False,
        clip_to_area: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._kwargs = kwargs
        self._clip_to_area = clip_to_area
        self._load_as_dataset = load_as_dataset

    def load(self, items, areas: GeoDataFrame) -> Dataset | DataArray:
        xr = odc.stac.load(
            items,
            geopolygon=areas,
            **self._kwargs,
        )

        if not self._load_as_dataset:
            xr = xr.to_array("band").rename("data").rio.write_crs(xr.odc.crs)

        if self._clip_to_area:
            xr = xr.rio.clip(
                areas.to_crs(xr.odc.crs).geometry, all_touched=True, from_disk=True
            )

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
