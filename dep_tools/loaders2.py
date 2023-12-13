from abc import ABC, abstractmethod
from typing import List, Union

# I get errors sometimes that `DataArray` has no attribute `rio`  which I _think_
# is a dask worker issue but it might be possible that `odc.stac` _sometimes_ needs
# rioxarray loaded ????
import antimeridian
import rioxarray  # noqa: F401
from geopandas import GeoDataFrame
import odc.stac
from odc.geo.geobox import GeoBox
import pystac_client
from pystac import ItemCollection
from rasterio.errors import RasterioError, RasterioIOError
from stackstac import stack
from xarray import DataArray, Dataset, concat

from .exceptions import EmptyCollectionError
from .utils import fix_bad_epsgs, remove_bad_items, search_across_180


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


class Searcher(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def search(self, area):
        pass


class SearchLoader(Loader):
    def __init__(self, searcher: Searcher, loader: StacLoader):
        self.searcher = searcher
        self.loader = loader

    def load(self, area):
        return self.loader.load(self.searcher.search(area), area)


class StackXrLoader(Loader):
    """An abstract base class for Loaders which support loading pystac Item
    Collections into Xarray DataArray or Dataset objects.
    """

    def load(self, area) -> DataArray:
        items = self._get_items(area)
        return self._get_xr(items, area)

    @abstractmethod
    def _get_items(self, area) -> ItemCollection:
        pass

    @abstractmethod
    def _get_xr(
        self,
        items,
        area: GeoDataFrame,
    ) -> DataArray:
        pass


class PystacSearcher(Searcher):
    def __init__(self, client: pystac_client.Client | None = None, **kwargs):
        self._client = client
        self._kwargs = kwargs

    def search(self, area):
        item_collection = search_across_180(
            region=area, client=self._client, **self._kwargs
        )

        if len(item_collection) == 0:
            raise EmptyCollectionError()

        fix_bad_epsgs(item_collection)
        item_collection = remove_bad_items(item_collection)

        return item_collection


class SentinelPystacSearcher(PystacSearcher):
    def __init__(self, client, **kwargs):
        if "collections" in kwargs.keys():
            kwargs.pop("collections")
        super().__init__(client, collections=["sentinel-2-l2a"], **kwargs)


class LandsatPystacSearcher(PystacSearcher):
    def __init__(self, client, **kwargs):
        if "collections" in kwargs.keys():
            kwargs.pop("collections")
        super().__init__(client, collections=["landsat-c2-l2"], **kwargs)


class LandsatSearcher(PystacSearcher):
    def __init__(
        self,
        load_tile_pathrow_only: bool = False,
        exclude_platforms: Union[List, None] = None,
        only_tier_one: bool = False,
        fall_back_to_tier_two: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.load_tile_pathrow_only = load_tile_pathrow_only
        self._exclude_platforms = exclude_platforms
        self._only_tier_one = only_tier_one
        self._fall_back_to_tier_two = fall_back_to_tier_two

        self.query = {}
        if self._exclude_platforms is not None:
            # I don't know the syntax for `not in`, so I'm using `in` instead
            landsat_platforms = ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]
            self.query["platform"] = {
                "in": [p for p in landsat_platforms if p not in self._exclude_platforms]
            }

        if self._only_tier_one:
            self.query["landsat:collection_category"] = {"eq": "T1"}

    def search(self, area):
        try:
            return super().search(area)
        except EmptyCollectionError:
            # If we're only looking for tier one items, try falling back to both T1 and T2
            if self._only_tier_one and self._fall_back_to_tier_two:
                self._only_tier_one = False
                return self.search(area)
            else:
                raise EmptyCollectionError()

        # Filtering by path/row...
        # Not lifting this into a query parameter yet as I'm not sure
        # how it's used. - Alex Nov 2023
        # TODO: move path/row filtering to the query
        # query = {
        #     "landsat:wrs_path": {"eq": PATH},
        #     "landsat:wrs_row": {"eq": ROW}
        # }
        try:
            index_dict = dict(zip(area.index.names, area.index[0]))
        except TypeError:
            index_dict = {}

        if "PATH" in index_dict.keys() and "ROW" in index_dict.keys():
            item_collection_for_this_pathrow = [
                i
                for i in item_collection
                if i.properties["landsat:wrs_path"] == f"{index_dict['PATH'].zfill(3)}"
                and i.properties["landsat:wrs_row"] == f"{index_dict['ROW'].zfill(3)}"
            ]

            if len(item_collection_for_this_pathrow) == 0:
                raise EmptyCollectionError()

            if self.load_tile_pathrow_only:
                item_collection = item_collection_for_this_pathrow

            if self.epsg is None:
                self._current_epsg = item_collection_for_this_pathrow[0].properties[
                    "proj:epsg"
                ]

        return item_collection


class OdcLoader(StacLoader):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def load(self, items, area) -> DataArray | Dataset:
        return odc.stac.load(
            items,
            geopolygon=area.to_crs(4326),
            **self._kwargs,
        )


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
