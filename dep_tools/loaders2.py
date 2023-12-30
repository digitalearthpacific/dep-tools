from abc import ABC, abstractmethod
from statistics import mode
from typing import List, Union

# I get errors sometimes that `DataArray` has no attribute `rio`  which I _think_
# is a dask worker issue but it might be possible that `odc.stac` _sometimes_ needs
# rioxarray loaded ????
import rioxarray  # noqa: F401
from geopandas import GeoDataFrame
import odc.stac
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


class PystacSearcher(Searcher):
    def __init__(
        self,
        client: pystac_client.Client | None = None,
        raise_empty_collection_error: bool = True,
        **kwargs,
    ):
        self._client = client
        self._raise_errors = raise_empty_collection_error
        self._kwargs = kwargs

    def search(self, area):
        item_collection = search_across_180(
            region=area, client=self._client, **self._kwargs
        )

        if len(item_collection) == 0 and self._raise_errors:
            raise EmptyCollectionError()

        fix_bad_epsgs(item_collection)
        item_collection = remove_bad_items(item_collection)

        return item_collection


import geopandas as gpd
from shapely.geometry import box


def bbox_across_180(area):
    bbox = area.to_crs(4326).total_bounds
    # If the lower left X coordinate is greater than 180 it needs to shift
    if bbox[0] > 180:
        bbox[0] = bbox[0] - 360
        # If the upper right X coordinate is greater than 180 it needs to shift
        # but only if the lower left one did too... otherwise we split it below
        if bbox[2] > 180:
            bbox[2] = bbox[2] - 360

    # These are Pacific specific tests!
    bbox_crosses_antimeridian = (bbox[0] < 0 and bbox[2] > 0) or (
        bbox[0] < 180 and bbox[2] > 180
    )
    if bbox_crosses_antimeridian:
        xmax_ll, ymin_ll = bbox[0], bbox[1]
        xmin_ll, ymax_ll = bbox[2], bbox[3]

        xmax_ll = xmax_ll - 360 if xmax_ll > 180 else xmax_ll

        left_bbox = [xmin_ll, ymin_ll, 180, ymax_ll]
        right_bbox = [-180, ymin_ll, xmax_ll, ymax_ll]
        return (left_bbox, right_bbox)
    else:
        return bbox


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


class PathrowPystacSearcher(LandsatPystacSearcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._landsat_pathrows = gpd.read_file(
            "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
        )

    def _get_pathrows(self, area):
        bbox = bbox_across_180(area)
        if len(bbox) == 2:
            return self._landsat_pathrows[
                self._landsat_pathrows.intersects(box(*bbox[0]))
                | self._landsat_pathrows.intersects(box(*bbox[1]))
            ]

        return self._landsat_pathrows[self._landsat_pathrows.intersects(box(*bbox))]

    def search(self, area):
        pathrows = self._get_pathrows(area)

        # because this search is by bbox, it will get items that are not in
        # these pathrows
        items = super().search(pathrows)

        return ItemCollection(
            pathrows.apply(
                lambda row: [
                    i
                    for i in items
                    if i.properties["landsat:wrs_path"] == str(row["PATH"]).zfill(3)
                    and i.properties["landsat:wrs_row"] == str(row["ROW"]).zfill(3)
                ],
                axis=1,
            ).sum()
        )


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


class DeluxeOdcLoader(OdcLoader):
    def __init__(
        self,
        clip_to_area: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._clip_to_area = clip_to_area

    def load(self, items, area) -> Dataset | DataArray:
        ds = super().load(items, area)

        if self._clip_to_area:
            ds = ds.rio.clip(
                area.to_crs(ds.odc.crs).geometry, all_touched=True, from_disk=True
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
