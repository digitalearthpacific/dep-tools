from abc import ABC, abstractmethod
from typing import List, Union

# I get errors sometimes that `DataArray` has no attribute `rio`  which I _think_
# is a dask worker issue but it might be possible that `odc.stac` _sometimes_ needs
# rioxarray loaded ????
import rioxarray  # noqa: F401
from geopandas import GeoDataFrame
from odc.stac import load
from pystac import ItemCollection
from rasterio.errors import RasterioError, RasterioIOError
from stackstac import stack
from xarray import DataArray, Dataset, concat

from .exceptions import EmptyCollectionError
from .utils import fix_bad_epsgs, remove_bad_items, search_across_180

LANDSAT_PLATFORMS = ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]


class Loader(ABC):
    """A loader loads data."""

    def __init__(self):
        pass

    def load(self, area):
        pass


class StackXrLoader(Loader):
    """An abstract base class for Loaders which support loading pystac Item
    Collections into Xarray DataArray or Dataset objects.
    """

    def __init__(self, epsg=None, datetime=None, dask_chunksize=None):
        self.epsg = epsg
        self.datetime = datetime
        self.dask_chunksize = dask_chunksize
        self._current_epsg = epsg

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


class Sentinel2LoaderMixin(object):
    def _get_items(self, area):
        item_collection = search_across_180(
            area, collections=["sentinel-2-l2a"], datetime=self.datetime
        )
        if len(item_collection) == 0:
            raise EmptyCollectionError()

        item_collection = remove_bad_items(item_collection)

        return item_collection


class LandsatLoaderMixin(object):
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

    def _get_items(self, area):
        # TODO: move path/row filtering to the query
        # query = {
        #     "landsat:wrs_path": {"eq": PATH},
        #     "landsat:wrs_row": {"eq": ROW}
        # }
        query = {}
        if self._exclude_platforms is not None:
            # I don't know the syntax for `not in`, so I'm using `in` instead
            query["platform"] = {
                "in": [p for p in LANDSAT_PLATFORMS if p not in self._exclude_platforms]
            }

        if self._only_tier_one:
            query["landsat:collection_category"] = {"eq": "T1"}

        # Do the search
        item_collection = search_across_180(
            area, collections=["landsat-c2-l2"], datetime=self.datetime, query=query
        )

        # Fix a few issues with STAC items
        fix_bad_epsgs(item_collection)
        item_collection = remove_bad_items(item_collection)

        if len(item_collection) == 0:
            # If we're only looking for tier one items, try falling back to both T1 and T2
            if self._only_tier_one and self._fall_back_to_tier_two:
                self._only_tier_one = False
                return self._get_items(area)
            else:
                raise EmptyCollectionError()

        # Filtering by path/row...
        # Not lifting this into a query parameter yet as I'm not sure
        # how it's used. - Alex Nov 2023
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


class OdcLoaderMixin(object):
    def __init__(
        self,
        odc_load_kwargs=dict(),
        nodata_value: int | float | None = None,
        load_as_dataset: bool = False,
        keep_ints: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.odc_load_kwargs = odc_load_kwargs
        self.nodata = nodata_value
        self.load_as_dataset = load_as_dataset
        self.keep_ints = keep_ints

    def _get_xr(
        self,
        items,
        areas: GeoDataFrame,
    ) -> DataArray | Dataset:
        data_type = "uint16" if self.keep_ints else "float32"

        xr = load(
            items,
            geopolygon=areas,
            crs=self._current_epsg,
            chunks=self.dask_chunksize,
            dtype=data_type,
            nodata=self.nodata,
            **self.odc_load_kwargs,
        )

        if self.nodata is not None:
            xr.attrs["nodata"] = self.nodata

        if not self.keep_ints:
            for name in xr:
                nodata_value = (
                    xr[name].rio.nodata if self.nodata is None else self.nodata
                )
                xr[name] = xr[name].where(xr[name] != nodata_value, float("nan"))

            xr.attrs["nodata"] = float("nan")

        if not self.load_as_dataset:
            # This creates a "bands" dimension.
            xr = (
                xr.to_array(
                    "band"
                )  # ^^ just to match what stackstac makes, at least for now
                # stackstac names it stackstac-lkj1928d-l81938d890 or similar,
                # in places a name is needed (for instance .to_dataset())
                .rename("data")
                .rio.write_crs(self._current_epsg)
                .rio.write_nodata(float("nan"))
                .rio.clip(
                    areas.to_crs(self._current_epsg).geometry,
                    all_touched=True,
                    from_disk=True,
                )
            )

        return xr


class StackStacLoaderMixin:
    def __init__(
        self, stack_kwargs=dict(resolution=30), resamplers_and_assets=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.stack_kwargs = stack_kwargs
        self.resamplers_and_assets = resamplers_and_assets

    def _get_xr(
        self,
        item_collection: ItemCollection,
        areas: GeoDataFrame,
    ) -> DataArray:
        areas_proj = areas.to_crs(self._current_epsg)
        if self.resamplers_and_assets is not None:
            s = concat(
                [
                    stack(
                        item_collection,
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
                item_collection,
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


class Sentinel2OdcLoader(Sentinel2LoaderMixin, OdcLoaderMixin, StackXrLoader):
    def __init__(self, nodata_value=0, **kwargs):
        super().__init__(nodata_value=nodata_value, **kwargs)


class Sentinel2StackLoader(Sentinel2LoaderMixin, StackStacLoaderMixin, StackXrLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LandsatOdcLoader(LandsatLoaderMixin, OdcLoaderMixin, StackXrLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LandsatStackLoader(LandsatLoaderMixin, StackStacLoaderMixin, StackXrLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
