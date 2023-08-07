from abc import ABC, abstractmethod
from typing import Union, List

from geopandas import GeoDataFrame
from odc.stac import load
from pystac import ItemCollection
from rasterio import RasterioIOError

# I get errors sometimes that `DataArray` has no attribute `rio`  which I _think_
# is a dask worker issue but it might be possible that `odc.stac` _sometimes_ needs
# rioxarray loaded ????
import rioxarray


from stackstac import stack
from xarray import DataArray, concat

from .exceptions import EmptyCollectionError
from .utils import search_across_180, fix_bad_epsgs


class Loader(ABC):
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


class LandsatLoaderMixin(object):
    def __init__(
        self,
        load_tile_pathrow_only: bool = False,
        exclude_platforms: Union[List, None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.load_tile_pathrow_only = load_tile_pathrow_only
        # e.g. exclude_platforms = ['landsat-7']
        self._exclude_platforms = exclude_platforms

    def _get_items(self, area):
        index_dict = dict(zip(area.index.names, area.index[0]))
        item_collection = search_across_180(
            area,
            collections=["landsat-c2-l2"],
            datetime=self.datetime,
        )
        fix_bad_epsgs(item_collection)

        if self._exclude_platforms:
            item_collection = [
                i
                for i in item_collection
                if i.properties["platform"] not in self._exclude_platforms
            ]

        # If there are not items in this collection for _this_ pathrow,
        # we don't want to process, since they will be captured in
        # other pathrows (or are areas not covered by our aoi)

        item_collection_for_this_pathrow = [
            i
            for i in item_collection
            if i.properties["landsat:wrs_path"] == f"{index_dict['PATH']:03d}"
            and i.properties["landsat:wrs_row"] == f"{index_dict['ROW']:03d}"
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


class OdcLoaderMixin:
    def __init__(self, odc_load_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.odc_load_kwargs = odc_load_kwargs

    def _get_xr(
        self,
        items,
        areas: GeoDataFrame,
    ) -> DataArray:
        # For most EO data native dtype is int. Loading as such saves space but
        # the only more-or-less universally accepted nodata value is nan,
        # which is not available for int types. So we need to load as float and
        # then replace existing nodata values (usually 0) with nan. At least
        # I _think_ all this is necessary and there's not an easier way I didn't
        # see in the docs.
        xr = load(
            items,
            crs=self._current_epsg,
            chunks=self.dask_chunksize,
            **self.odc_load_kwargs,
            dtype="float32",
        )

        for name in xr:
            xr[name] = xr[name].where(xr[name] != xr[name].rio.nodata, float("nan"))

        return (
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


class StackStacLoaderMixin:
    def __init__(self, stack_kwargs=dict(), resamplers_and_assets=None, **kwargs):
        super().__init__(**kwargs)
        self.stack_kwargs = stack_kwargs
        self.resamplers_and_assets = resamplers_and_assets

    def _get_xr(
        self,
        item_collection: ItemCollection,
        areas: GeoDataFrame,
    ) -> DataArray:
        if self.resamplers_and_assets is not None:
            s = concat(
                [
                    stack(
                        item_collection,
                        chunksize=self.dask_chunksize,
                        epsg=self._current_epsg,
                        resolution=30,
                        errors_as_nodata=(RasterioIOError(".*"),),
                        assets=resampler_and_assets["assets"],
                        resampling=resampler_and_assets["resampler"],
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
                resolution=30,
                errors_as_nodata=(RasterioIOError(".*"),),
                **self.stack_kwargs,
            )

        return s.rio.write_crs(self._current_epsg).rio.clip(
            areas.to_crs(self._current_epsg).geometry,
            all_touched=True,
            from_disk=True,
        )


class LandsatOdcLoader(LandsatLoaderMixin, OdcLoaderMixin, Loader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LandsatStackLoader(LandsatLoaderMixin, StackStacLoaderMixin, Loader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
