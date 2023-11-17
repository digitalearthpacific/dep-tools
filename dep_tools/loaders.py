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
from xarray import DataArray, concat

from .exceptions import EmptyCollectionError
from .utils import fix_bad_epsgs, remove_bad_items, search_across_180


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
        item_collection = search_across_180(
            area,
            collections=["landsat-c2-l2"],
            datetime=self.datetime,
        )
        fix_bad_epsgs(item_collection)
        item_collection = remove_bad_items(item_collection)

        # TODO: these can be queries, which means we don't get them back
        # from the server in the first place.
        if self._exclude_platforms:
            item_collection = [
                i
                for i in item_collection
                if i.properties["platform"] not in self._exclude_platforms
            ]

        if self._only_tier_one:
            item_collection = [
                i
                for i in item_collection
                if i.properties["landsat:collection_category"] == "T1"
            ]

        # If there are not items in this collection for _this_ pathrow,

        # we don't want to process, since they will be captured in
        # other pathrows (or are areas not covered by our aoi)

        try:
            index_dict = dict(zip(area.index.names, area.index[0]))
        except TypeError:
            index_dict = {}

        if len(item_collection) == 0:
            if self._fall_back_to_tier_two:
                self._only_tier_one = False
                return self._get_items(area)
            else:
                raise EmptyCollectionError()

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


class OdcLoaderMixin:
    def __init__(
        self,
        odc_load_kwargs,
        nodata_value: int | float | None = None,
        flat_array: bool = False,
        keep_ints: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.odc_load_kwargs = odc_load_kwargs
        self.nodata = nodata_value
        self.flat_array = flat_array
        self.keep_ints = keep_ints

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
        areas_proj = areas.to_crs(self._current_epsg)
        bounds = areas_proj.total_bounds.tolist()

        data_type = "uint16" if self.keep_ints else "float32"

        xr = load(
            items,
            crs=self._current_epsg,
            chunks=self.dask_chunksize,
            x=(bounds[0], bounds[2]),
            y=(bounds[1], bounds[3]),
            **self.odc_load_kwargs,
            dtype=data_type,
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

        if not self.flat_array:
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
                    areas_proj.geometry,
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
