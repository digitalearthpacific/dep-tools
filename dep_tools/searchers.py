from abc import ABC, abstractmethod
import warnings

from geopandas import GeoDataFrame, read_file
import pystac_client

from dep_tools.exceptions import EmptyCollectionError
from dep_tools.landsat_utils import items_in_pathrows, pathrows_in_area
from dep_tools.utils import (
    fix_bad_epsgs,
    remove_bad_items,
    search_across_180,
)


class Searcher(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def search(self, area):
        pass


class PystacSearcher(Searcher):
    """A Searcher which searches for stac items using pystac_client.Client.search.
    Fixes include correctly searching across the antimeridian by splitting the bounding
    box of the target area on either side and removal of known "bad" stac items (using
    dep_tools.utils.remove_bad_items).

    This is written to be used with the "Task" framework. If you just want to search
    for stac items and handle the antimeridian correctly, use
    :func:`dep_tools.utils.search_across_180`.

    Args:
        client: A search client.
        raise_empty_collection_error: Whether an EmptyCollectionError exception should
            be returned if no stac items are found.
        **kwargs: Additional arguments passed to client.search(). For example, passing
            `collections=["sentinel-2-l2a"]` will restrict results to Sentinel 2 stac
            items.
    """

    def __init__(
        self,
        client: pystac_client.Client | None = None,
        raise_empty_collection_error: bool = True,
        **kwargs,
    ):
        self._client = client
        self._raise_errors = raise_empty_collection_error
        self._kwargs = kwargs

    def search(self, area: GeoDataFrame):
        """Search for stac items within the bounds of the corresponding area.

        Args:
            area: An area in any projection as defined by the crs.

        Returns:
            An ItemCollection.
        """
        item_collection = search_across_180(
            region=area, client=self._client, **self._kwargs
        )

        if len(item_collection) == 0 and self._raise_errors:
            raise EmptyCollectionError()

        fix_bad_epsgs(item_collection)
        item_collection = remove_bad_items(item_collection)

        return item_collection


class LandsatPystacSearcher(PystacSearcher):
    def __init__(
        self,
        search_intersecting_pathrows: bool = False,
        exclude_platforms: list | None = None,
        only_tier_one: bool = False,
        fall_back_to_tier_two: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._kwargs["collections"] = ["landsat-c2-l2"]
        self._search_intersecting_pathrows = search_intersecting_pathrows
        self._exclude_platforms = exclude_platforms
        self._only_tier_one = only_tier_one
        self._fall_back_to_tier_two = fall_back_to_tier_two

        if "query" in self._kwargs.keys():
            warnings.warn(
                "`query` argument being ignored. To send a query directly, use `PystacSearcher`."
            )

        query = {}
        if self._exclude_platforms is not None:
            landsat_platforms = ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]
            query["platform"] = {
                "in": [p for p in landsat_platforms if p not in self._exclude_platforms]
            }

        if self._only_tier_one:
            query["landsat:collection_category"] = {"eq": "T1"}

        self._kwargs["query"] = query

        if self._search_intersecting_pathrows:
            self._landsat_pathrows = read_file(
                "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
            )

    def search(self, area: GeoDataFrame):
        search_area = (
            pathrows_in_area(area, self._landsat_pathrows)
            if self._search_intersecting_pathrows
            else area
        )
        try:
            items = super().search(search_area)
        except EmptyCollectionError:
            # If we're only looking for tier one items, try falling back to both T1 and T2
            if self._only_tier_one and self._fall_back_to_tier_two:
                self._only_tier_one = False
                items = self.search(search_area)
            else:
                raise EmptyCollectionError()

        if self._search_intersecting_pathrows:
            items = items_in_pathrows(items, search_area)

        return items
