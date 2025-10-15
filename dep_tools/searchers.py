"""This module contains the definition and implementation of :class:`Searcher` objects."""

import warnings
from abc import ABC, abstractmethod

from geopandas import GeoDataFrame
from odc.geo.geobox import GeoBox
from pystac import ItemCollection
from pystac_client import Client

from dep_tools.exceptions import EmptyCollectionError
from dep_tools.landsat_utils import items_in_pathrows, pathrows_in_area
from dep_tools.utils import fix_bad_epsgs, remove_bad_items, search_across_180


class Searcher(ABC):
    """An onbject which searches for something, based on an area."""

    def __init__(self):
        pass

    @abstractmethod
    def search(self, area):
        pass


class PystacSearcher(Searcher):

    def __init__(
        self,
        catalog: str | None = None,
        client: Client | None = None,
        raise_empty_collection_error: bool = True,
        **kwargs,
    ):
        """A Searcher which searches for stac items using pystac_client.Client.search.

        Fixes include correctly searching across the antimeridian by splitting the
        bounding box of the target area on either side and removal of known "bad"
        stac items (using dep_tools.utils.remove_bad_items).

        This is written to be used with the :class:`Task` framework. If you just
        want to search for stac items and handle the antimeridian correctly, use
        :func:`dep_tools.utils.search_across_180`.

        Args:
            catalog: The URL of a stac catalog. if client is specified, this is
                ignored.
            client: A search client. Either this or catalog must be specified.
            raise_empty_collection_error: Whether an EmptyCollectionError exception
                should be returned if no stac items are found.
            **kwargs: Additional arguments passed to client.search(). For example,
                passing `collections=["sentinel-2-l2a"]` will restrict results to
                Sentinel 2 stac items.
        """
        if client and catalog:
            warnings.warn(
                "Arguments for both 'client' and 'catalog' passed to PystacSearcher, ignoring catalog"
            )

        if not (client or catalog):
            raise ValueError("Must specify either client or catalog")

        self._client = client if client else Client.open(catalog)
        self._raise_errors = raise_empty_collection_error
        self._kwargs = kwargs

    def search(self, area: GeoDataFrame | GeoBox) -> ItemCollection:
        """Search for stac items within the bounds of the corresponding area.

        Args:
            area: An area in any projection as defined by the crs.

        Returns:
            An ItemCollection.
        """
        item_collection = search_across_180(
            region=area, client=self._client, **self._kwargs
        )

        if self._client.id == "microsoft-pc":
            fix_bad_epsgs(item_collection)
        item_collection = remove_bad_items(item_collection)

        if len(item_collection) == 0 and self._raise_errors:
            raise EmptyCollectionError()

        return item_collection


class LandsatPystacSearcher(PystacSearcher):
    def __init__(
        self,
        catalog: str | None = None,
        client: Client | None = None,
        collections: list[str] | None = ["landsat-c2-l2"],
        raise_empty_collection_error: bool = True,
        search_intersecting_pathrows: bool = False,
        exclude_platforms: list[str] | None = None,
        only_tier_one: bool = False,
        fall_back_to_tier_two: bool = False,
        **kwargs,
    ):
        """A PystacSearcher with special functionality for landsat data on the
        MSPC. Currently it overwrites any `query` kwarg, so if you want a direct
        query, just use :class:PystacSearcher.

        Args:
            catalog: The URL of a stac catalog. if client is specified, this is
                ignored.
            client: A search client. Either this or catalog must be specified.
            raise_empty_collection_error: Whether an EmptyCollectionError exception should
                be returned if no stac items are found.
            search_intersecting_pathrows: Whether to use landsat pathrows which
                intersect the area passed to :func:search rather than the area itself.
                This is a workaround for bad geometry in some stac items which
                cross the antimeridian.
            exclude_platforms: A list of platforms (e.g. ["landsat-7"]) to exclude
                from searching.
            only_tier_one: Whether to only search for tier one landsat data.
            fall_back_to_tier_two: If `only_tier_one` is set to True and no items
                are returned from the search, search again with tier two data
                included.
            **kwargs: Additional arguments passed to client.search(). `collections`
                and `query` arguments will be overwritten.
        """
        super().__init__(
            catalog=catalog,
            client=client,
            raise_empty_collection_error=raise_empty_collection_error,
            **kwargs,
        )
        self._kwargs["collections"] = collections
        self._search_intersecting_pathrows = search_intersecting_pathrows
        self._exclude_platforms = exclude_platforms
        self._only_tier_one = only_tier_one
        self._fall_back_to_tier_two = fall_back_to_tier_two

        # For now, just warn that we're overwriting the query. In future,
        # we might look to combine.
        if "query" in self._kwargs.keys():
            warnings.warn(
                "Portions of `query` argument may be replaced. To specify the full query directly, use `PystacSearcher`."
            )
            query = kwargs.pop("query")
        else:
            query = {}

        if self._exclude_platforms is not None:
            landsat_platforms = ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]
            query["platform"] = {
                "in": [p for p in landsat_platforms if p not in self._exclude_platforms]
            }

        if self._only_tier_one:
            query["landsat:collection_category"] = {"eq": "T1"}

        self._kwargs["query"] = query

    def search(self, area: GeoDataFrame):
        """Perform the search.

        Args:
            area: Any area.

        Returns:
            An ItemCollection.

        Raises:
            EmptyCollectionError: If the search finds no items.
        """
        search_area = (
            pathrows_in_area(area) if self._search_intersecting_pathrows else area
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

        if len(items) == 0 and self._raise_errors:
            raise EmptyCollectionError()

        return items
