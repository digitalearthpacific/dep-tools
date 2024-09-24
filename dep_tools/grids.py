from typing import Literal

import antimeridian
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from odc.geo import XY, BoundingBox
from odc.geo.gridspec import GridSpec
from shapely.geometry import shape

# This EPSG code is what we're using for now
# but it's not ideal, as its not an equal area projection...
PACIFIC_EPSG = "EPSG:3832"


def grid(
    resolution: int | float = 30,
    crs=PACIFIC_EPSG,
    return_type: Literal["GridSpec", "GeoSeries", "GeoDataFrame"] = "GridSpec",
    intersect_with: GeoDataFrame | None = None,
) -> GridSpec | GeoSeries | GeoDataFrame:
    """Returns a GridSpec or GeoSeries representing the Pacific grid, optionally
    intersected with an area of interest.

    Args:
        resolution: The resolution, in meters, of the output. As tiles are
            defined to be 96,000 meters on each side, it should divide 96,000
            evenly.
        crs: The desired crs of the output.
        return_type: The return type. If intersect_with (see below) is not None,
            this is ignored.
        intersect_with: The output is intersected with the supplied GeoDataFrame
            before returning, returning only tiles which overlap with those
            features. Forces the output to be a GeoDataFrame.
    """

    if intersect_with is not None:
        full_grid = _geoseries(resolution, crs)
        return _intersect_grid(full_grid, intersect_with)

    return {
        "GridSpec": _gridspec,
        "GeoSeries": _geoseries,
        "GeoDataFrame": _geodataframe,
    }[return_type](resolution, crs)


def _intersect_grid(grid: GeoSeries, areas_of_interest):
    return gpd.sjoin(
        gpd.GeoDataFrame(geometry=grid), areas_of_interest.to_crs(grid.crs)
    ).drop(columns=["index_right"])


def _gridspec(resolution, crs=PACIFIC_EPSG):
    gridspec_origin = XY(-3000000.0, -4000000.0)

    side_in_meters = 96_000
    shape = (side_in_meters / resolution, side_in_meters / resolution)

    return GridSpec(
        crs=crs,
        tile_shape=shape,
        resolution=resolution,
        origin=gridspec_origin,
    )


def _geodataframe(resolution, crs=PACIFIC_EPSG):
    return GeoDataFrame(geometry=_geoseries(resolution, crs), crs=crs)


def _geoseries(resolution, crs) -> GeoSeries:
    bounds = BoundingBox(120, -30, 280, 30, crs="EPSG:4326").to_crs(crs)
    tiles = _gridspec(resolution, crs).tiles(bounds)
    geometry, index = zip(
        *[(a_tile[1].boundingbox.polygon.geom, a_tile[0]) for a_tile in tiles]
    )

    gs = gpd.GeoSeries(geometry, index, crs=PACIFIC_EPSG)
    if crs != PACIFIC_EPSG:
        gs = gs.to_crs(crs)
        if crs == 4326:
            gs = gs.apply(lambda geom: shape(antimeridian.fix_shape(geom)))

    return gs


# The origin is in the projected CRS. This works for Landsat.
PACIFIC_GRID_30 = grid()

# This grid is for Sentinel-2 and has the same footprint
PACIFIC_GRID_10 = grid(resolution=10)
