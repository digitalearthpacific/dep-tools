from json import loads
from pathlib import Path
from typing import Literal

import antimeridian
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from odc.geo import XY, BoundingBox, Geometry
from odc.geo.gridspec import GridSpec
from shapely.geometry import shape

# This EPSG code is what we're using for now
# but it's not ideal, as its not an equal area projection...
PACIFIC_EPSG = "EPSG:3832"

GADM_FILE = Path(__file__).parent / "gadm_pacific.gpkg"
GADM_UNION_FILE = Path(__file__).parent / "gadm_pacific_union.gpkg"


def _get_gadm() -> GeoDataFrame:
    if not GADM_FILE.exists() or not GADM_UNION_FILE.exists():
        countries_and_codes = {
            "American Samoa": "ASM",
            "Cook Islands": "COK",
            "Fiji": "FJI",
            "French Polynesia": "PYF",
            "Guam": "GUM",
            "Kiribati": "KIR",
            "Marshall Islands": "MHL",
            "Micronesia": "FSM",
            "Nauru": "NRU",
            "New Caledonia": "NCL",
            "Niue": "NIU",
            "Northern Mariana Islands": "MNP",
            "Palau": "PLW",
            "Papua New Guinea": "PNG",
            "Pitcairn Islands": "PCN",
            "Solomon Islands": "SLB",
            "Samoa": "WSM",
            "Tokelau": "TKL",
            "Tonga": "TON",
            "Tuvalu": "TUV",
            "Vanuatu": "VUT",
            "Wallis and Futuna": "WLF",
        }

        all_polys = pd.concat(
            [
                gpd.read_file(
                    f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{code}.gpkg"
                )
                for code in countries_and_codes.values()
            ]
        )

        all_polys.to_file(GADM_FILE)
        all_polys.dissolve()[["geometry"]].to_file(GADM_UNION_FILE)

    return gpd.read_file(GADM_FILE)


def _get_gadm_union() -> GeoDataFrame:
    if not GADM_UNION_FILE.exists():
        _get_gadm()

    return gpd.read_file(GADM_UNION_FILE)


def get_tiles(
    resolution: int | float = 30,
    country_codes: list[str] | None = None,
    tight: bool = False,
) -> list[(list[int, int], GridSpec)]:
    """Returns a list of tile IDs for the Pacific region, optionally filtered by country code."""

    if country_codes is None:
        geometries = _get_gadm_union()
    else:
        geometries = _get_gadm().loc[lambda df: df["GID_0"].isin(country_codes)]

    return grid(
        resolution=resolution,
        return_type="GridSpec",
        intersect_with=geometries,
        tight=tight,
    )


def grid(
    resolution: int | float = 30,
    crs=PACIFIC_EPSG,
    return_type: Literal["GridSpec", "GeoSeries", "GeoDataFrame"] = "GridSpec",
    intersect_with: GeoDataFrame | None = None,
    tight: bool = True,
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
        if return_type != "GridSpec":
            full_grid = _geoseries(resolution, crs)
            return _intersect_grid(full_grid, intersect_with)
        else:
            gridspec = _gridspec(resolution, crs)
            geometry = Geometry(loads(intersect_with.to_json()))
            # This is a bit of a hack, but it works. Geometries that are transformed by the tiles_from_geopolygon
            # are not valid, but doing the simplification and buffer fixes them.
            buffer = 0.0 if tight else 1000
            fixed = (
                geometry.to_crs(PACIFIC_EPSG, check_and_fix=True, wrapdateline=True)
                .simplify(0.01)
                .buffer(buffer)
            )
            return gridspec.tiles_from_geopolygon(geopolygon=fixed)

    return {
        "GridSpec": _gridspec,
        "GeoSeries": _geoseries,
        "GeoDataFrame": _geodataframe,
    }[return_type](resolution, crs)


def _intersect_grid(grid: GeoSeries, areas_of_interest) -> GeoDataFrame:
    return gpd.sjoin(
        gpd.GeoDataFrame(geometry=grid), areas_of_interest.to_crs(grid.crs)
    ).drop(columns=["index_right"])


def _gridspec(resolution, crs=PACIFIC_EPSG) -> GridSpec:
    gridspec_origin = XY(-3000000.0, -4000000.0)

    side_in_meters = 96_000
    shape = (side_in_meters / resolution, side_in_meters / resolution)

    return GridSpec(
        crs=crs,
        tile_shape=shape,
        resolution=resolution,
        origin=gridspec_origin,
    )


def _geodataframe(resolution, crs=PACIFIC_EPSG) -> GeoDataFrame:
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
