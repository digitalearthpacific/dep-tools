from pathlib import Path
from typing import Literal, Iterator

import antimeridian
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from odc.geo import XY, BoundingBox, Geometry
from odc.geo.gridspec import GridSpec, GeoBox
from shapely.geometry import shape

# This EPSG code is what we're using for now
# but it's not ideal, as its not an equal area projection...
PACIFIC_EPSG = "EPSG:3832"

GADM_FILE = Path(__file__).parent / "gadm_pacific.gpkg"
GADM_UNION_FILE = Path(__file__).parent / "gadm_pacific_union.gpkg"
COUNTRIES_AND_CODES = {
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


def gadm() -> GeoDataFrame:
    if not GADM_FILE.exists() or not GADM_UNION_FILE.exists():
        all_polys = pd.concat(
            [
                gpd.read_file(
                    f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{code}.gpkg",
                    layer="ADM_ADM_0",
                )
                for code in COUNTRIES_AND_CODES.values()
            ]
        )

        all_polys.to_file(GADM_FILE)
        all_polys.dissolve()[["geometry"]].to_file(GADM_UNION_FILE)

    return gpd.read_file(GADM_FILE)


def gadm_union() -> GeoDataFrame:
    if not GADM_UNION_FILE.exists():
        gadm()

    return gpd.read_file(GADM_UNION_FILE)


def get_tiles(
    resolution: int | float = 30,
    country_codes: list[str] | None = None,
    buffer_distance: int | float | None = None,
) -> Iterator[tuple[tuple[int, int], GeoBox]]:
    """Returns a list of tile IDs for the Pacific region, optionally filtered by country code."""

    if country_codes is None:
        geometries = gadm_union()
    else:
        if not all(code in COUNTRIES_AND_CODES.values() for code in country_codes):
            raise ValueError(
                f"Invalid country code. Must be one of {', '.join(COUNTRIES_AND_CODES.values())}"
            )
        geometries = gadm().loc[lambda df: df["GID_0"].isin(country_codes)]

    return grid(
        resolution=resolution,
        return_type="GridSpec",
        intersect_with=geometries,
        buffer_distance=buffer_distance,
    )


def grid(
    resolution: int | float = 30,
    simplify_tolerance: float = 0.1,
    crs=PACIFIC_EPSG,
    return_type: Literal["GridSpec", "GeoSeries", "GeoDataFrame"] = "GridSpec",
    intersect_with: GeoDataFrame | None = None,
    buffer_distance: int | float | None = None,
) -> GridSpec | GeoSeries | GeoDataFrame | Iterator[tuple[tuple[int, int], GeoBox]]:
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
            features. If `return_type` is `GridSpec`, an iterator of tuples each
            containing the tile id (in column, row order) and its GeoBox.
            Otherwise, a GeoDataFrame containing only the portions of each tile
            that intersect the given GeoDataFrame is returned.
    """
    if intersect_with is not None:
        if return_type != "GridSpec":
            full_grid = _geoseries(resolution, crs)
            return _intersect_grid(full_grid, intersect_with)
        else:
            gridspec = _gridspec(resolution, crs)
            simplified = (
                intersect_with.to_crs(PACIFIC_EPSG)
                .simplify(simplify_tolerance)
                .to_frame()
                .to_geo_dict()
            )
            geometry = Geometry(
                simplified,
                crs=PACIFIC_EPSG,
            )
            if buffer_distance is not None:
                geometry = geometry.buffer(buffer_distance)
            else:
                geometry = geometry.buffer(0.0)
            return gridspec.tiles_from_geopolygon(geopolygon=geometry)

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


def landsat_grid():
    """The official Landsat grid filtered to Pacific Island Countries and
    Territories as defined by GADM."""
    ls_grid_path = Path(__file__).parent / "landsat_grid.gpkg"
    if not ls_grid_path.exists():
        landsat_pathrows = gpd.read_file(
            "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
        )
        ls_grid = landsat_pathrows.loc[
            landsat_pathrows.sjoin(
                gadm_union.to_crs(landsat_pathrows.crs), how="inner"
            ).index
        ]
        ls_grid.to_file(ls_grid_path)

    return gpd.read_file(ls_grid_path).set_index(["PATH", "ROW"])
