import geopandas as gpd
import pytest
from shapely.geometry import box

from dep_tools.landsat_utils import pathrows_in_area
from dep_tools.searchers import LandsatPystacSearcher

# From https://github.com/microsoft/PlanetaryComputer/issues/296

DATETIME = "2022-09/2022-10"
COLLECTIONS = ["landsat-c2-l2"]
BBOX = [179.1, -17.7, 179.95, -17.0]


@pytest.fixture()
def near_antimeridian_area() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[box(*BBOX)],
        crs="EPSG:4326",
    )


def test_pathrows_in_area(near_antimeridian_area):
    pathrows = pathrows_in_area(near_antimeridian_area)
    assert pathrows.PR.tolist() == ["073072", "074072"]


def test_search_for_stac_items_with_bad_geoms(near_antimeridian_area):
    searcher = LandsatPystacSearcher(
        datetime=DATETIME, search_intersecting_pathrows=True
    )

    items = searcher.search(near_antimeridian_area)
    assert len(items) == 24
