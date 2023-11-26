import geopandas as gpd
import pytest
from pystac_client import Client
from shapely.geometry import box

from dep_tools.loaders import LandsatOdcLoader

HOBART_BBOX = [147.0, -43.0, 148.0, -42.0]
LANDSAT_COLLECTION = "landsat-c2-l2"
DATETIME = "2020-01/2020-03"


@pytest.fixture
def mspc_client():
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


@pytest.fixture()
def hobart_area():
    return gpd.GeoDataFrame(
        geometry=[box(*HOBART_BBOX)],
        crs="EPSG:4326",
    )


def test_hobart_all(hobart_area):
    loader = LandsatOdcLoader(
        datetime=DATETIME,
        exclude_platforms=None,
        only_tier_one=True,
        fall_back_to_tier_two=False,
    )

    items = list(loader._get_items(hobart_area))

    assert len(items) == 56


def test_hobart_tier_one_only(hobart_area):
    loader = LandsatOdcLoader(
        datetime=DATETIME,
        exclude_platforms=None,
        only_tier_one=True,
        fall_back_to_tier_two=False,
    )

    items = list(loader._get_items(hobart_area))

    assert len(items) == 41


def test_hobart_tier_exclude_7(hobart_area):
    loader = LandsatOdcLoader(
        datetime=DATETIME,
        exclude_platforms=["landsat-7"],
        only_tier_one=False,
        fall_back_to_tier_two=False,
    )

    items = list(loader._get_items(hobart_area))

    assert len(items) == 34
