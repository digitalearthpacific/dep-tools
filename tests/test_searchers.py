import geopandas as gpd
import odc.stac
from pystac import ItemCollection
import planetary_computer as pc
import pytest
from shapely.geometry import box
from xarray import Dataset

from dep_tools.searchers import PystacSearcher, LandsatPystacSearcher

BBOX = [-111, 35, -110, 36]


@pytest.fixture()
def area() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[box(*BBOX)],
        crs="EPSG:4326",
    )


@pytest.fixture
def mspc_catalog():
    return "https://planetarycomputer.microsoft.com/api/stac/v1"


def test_PystacSearcher(area, mspc_catalog):
    s = PystacSearcher(
        catalog=mspc_catalog, collections=["landsat-c2-l2"], datetime="2007"
    )
    items = s.search(area)
    assert isinstance(items, ItemCollection)

    an_item = list(items)[0]
    assert an_item.properties["description"] == "Landsat Collection 2 Level-2"


def test_LandsatPystacSearcher_exclude_platforms(area):
    s = LandsatPystacSearcher(
        exclude_platforms=["landsat-7"],
        catalog="https://planetarycomputer.microsoft.com/api/stac/v1/",
    )
    s.search(area)


def test_unsigned_search(area, mspc_catalog):
    s = PystacSearcher(
        catalog=mspc_catalog, collections=["landsat-c2-l2"], datetime="2007"
    )
    items = s.search(area)
    ds = odc.stac.load([items[0]], patch_url=pc.sign, chunks=dict(x=2048, y=2048))
    assert isinstance(ds, Dataset)
