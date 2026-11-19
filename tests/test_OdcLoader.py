from geopandas import GeoDataFrame
from odc.geo.geobox import GeoBox
from pystac import Item
from pystac_client import Client
import pytest
from shapely.geometry import box
from xarray import DataArray, Dataset

from dep_tools.loaders import OdcLoader


@pytest.fixture()
def bbox() -> list[float]:
    return [168.3, -17.8, 168.4, -17.6]


@pytest.fixture()
def area_geodataframe(bbox) -> GeoDataFrame:
    return GeoDataFrame(
        geometry=[box(*bbox)],
        crs="EPSG:4326",
    )


@pytest.fixture()
def area_geobox(bbox) -> GeoBox:
    return GeoBox.from_bbox(bbox, crs=4326, resolution=10)


@pytest.fixture()
def items(bbox) -> list[Item]:
    client = Client.open("https://stac.digitalearthpacific.org")
    return list(
        client.search(collections="dep_s2_geomad", bbox=bbox, limit=2).item_collection()
    )


def test_loading(items):
    loader = OdcLoader(chunks=dict(x=256, y=256))
    xr_ds = loader.load(items)
    assert isinstance(xr_ds, Dataset)


def test_loading_geodataframe(items, area_geodataframe):
    loader = OdcLoader(chunks=dict(x=256, y=256))
    xr_ds = loader.load(items=items, areas=area_geodataframe)
    assert isinstance(xr_ds, Dataset)


def test_loading_geobox(items, area_geobox):
    loader = OdcLoader(chunks=dict(x=256, y=256))
    xr_ds = loader.load(items=items, areas=area_geobox)
    assert isinstance(xr_ds, Dataset)


def test_loading_clip(items, area_geodataframe):
    loader = OdcLoader(chunks=dict(x=256, y=256), clip_to_areas=True)
    xr_ds = loader.load(items=items, areas=area_geodataframe)
    assert isinstance(xr_ds, Dataset)


def test_loading_array(items):
    loader = OdcLoader(chunks=dict(x=256, y=256), load_as_dataset=False)
    xr_da = loader.load(items=items)
    assert isinstance(xr_da, DataArray)
