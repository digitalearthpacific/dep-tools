from pathlib import Path
from pystac import Item
import pytest
import rioxarray
from xarray import Dataset

from dep_tools.namers import GenericItemPath

from dep_tools.stac_utils import get_stac_item
from dep_tools.namers import S3ItemPath
from dep_tools.utils import join_path_or_url

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def itempath() -> GenericItemPath:
    return GenericItemPath(
        sensor="spysat",
        dataset_id="wofs",
        version="1.0.0",
        time="2021-01-01",
        zero_pad_numbers=False,
        full_path_prefix=str(DATA_DIR),
    )


@pytest.fixture
def dataset(itempath: GenericItemPath) -> Dataset:
    tif = itempath.path("12,34", asset_name="wofs", absolute=True)
    return rioxarray.open_rasterio(tif).to_dataset(name="wofs")


@pytest.fixture
def stac_item(itempath: GenericItemPath, dataset: Dataset) -> Item:
    item = get_stac_item(itempath=itempath, item_id="12,34", data=dataset)
    return item


def test_dont_set_geometry_from_input(stac_item):
    # ensure the geometry is set from the data by default
    assert stac_item.geometry == {
        "type": "Polygon",
        "coordinates": [
            [
                (178.419653974838, -18.27433771663508),
                (178.73756194627958, -18.27433771663508),
                (178.73756194627958, -17.97026123112962),
                (178.419653974838, -17.97026123112962),
                (178.419653974838, -18.27433771663508),
            ]
        ],
    }


def test_set_geometry_from_input(itempath, dataset):
    # test that we can set the geometry to what we want if necessary
    test_geometry = {"type": "Point", "coordinates": [0, 0]}
    test_dataset = dataset.assign_attrs(
        stac_properties={
            "datetime": "2020-01-01T12:00:00.123Z",
            "proj:geometry": test_geometry,
        }
    )
    stac_item = get_stac_item(
        itempath, "12,34", test_dataset, set_geometry_from_input=True
    )
    assert stac_item.geometry == test_geometry


def test_get_stac_item_properties(stac_item):
    properties = stac_item.properties
    keys = [
        "proj:epsg",
        "proj:geometry",
        "proj:bbox",
        "proj:shape",
        "proj:transform",
    ]
    assert all([key in properties.keys() for key in keys])


def test_get_stac_item_collection_id(stac_item):
    assert stac_item.collection_id == "dep_spysat_wofs"


def test_stac_asset_href_is_valid(stac_item):
    assert stac_item.assets["wofs"].href == str(
        DATA_DIR
        / "dep_spysat_wofs/1-0-0/12/34/2021-01-01/dep_spysat_wofs_12_34_2021-01-01_wofs.tif"
    )


def test_join_path_or_url_file():
    joined = join_path_or_url("/home/data/", "test.txt")
    assert joined == "/home/data/test.txt"


def test_join_path_or_url_slashes_everywhere():
    joined = join_path_or_url("/home/data/", "/test.txt")
    assert joined == "/home/data/test.txt"


def test_join_path_or_url_s3():
    joined = join_path_or_url("s3://home/data", "test.txt")
    assert joined == "s3://home/data/test.txt"


def test_join_path_or_url_https():
    joined = join_path_or_url("https://home.com/data", "test.txt")
    assert joined == "https://home.com/data/test.txt"


def test_stac_url():
    item_path = S3ItemPath(
        "test-bucket",
        "nose",
        "aroma",
        "test",
        "2024",
        make_hrefs_https=True,
        full_path_prefix="https://test.com/",
    )

    assert item_path.full_path_prefix == "https://test.com/"

    assert (
        item_path.stac_path((99, 66), absolute=True)
        == "https://test.com/dep_nose_aroma/test/099/066/2024/dep_nose_aroma_099_066_2024.stac-item.json"
    )
