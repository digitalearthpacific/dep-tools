import rioxarray

from dep_tools.stac_utils import _get_stac_item

import pytest

TEST_ITEM_URL = "https://deppcpublicstorage.blob.core.windows.net/output/dep_ls_wofs/0-0-2/TV/001/2021/dep_ls_wofs_TV_001_2021_mean.tif"


@pytest.fixture
def stac_item():
    test_path = "dep_ls_wofs/0-0-2/TV/001/2021/dep_ls_wofs_TV_001_2021_mean.tif"
    test_xr = rioxarray.open_rasterio(TEST_ITEM_URL, chunks=True)
    item = _get_stac_item(test_xr, test_path, collection="dep_ls_wofs")
    return item


def test_get_stac_item_properties(stac_item):
    properties = stac_item.properties
    keys = [
        "start_datetime",
        "end_datetime",
        "proj:epsg",
        "proj:geometry",
        "proj:bbox",
        "proj:shape",
        "proj:transform",
        "proj:projjson",
    ]
    assert all([key in properties.keys() for key in keys])


def test_get_stac_item_collection_id(stac_item):
    assert stac_item.collection_id == "dep_ls_wofs"


def test_stac_asset_href_is_valid(stac_item):
    assert stac_item.assets["asset"].href == TEST_ITEM_URL
