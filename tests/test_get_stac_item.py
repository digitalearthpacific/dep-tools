import rioxarray

from dep_tools.stac_utils import _get_stac_item

from test_namers import testItemPath, item_id, asset_name

test_url = "https://deppcpublicstorage.blob.core.windows.net/output/dep_ls_wofs/0-0-2/TV/001/2021/dep_ls_wofs_TV_001_2021_mean.tif"
test_path = "dep_ls_wofs/0-0-2/TV/001/2021/dep_ls_wofs_TV_001_2021_mean.tif"
test_xr = rioxarray.open_rasterio(test_url, chunks=True)
item = _get_stac_item(test_xr, test_path, collection="dep_ls_wofs")
properties = item.properties


def test_get_stac_item_properties():
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


def test_get_stac_item_collection_id():
    assert item.collection_id == "dep_ls_wofs"
