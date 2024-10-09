import rioxarray

from dep_tools.namers import LocalPath
from dep_tools.stac_utils import StacCreator
from dep_tools.writers import StacWriter

from dep_tools.stac_utils import get_stac_item
from pathlib import Path
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def stac_item():
    itempath = LocalPath(
        str(DATA_DIR),
        sensor="spysat",
        dataset_id="wofs",
        version="1.0.0",
        time="2021-01-01",
        zero_pad_numbers=False,
    )
    tif = itempath.path("12,34", asset_name="wofs")
    test_xr = rioxarray.open_rasterio(tif).to_dataset(name="wofs")
    item = get_stac_item(
        itempath=itempath,
        item_id="12,34",
        data=test_xr,
        remote=False
    )

    return item


def test_get_stac_item_properties(stac_item):
    properties = stac_item.properties
    keys = [
        "proj:epsg",
        "proj:geometry",
        "proj:bbox",
        "proj:shape",
        "proj:transform",
        "proj:projjson",
    ]
    assert all([key in properties.keys() for key in keys])


def test_get_stac_item_collection_id(stac_item):
    assert stac_item.collection_id == "dep_spysat_wofs"


def test_stac_asset_href_is_valid(stac_item):
    assert stac_item.assets["wofs"].href == str(
        DATA_DIR
        / "dep_spysat_wofs/1-0-0/12/34/2021-01-01/dep_spysat_wofs_12_34_2021-01-01_wofs.tif"
    )
