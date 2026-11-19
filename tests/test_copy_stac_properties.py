import odc.stac
from pystac_client import Client
import pytest

from dep_tools.stac_utils import copy_stac_properties, use_alternate_s3_href


@pytest.fixture
def item():
    client = Client.open(
        "https://landsatlook.usgs.gov/stac-server",
        modifier=use_alternate_s3_href,
    )
    return next(
        client.search(ids=["LC08_L2SR_076068_20201231_20210308_02_T2_SR"]).items()
    )


@pytest.fixture
def ds(item):
    return odc.stac.load([item], chunks=dict(x=1024, y=1024))


def test_copy_stac_properties(item, ds):
    output = copy_stac_properties(item, ds)
    props = item.properties.copy()
    output_props = output.attrs["stac_properties"]
    props["start_datetime"] = output_props["start_datetime"]
    props["end_datetime"] = output_props["end_datetime"]
    assert props == output_props


def test_copy_stac_properties_with_existings(item, ds):
    ds.attrs["stac_properties"] = {"a": 1}
    output = copy_stac_properties(item, ds)
    output_props = output.attrs["stac_properties"]
    props = item.properties.copy()
    props["start_datetime"] = output_props["start_datetime"]
    props["end_datetime"] = output_props["end_datetime"]
    props["a"] = 1
    assert props == output.attrs["stac_properties"]
