from pystac_client import Client
from pystac import Item

from dep_tools.stac_utils import use_alternate_s3_href


def test_patching():
    client = Client.open(
        "https://landsatlook.usgs.gov/stac-server",
        modifier=use_alternate_s3_href,
    )
    item = list(
        client.search(ids=["LC08_L2SR_076068_20201231_20210308_02_T2_SR"]).items()
    )[0]
    assert (
        item.assets["swir16"].href
        == "s3://usgs-landsat-ard/collection02/level-2/standard/oli-tirs/2020/076/068/LC08_L2SR_076068_20201231_20210308_02_T2/LC08_L2SR_076068_20201231_20210308_02_T2_SR_B6.TIF"
    )
