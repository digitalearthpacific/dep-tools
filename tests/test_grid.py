import warnings

from dep_tools.grids import get_tiles, gadm, PACIFIC_EPSG
from json import loads
import urllib.error

from odc.geo import Geometry
import pytest


@pytest.mark.xfail(
    raises=urllib.error.URLError,
    reason="GADM load failed, source server is likely down",
)
def test_get_gadm():
    all = gadm()

    assert len(all) == 22

    # Convert to a ODC Geometry
    geom = Geometry(loads(all.to_json()), crs=all.crs)
    assert geom.crs == 4326


@pytest.mark.xfail(
    raises=urllib.error.URLError,
    reason="GADM load failed, source server is likely down",
)
def test_get_tiles():
    # This takes 2 minutes to retrieve all the tiles. Keep it as a generator.
    tiles = get_tiles(resolution=30)

    # Let's just check one
    # Each item in the generator is a tuple with the tile index and the geobox object
    tile, geobox = next(tiles)
    assert type(tile) is tuple
    assert geobox.crs == PACIFIC_EPSG

    # Check the count here, shouldn't take too long
    tiles = list(get_tiles(resolution=30, country_codes=["FJI"]))
    assert len(tiles) == 27

    print("c")
