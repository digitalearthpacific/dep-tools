from shapely.geometry import Polygon, MultiPolygon
from dep_tools.utils import fix_winding


def test_fix_winding():
    bad_poly = Polygon(shell=[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    assert not bad_poly.exterior.is_ccw
    assert fix_winding(bad_poly).exterior.is_ccw
