from geopandas import GeoDataFrame
from shapely.geometry import LineString, Polygon

from dep_tools.utils import (
    shift_negative_longitudes,
    bbox_across_180,
    search_across_180,
)


def test_shift_negative_longitudes_crossing_linestring():
    crossing_linestring = LineString([(179.0, 1.0), (-179.0, 2.0)])
    fixed_linestring = shift_negative_longitudes(crossing_linestring)
    assert list(fixed_linestring.coords) == [(179.0, 1.0), (181.0, 2.0)]


def test_shift_negative_longitudes_noncrossing_linestring():
    coords = [(172.0, 1.0), (179.0, 2.0)]
    noncrossing_linestring = LineString(coords)
    fixed_linestring = shift_negative_longitudes(noncrossing_linestring)
    assert list(fixed_linestring.coords) == coords


def test_bbox_across_180_crossing():
    crossing_polygon = Polygon(
        [(179.0, 1.0), (-179.0, 1.0), (-179.0, -1.0), (179.0, -1.0), (179.0, 1.0)]
    )
    crossing_gdf = GeoDataFrame(geometry=[crossing_polygon], crs=4326)
    bbox = bbox_across_180(crossing_gdf)
    assert isinstance(bbox, tuple)
    assert bbox[0] == [179.0, -1.0, 180, 1.0]
    assert bbox[1] == [-180, -1.0, -179.0, 1.0]


def test_bbox_across_180_noncrossing():
    noncrossing_polygon = Polygon(
        [(175.0, 1.0), (179.0, 1.0), (179.0, -1.0), (175.0, -1.0), (175.0, 1.0)]
    )
    crossing_gdf = GeoDataFrame(geometry=[noncrossing_polygon], crs=4326)
    bbox = bbox_across_180(crossing_gdf)
    assert isinstance(bbox, list)
    assert bbox == [175.0, -1.0, 179.0, 1.0]
