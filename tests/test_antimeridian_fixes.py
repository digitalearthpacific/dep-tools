from geopandas import GeoDataFrame, read_file
from numpy import isclose
from shapely.geometry import LineString, Polygon
import shapely.wkt

from dep_tools.utils import (
    shift_negative_longitudes,
    bbox_across_180,
)

TOLERANCE = 0.01


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
        [(179.0, 1.0), (179.0, -1.0), (-179.0, -1.0), (-179.0, 1.0), (179.0, 1.0)]
    )

    crossing_gdf = GeoDataFrame(geometry=[crossing_polygon], crs=4326)
    bbox = bbox_across_180(crossing_gdf)
    assert isinstance(bbox, list | tuple)
    assert isclose(bbox[0], [179, -1, 180, 1], atol=TOLERANCE).all()
    assert isclose(bbox[1], [-180, -1, -179, 1], atol=TOLERANCE).all()


def test_bbox_across_180_noncrossing():
    noncrossing_polygon = Polygon(
        [(175.0, 1.0), (179.0, 1.0), (179.0, -1.0), (175.0, -1.0), (175.0, 1.0)]
    )
    crossing_gdf = GeoDataFrame(geometry=[noncrossing_polygon], crs=4326)
    bbox = bbox_across_180(crossing_gdf)
    assert isinstance(bbox, list)
    assert bbox == [175.0, -1.0, 179.0, 1.0]


def test_features_on_either_side_of_180():
    features = read_file("tests/features_on_either_side.geojson")
    bbox = bbox_across_180(features)
    assert len(bbox) == 2
    assert isclose(
        bbox[0],
        [179.97098076944076, -19.044135782844712, 180, -18.555752850452095],
        atol=TOLERANCE,
    ).all()
    assert isclose(
        bbox[1], [-180, -19.044135782844712, -179.92643501801794, -18.555752850452095]
    ).all()


def test_geom_split_at_180():
    # This is from
    #    split_geom = GeoDataFrame(
    #        read_file(
    #            "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
    #        )
    #    ).loc[[7759]]
    wkt = "MULTIPOLYGON (((180 -16.581744457409567, 180 -17.994919108280254, 180 -18.07255347222222, 179.3583829354262 -17.9793556231435, 179.358 -17.9793, 179.709 -16.4619, 179.70911279263044 -16.461916385860548, 180 -16.504174795799297, 180 -16.581744457409567)), ((-178.89731077442417 -18.155054868427573, -178.914 -18.2303, -180 -18.07255347222222, -180 -17.994919108280254, -180 -16.581744457409567, -180 -16.5041747957993, -178.5774802586448 -16.710830230803637, -178.577 -16.7109, -178.89731077442417 -18.155054868427573)))"
    split_geom = GeoDataFrame(geometry=[shapely.wkt.loads(wkt)], crs=4326)
    bbox = bbox_across_180(split_geom)
    assert isinstance(bbox, tuple)
    assert bbox[0] == [179.358, -18.2303, 180, -16.4619]
    assert bbox[1] == [-180, -18.2303, -178.577, -16.4619]


def test_geom_tile_across_180():
    # This is from tile 66,22 but with rounded coords
    crossing_polygon = Polygon(
        [
            [179.9, -15.9],
            [179.9, -16.8],
            [180.8, -16.8],
            [180.8, -15.9],
            [179.9, -15.9],
        ]
    )
    gdf = GeoDataFrame(geometry=[crossing_polygon], crs=4326)
    bbox = bbox_across_180(gdf)

    assert isinstance(bbox, tuple)
    assert isclose(bbox[0], [179.9, -16.8, 180, -15.9], atol=TOLERANCE).all()
    assert isclose(bbox[1], [-180, -16.8, -179.2, -15.9], atol=TOLERANCE).all()
