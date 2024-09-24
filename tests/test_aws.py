from geopandas import GeoDataFrame
from shapely import box

from dep_tools.aws import write_to_s3, object_exists


# def test_write_to_s3_kwargs():
#    bucket = "dep-public-staging"
#    key = "test.gpkg"
#    d = GeoDataFrame(geometry=[box(-170, 0, -169, 1)])
#    write_to_s3(d, path=key, bucket=bucket, driver="GPKG")
#    assert object_exists(bucket, key)
