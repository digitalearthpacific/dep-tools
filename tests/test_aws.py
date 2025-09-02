from dep_tools.aws import get_s3_bucket_region

TEST_BUCKET_NAME = "dep-test-bucket"


def test_get_s3_bucket_region(s3):
    s3.create_bucket(Bucket=TEST_BUCKET_NAME, CreateBucketConfiguration={
        'LocationConstraint': 'us-west-2'})
    region = get_s3_bucket_region(TEST_BUCKET_NAME)
    assert region == "us-west-2"


# def test_write_to_s3_kwargs():
#    bucket = "dep-public-staging"
#    key = "test.gpkg"
#    d = GeoDataFrame(geometry=[box(-170, 0, -169, 1)])
#    write_to_s3(d, path=key, bucket=bucket, driver="GPKG")
#    assert object_exists(bucket, key)
