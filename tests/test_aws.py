from pathlib import Path

import boto3
import pytest
from moto import mock_aws
from rioxarray import open_rasterio

from dep_tools.aws import write_to_s3

BUCKET = "test-bucket"
TEST_FOLDER = Path(__file__).parent
DATA_FOLDER = TEST_FOLDER / "data"


@pytest.fixture
def s3():
    """Create an S3 boto3 client and return the client object"""

    s3 = boto3.client("s3", region_name="us-east-1")
    return s3


@pytest.fixture
def dataset():
    data = open_rasterio("tests/data/wofs_suva.tif", default_name="water").squeeze(
        "band"
    )
    return data


@mock_aws
def test_upload_string(s3):
    s3.create_bucket(Bucket=BUCKET)

    write_to_s3("test", f"s3://{BUCKET}/test.txt")

    response = s3.get_object(Bucket=BUCKET, Key="test.txt")
    assert response["Body"].read().decode("utf-8") == "test"


@pytest.mark.xfail(reason="Not working without AWS creds")
@mock_aws(config={"core": {"mock_credentials": True}})
def test_upload_dataset(s3, dataset):
    s3.create_bucket(Bucket=BUCKET)

    write_to_s3(dataset, f"s3://{BUCKET}/test.tif")

    response = s3.get_object(Bucket=BUCKET, Key="test.tif")

    # Check that the file is a tif
    assert response["ContentType"] == "image/tiff"
