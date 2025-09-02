import pytest
import moto
import boto3
import os

@pytest.fixture
def aws_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

@pytest.fixture
def s3(aws_credentials):
    with moto.mock_aws():
        yield boto3.client("s3", region_name="us-west-2")
