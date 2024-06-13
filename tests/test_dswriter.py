from dep_tools.writers import DsWriter
from dep_tools.namers import DepItemPath


def test_kwargs():
    itempath = DepItemPath(
        sensor="spc_sat", dataset_id="clear_waters", version="1.0", time="1901"
    )
    writer = DsWriter(itempath=itempath, bucket="dep_bucket")
    assert isinstance(writer, DsWriter)
