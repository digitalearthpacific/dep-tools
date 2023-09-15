from dep_tools.namers import DepItemPath

sensor = "ls"
dataset_id = "wofs"
version = "1.0.1"
time = "2045"
testItemPath = DepItemPath(sensor, dataset_id, version, time)

item_id = ("FJ", "001")
asset_name = "mean"


def test_path():
    assert (
        testItemPath.path(item_id, asset_name)
        == "dep_ls_wofs/1-0-1/FJ/001/2045/dep_ls_wofs_FJ_001_2045_mean.tif"
    )


def test_log_path():
    assert testItemPath.log_path() == "dep_ls_wofs/1-0-1/logs/dep_ls_wofs_2045_log.csv"


def test_basename():
    assert testItemPath.basename(item_id) == "dep_ls_wofs_FJ_001_2045"


def set_item_prefix():
    assert testItemPath.item_prefix == "dep_ls_wofs"
