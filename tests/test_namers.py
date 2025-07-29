from dep_tools.namers import DailyItemPath, DepItemPath

bucket = "dep-staging-public"
sensor = "ls"
dataset_id = "wofs"
version = "1.0.1"
time = "2045"
testItemPath = DepItemPath(sensor, dataset_id, version, time)
paddedItemPath = DepItemPath(sensor, dataset_id, version, time, zero_pad_numbers=True)
nonPaddedItemPath = DepItemPath(
    sensor, dataset_id, version, time, zero_pad_numbers=False
)
item_id = "001,002"
asset_name = "mean"


def test_path():
    assert (
        testItemPath.path(item_id, asset_name)
        == "dep_ls_wofs/1-0-1/001/002/2045/dep_ls_wofs_001_002_2045_mean.tif"
    )


def test_log_path():
    assert testItemPath.log_path() == "dep_ls_wofs/1-0-1/logs/dep_ls_wofs_2045_log.csv"


def test_basename():
    assert testItemPath.basename(item_id) == "dep_ls_wofs_001_002_2045"


def set_item_prefix():
    assert testItemPath.item_prefix == "dep_ls_wofs"


def test_padded_format_item_id_iterable():
    assert paddedItemPath._format_item_id(("66", "23"), "_") == "066_023"


def test_padded_format_item_id_iterable_int():
    assert paddedItemPath._format_item_id((66, 23), "_") == "066_023"


def test_padded_format_item_id_list_as_string():
    assert paddedItemPath._format_item_id("001,002", "_") == "001_002"


def test_padded_format_item_id_string():
    assert paddedItemPath._format_item_id("66", "_") == "066"


def test_format_item_id_iterable():
    assert testItemPath._format_item_id(("66", "23"), "_") == "066_023"


def test_format_item_id_iterable_int():
    assert testItemPath._format_item_id((66, 23), "_") == "066_023"


def test_format_item_id_list_as_string():
    assert testItemPath._format_item_id("001,002", "_") == "001_002"


def test_format_item_id_string():
    assert testItemPath._format_item_id("66", "_") == "066"


def test_non_padded_format_item_id_iterable():
    assert nonPaddedItemPath._format_item_id(("66", "23"), "_") == "66_23"


def test_non_padded_format_item_id_iterable_int():
    assert nonPaddedItemPath._format_item_id((66, 23), "_") == "66_23"


def test_non_padded_format_item_id_list_as_string():
    assert nonPaddedItemPath._format_item_id("001,002", "_") == "001_002"


dailyItemPath = DailyItemPath(
    bucket=bucket,
    sensor=sensor,
    dataset_id=dataset_id,
    version=version,
    time="2025-07-28 15:44:14.926241",
)


def test_daily_path():
    assert (
        dailyItemPath.path(item_id, asset_name)
        == "dep_ls_wofs/1-0-1/001/002/2025/07/28/dep_ls_wofs_001_002_2025-07-28_mean.tif"
    )
