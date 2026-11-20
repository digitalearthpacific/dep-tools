from unittest.mock import MagicMock

import numpy as np
from xarray import DataArray, Dataset

from dep_tools.loaders import StacLoader
from dep_tools.processors import Processor
from dep_tools.task import StacTask


class _Loader(StacLoader):
    def load(self, items, areas):
        # a few scenes in the middle of 2024 only
        time = np.array(["2024-03-04", "2024-03-10"], dtype="datetime64[ns]")
        return DataArray(np.zeros(2), coords={"time": time}, dims="time")


class _Processor(Processor):
    def process(self, xr):
        return Dataset()


def _run(**kwargs):
    writer = MagicMock()
    task = StacTask(
        id="1,2",
        area=None,
        searcher=MagicMock(),
        loader=_Loader(),
        processor=_Processor(),
        writer=writer,
        **kwargs,
    )
    task.run()
    return writer.write.call_args.args[0].attrs["stac_properties"]


def test_period_is_used_for_the_stac_dates():
    props = _run(period="2024")
    assert props["start_datetime"] == "2024-01-01T00:00:00Z"
    assert props["end_datetime"] == "2024-12-31T23:59:59Z"
    assert props["datetime"] == "2024-07-02T00:00:00Z"


def test_without_a_period_the_dates_come_from_the_data():
    props = _run()
    assert props["start_datetime"] == "2024-03-01T00:00:00Z"
    assert props["end_datetime"] == "2024-03-31T23:59:59Z"
