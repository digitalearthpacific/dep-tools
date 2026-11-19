from datetime import datetime, timezone

import numpy as np
import pytest
from xarray import DataArray, Dataset

from dep_tools.stac_utils import set_stac_properties


def _props(*dates, **kwargs):
    x = DataArray(
        np.zeros(len(dates)),
        coords={"time": np.array(dates, dtype="datetime64[ns]")},
        dims="time",
    )
    return set_stac_properties(x, Dataset(), **kwargs).attrs["stac_properties"]


def _get(props):
    return props["start_datetime"], props["datetime"], props["end_datetime"]


# From the data (no period given)

# Annual GeoMAD: Jan 1 to Dec 31, centred on the middle of the year.


def test_annual_is_jan_1_to_dec_31_with_datetime_at_centre():
    # scenes through the year, none exactly on Jan 1 or Dec 31
    start, centre, end = _get(_props("2023-01-04", "2023-06-15", "2023-12-27"))
    assert start == "2023-01-01T00:00:00Z"
    assert end == "2023-12-31T23:59:59Z"
    assert centre == "2023-07-02T12:00:00Z"  # 365 days, so noon on 2 July


def test_annual_leap_year_centre():
    start, centre, end = _get(_props("2024-01-04", "2024-12-27"))
    assert (start, end) == ("2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z")
    assert centre == "2024-07-02T00:00:00Z"  # 366 days


def test_annual_with_no_scenes_in_first_and_last_months_is_still_a_full_year():
    # cloudy Jan and Dec: the data only spans Feb to Nov, but it is an annual product
    start, _, end = _get(_props("2023-02-10", "2023-06-15", "2023-11-20"))
    assert (start, end) == ("2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z")


def test_multi_year_input_spans_whole_years_and_centres_on_the_middle():
    start, centre, end = _get(_props("2011-05-01", "2012-05-01", "2013-05-01"))
    assert (start, end) == ("2011-01-01T00:00:00Z", "2013-12-31T23:59:59Z")
    assert centre == "2012-07-02T00:00:00Z"  # 1096 days from 1 Jan 2011


# Six month GeoMAD: first day of the first month to last day of the last month.


@pytest.mark.parametrize(
    "dates,start,end,centre",
    [
        # Jan to Jun
        (
            ("2024-01-06", "2024-03-30", "2024-06-27"),
            "2024-01-01T00:00:00Z",
            "2024-06-30T23:59:59Z",
            "2024-04-01T00:00:00Z",  # 182 days (leap year): 91 days in
        ),
        # Jul to Dec
        (
            ("2024-07-03", "2024-12-29"),
            "2024-07-01T00:00:00Z",
            "2024-12-31T23:59:59Z",
            "2024-10-01T00:00:00Z",  # 184 days: 92 days in
        ),
        # not calendar half years: Apr to Sep
        (
            ("2024-04-09", "2024-07-01", "2024-09-25"),
            "2024-04-01T00:00:00Z",
            "2024-09-30T23:59:59Z",
            "2024-07-01T12:00:00Z",
        ),
        # spanning a year boundary: Oct to Mar
        (
            ("2023-10-12", "2024-03-20"),
            "2023-10-01T00:00:00Z",
            "2024-03-31T23:59:59Z",
            "2023-12-31T12:00:00Z",
        ),
    ],
)
def test_six_months_uses_whole_months_and_centre(dates, start, end, centre):
    assert _get(_props(*dates)) == (start, centre, end)


def test_single_month():
    start, centre, end = _get(_props("2024-03-04", "2024-03-28"))
    assert (start, end) == ("2024-03-01T00:00:00Z", "2024-03-31T23:59:59Z")
    assert centre == "2024-03-16T12:00:00Z"


# Other (from the data)


def test_nominal_datetime_overrides_the_centre_only():
    # e.g. a nominal year made from buffered, multi-year inputs
    start, centre, end = _get(
        _props(
            "2011-05-01",
            "2012-05-01",
            "2013-05-01",
            nominal_datetime="2012-06-30T00:00:00Z",
        )
    )
    assert (start, end) == ("2011-01-01T00:00:00Z", "2013-12-31T23:59:59Z")
    assert centre == "2012-06-30T00:00:00Z"


def test_created_is_utc():
    created = datetime.strptime(
        _props("2020-03-01")["created"], "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    age = datetime.now(timezone.utc) - created.replace(tzinfo=timezone.utc)
    assert abs(age.total_seconds()) < 60


# From the period (the intention), whatever data was found


@pytest.mark.parametrize(
    "period,start,centre,end",
    [
        # annual, leap year and not
        (
            "2024",
            "2024-01-01T00:00:00Z",
            "2024-07-02T00:00:00Z",
            "2024-12-31T23:59:59Z",
        ),
        (
            "2023",
            "2023-01-01T00:00:00Z",
            "2023-07-02T12:00:00Z",
            "2023-12-31T23:59:59Z",
        ),
        # six months, as an interval
        (
            "2024-01/2024-06",
            "2024-01-01T00:00:00Z",
            "2024-04-01T00:00:00Z",
            "2024-06-30T23:59:59Z",
        ),
        (
            "2024-04/2024-09",
            "2024-04-01T00:00:00Z",
            "2024-07-01T12:00:00Z",
            "2024-09-30T23:59:59Z",
        ),
        # other granularities
        (
            "2024Q1",
            "2024-01-01T00:00:00Z",
            "2024-02-15T12:00:00Z",
            "2024-03-31T23:59:59Z",
        ),
        (
            "2024-03",
            "2024-03-01T00:00:00Z",
            "2024-03-16T12:00:00Z",
            "2024-03-31T23:59:59Z",
        ),
        (
            "2020/2022",
            "2020-01-01T00:00:00Z",
            "2021-07-02T00:00:00Z",
            "2022-12-31T23:59:59Z",
        ),
    ],
)
def test_period_gives_exact_start_end_and_centre(period, start, centre, end):
    # the data covers only a sliver of the period, and it doesn't matter
    assert _get(_props("2024-03-04", "2024-03-10", period=period)) == (
        start,
        centre,
        end,
    )


def test_period_matches_data_when_data_fills_the_period():
    from_data = _get(_props("2024-01-04", "2024-12-27"))
    assert _get(_props("2020-01-01", period="2024")) == from_data


def test_window_sets_start_and_end_but_datetime_stays_at_the_period_centre():
    # nominal 2012, inputs 2011-2013
    start, centre, end = _get(_props("2012-05-01", period="2012", window="2011/2013"))
    assert (start, end) == ("2011-01-01T00:00:00Z", "2013-12-31T23:59:59Z")
    assert (
        centre == "2012-07-02T00:00:00Z"
    )  # centre of 2012 (366 days), not of the window


def test_window_alone_centres_on_the_window():
    start, centre, end = _get(_props("2012-05-01", window="2011/2013"))
    assert (start, end) == ("2011-01-01T00:00:00Z", "2013-12-31T23:59:59Z")
    assert centre == "2012-07-02T00:00:00Z"


def test_nominal_datetime_overrides_the_centre_for_a_period_too():
    props = _props("2012-05-01", period="2012", nominal_datetime="2012-06-30T00:00:00Z")
    assert props["datetime"] == "2012-06-30T00:00:00Z"
    assert props["start_datetime"] == "2012-01-01T00:00:00Z"


@pytest.mark.parametrize("bad", ["not a date", "2024/..", ""])
def test_bad_period_raises(bad):
    with pytest.raises(Exception):
        _props("2024-03-04", period=bad)
