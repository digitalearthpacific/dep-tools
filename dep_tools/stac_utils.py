import numpy as np
import xarray as xr


def set_stac_properties(input_xr, output_xr):
    start_datetime = np.datetime_as_string(
        np.datetime64(input_xr.time.min().values, "Y"), unit="ms"
    )

    end_datetime = np.datetime_as_string(
        np.datetime64(input_xr.time.max().values, "Y")
        + np.timedelta64(1, "Y")
        - np.timedelta64(1, "ns")
    )
    # This _should_ set this attr on the output cog
    output_xr["time"] = start_datetime
    output_xr.attrs["stac_properties"] = dict(
        start_datetime=start_datetime, end_datetime=end_datetime
    )
