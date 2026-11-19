"""This module contains classes and functions to aid in STAC Item creation."""

from datetime import datetime, timezone
import json
import os
import warnings

import numpy as np
from pandas import DataFrame, Period
from pystac import Asset, Item, Link, MediaType, read_dict
import pystac_client
import rasterio
from rio_stac.stac import create_stac_item, get_raster_info
from xarray import DataArray, Dataset


from .aws import object_exists
from .namers import GenericItemPath, S3ItemPath
from .processors import Processor

PROCESSING_SCHEMA = "https://stac-extensions.github.io/processing/v1.3.0/schema.json"


def set_derived_from(items: list[Item], ds: DataArray | Dataset):
    """Record the canonical hrefs of source `items` in `ds.attrs["stac_derived_from"]`,
    for :func:`get_stac_item` to turn into `derived_from` links."""
    ds.attrs["stac_derived_from"] = [
        i.get_self_href() for i in items if i.get_self_href() is not None
    ]
    return ds


def _software_from_env() -> dict[str, str]:
    """Processing software info from the IMAGE_TAG and COMMIT environment
    variables. Unset or empty variables are skipped; there are no defaults."""
    names = {"docker-image": "IMAGE_TAG", "commit": "COMMIT"}
    return {k: os.environ[v] for k, v in names.items() if os.environ.get(v)}


class StacCreator(Processor):
    def __init__(
        self,
        itempath: GenericItemPath,
        collection_url_root: str = "https://stac.staging.digitalearthpacific.io/collections",
        **kwargs,
    ):
        """A :class:`Processor` which creates STAC Items.

        This is a wrapper around :func:`get_stac_item`.

        Args:
            itempath: A :class:`GenericItemPath` used to identify the output.
            collection_url_root: The collections endpoint of the target STAC
                catalog.
            **kwargs: Additional arguments to :func:`get_stac_item`.
        """
        self._itempath = itempath
        self._collection_url_root = collection_url_root
        self._kwargs = kwargs

    def process(
        self,
        data: DataArray | Dataset,
        item_id: str,
    ) -> Item | str:
        return get_stac_item(
            itempath=self._itempath,
            item_id=item_id,
            data=data,
            collection_url_root=self._collection_url_root,
            **self._kwargs,
        )


def get_stac_item(
    itempath: GenericItemPath,
    item_id: str,
    data: DataArray | Dataset,
    collection_url_root: str = "https://stac.staging.digitalearthpacific.io/collections",
    set_geometry_from_input: bool = False,
    asset_overrides: dict[str, dict] | None = None,
    disable_derived_from: bool = False,
    **kwargs,
) -> Item:
    """Create a STAC Item.

    This is a wrapper around :func:`rio_stac.create_stac_item` which adds the folllowing
    functionality:

    - It copies information at the "stac_properties" attribute of `data` to the
      properties of the output.
    - If the kwargs include "with_raster=True" and `data` is an
      :class:`xarray.Dataset`, raster information is created for each variable.
    - The self href of the output is set, using `itempath.stac_path`.
    - Provenance: `processing:datetime` is set to the current UTC time (processing
      extension). `derived_from` links are added for hrefs in the
      `stac_derived_from` attribute of `data` (see :func:`set_derived_from`).
      If the `IMAGE_TAG` / `COMMIT` environment variables are set they are
      recorded as `processing:software` (processing extension), and
      `IMAGE_REPO` as a `via` link.

    Args:
        itempath: The :class:`ItemPath` for the raster output.
        item_id: The identifier passed to `itempath.path`.
        data: The data for which this item is to be created. Anything at the
            `stac_properties` attribute will be copied to the properties of
            the output.
        collection_url_root: The URL to the collections endpoint where this STAC Item
            belongs.
        set_geometry_from_input: Whether the geometry property should be copied from
            the `stac_properties` attribute of `data`. If this property is missing,
            a warning is emitted and nothing is set.
        asset_overrides: Optional per-asset (variable name) overrides of the
            default asset fields, e.g.
            `{"visual": {"roles": ["visual"], "title": "True colour image"}}`.
            Keys can be any :class:`pystac.Asset` argument (such as `roles`, `title`,
            and `media_type`), except `href` and `extra_fields`.
        disable_derived_from: Set to True to not add `derived_from` links (which
            can be numerous, e.g. one for every scene in an annual composite).
            They are added by default.
        **kwargs: Additional arguments to :func:`rio_stac.create_stac_item`.

    Returns:
        A STAC Item.
    """
    properties = {}
    if "stac_properties" in data.attrs:
        # Copying properties or `create_stac_item` modifies them.
        data_properties = data.attrs["stac_properties"].copy()
        properties = (
            json.loads(data_properties.replace("'", '"'))
            if isinstance(data.attrs["stac_properties"], str)
            else data_properties
        )

    assets = {}
    for variable in data:
        raster_info = {}
        full_path = itempath.path(item_id, variable, absolute=True)
        if kwargs.get("with_raster"):
            with rasterio.open(full_path) as src_dst:
                raster_info = {"raster:bands": get_raster_info(src_dst, max_size=1024)}

        asset_args = {
            "media_type": MediaType.COG,
            "roles": ["data"],
            **(asset_overrides or {}).get(variable, {}),
        }
        assets[variable] = Asset(
            href=full_path, extra_fields={**raster_info}, **asset_args
        )
    stac_id = itempath.basename(item_id)
    collection = itempath.item_prefix
    collection_url = f"{collection_url_root}/{collection}"

    input_datetime = properties.get("datetime", None)
    if input_datetime is not None:
        format_string = (
            "%Y-%m-%dT%H:%M:%S.%fZ" if "." in input_datetime else "%Y-%m-%dT%H:%M:%SZ"
        )
        input_datetime = datetime.strptime(input_datetime, format_string)

    an_href = next(iter(assets.values())).href
    item = create_stac_item(
        an_href,
        id=stac_id,
        input_datetime=input_datetime,
        assets=assets,
        with_proj=True,
        properties=properties,
        collection_url=collection_url,
        collection=collection,
        **kwargs,
    )
    if not disable_derived_from:
        for href in data.attrs.get("stac_derived_from", []):
            item.add_link(Link("derived_from", href, media_type="application/geo+json"))

    item.properties["processing:datetime"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    item.stac_extensions.append(PROCESSING_SCHEMA)
    software = _software_from_env()
    if software:
        item.properties["processing:software"] = software
    if image_repo := os.environ.get("IMAGE_REPO"):
        item.add_link(Link("via", image_repo, title="Processing code"))

    if set_geometry_from_input:
        if "stac_geometry" in data.attrs:
            item.geometry = data.attrs["stac_geometry"]
        else:
            warnings.warn(
                "set_geometry_from_input=True, but no geometry found in 'stac_properties' attribute of input data, skipping."
            )

    item.set_self_href(itempath.stac_path(item_id, absolute=True))

    return item


def write_stac_local(item: Item, stac_path: str) -> None:
    """Writes a STAC Item to a local file.

    The item is converted to json with appropriate indentation.

    Args:
        item: A STAC Item.
        stac_path: The output path.
    """
    with open(stac_path, "w") as f:
        json.dump(item.to_dict(), f, indent=4)


def existing_stac_items(possible_ids: list, itempath: S3ItemPath) -> list:
    """Return the list of input IDs which have an existing STAC Item.

    Args:
        possible_ids: Identifiers, each passed as the first argument
            to `itempath.stac_path`.
        itempath: The S3ItemPath which identifies the STAC Items to compare.

    Returns:
        A list of IDs, filtered to those for which a STAC Item already exists.
    """
    return [
        id
        for id in possible_ids
        if object_exists(itempath.bucket, itempath.stac_path(id))
    ]


def remove_items_with_existing_stac(grid: DataFrame, itempath: S3ItemPath) -> DataFrame:
    """Filter a dataframe to only include items which don't have an existing stac output.

    Args:
        grid: A DataFram, whose index corresponds to ids for the given itempath.
        itempath: The :class:`S3ItemPath` used to identify Items.

    Returns:
        The input :class:`DataFrame`, only including rows which don't already
        have a STAC Item (as identified by the dataframe index).
    """
    return grid[~grid.index.isin(existing_stac_items(list(grid.index), itempath))]


def _period_bounds(period: str) -> tuple[np.datetime64, np.datetime64]:
    """Start and (exclusive) end of an ISO period, as `datetime64[s]`.

    `period` is a partial ISO date understood by :class:`pandas.Period`
    (e.g. `"2024"`, `"2024-03"`, `"2024Q1"`, `"2024-03-05"`) or an interval of
    two of them separated by `/` (e.g. `"2020/2022"`, `"2024-04/2024-09"`).
    """
    first, _, last = period.partition("/")
    return (
        Period(first).start_time.to_datetime64().astype("datetime64[s]"),
        Period(last or first)
        .end_time.floor("s")
        .to_datetime64()
        .astype("datetime64[s]")
        + np.timedelta64(1, "s"),
    )


def _data_bounds(time: DataArray) -> tuple[np.datetime64, np.datetime64]:
    """Start and (exclusive) end of the period covered by `time`, snapped
    outwards to whole years if it spans more than about half a year, otherwise to
    whole months."""
    first, last = time.min().values, time.max().values
    # A six month period spans at most 184 days between its first and last scene
    unit = "M" if (last - first) <= np.timedelta64(190, "D") else "Y"
    return (
        np.datetime64(first, unit).astype("datetime64[s]"),
        (np.datetime64(last, unit) + np.timedelta64(1, unit)).astype("datetime64[s]"),
    )


def set_stac_properties(
    input_xr: DataArray | Dataset,
    output_xr: DataArray | Dataset,
    period: str | None = None,
    window: str | None = None,
    nominal_datetime: str | None = None,
) -> Dataset | DataArray:
    """Set the `"stac_properties"` attribute of an Xarray object based on another.

    Sets an attribute called `"stac_properties"` in the output which is a
    dictionary containing the following properties for use in stac writing:
    `"start_datetime"`, `"end_datetime"`, `"datetime"`, and `"created"` (the
    current time in UTC). Typically, `input_xr` would be an array of EO data
    (e.g. Landsat) containing data over a range of dates (such as a year).

    The preferred way to set the dates is to say what period the output
    represents with `period`. The start and end are then exact, whatever scenes
    happened to be found, and `"datetime"` is the centre of the period.

    Without `period` (or `window`), the period is worked out from the
    `input_xr.time` coordinate, snapped outwards to calendar boundaries so that
    a sparse start or end (e.g. a cloudy January) doesn't shorten it: whole
    years (Jan 1 to Dec 31) if the data spans more than about half a year,
    otherwise whole months (first day of the first month to last day of the last
    month). `"datetime"` is the centre of that.

    Args:
        input_xr: The object whose `time` coordinate is used if neither `period`
            nor `window` is given.
        output_xr: The object whose properties are set.
        period: The period the output represents, as a partial ISO date
            (`"2024"`, `"2024-03"`, `"2024Q1"`) or an interval of two of them
            separated by `/` (`"2024-04/2024-09"`).
        window: The time span of the input observations, in the same format, if
            it is different from `period`, e.g. `"2011/2013"` for a nominal
            2012 made from inputs with a buffer year either side. Sets
            `"start_datetime"` and `"end_datetime"`; `"datetime"` is still the
            centre of `period` (or of `window` if there is no `period`).
        nominal_datetime: An exact value for `"datetime"` (an RFC 3339 string),
            instead of the centre.

    Returns:
        `output_xr` with the `"stac_properties"` attribute set as described.
    """
    if period is None and window is None:
        start, end = centre_span = _data_bounds(input_xr.time)
    else:
        centre_span = _period_bounds(period or window)
        start, end = _period_bounds(window) if window else centre_span
    centre = centre_span[0] + (centre_span[1] - centre_span[0]) // 2

    def fmt(t):
        return np.datetime_as_string(t, unit="s", timezone="UTC")

    output_xr.attrs["stac_properties"] = dict(
        start_datetime=fmt(start),
        datetime=nominal_datetime or fmt(centre),
        end_datetime=fmt(end - np.timedelta64(1, "s")),
        created=datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
    )

    return output_xr


def copy_stac_properties(item: Item, ds: Dataset) -> Dataset:
    """Copy properties from an :class:`pystac.Item` to the attrs of an :class:`xarray.Dataset`.

    Copy `item`'s properties to `ds["stac_properties"]`, merging/updating any existing
    values. Sets `ds.attrs["stac_properties"]["start_datetime"]` and
    `ds.attrs["stac_properties"]["end_datetime"]` to `item.properties["datetime"]`.
    Finally, `item.geometry` is copied to the `ds.attrs["stac_geometry"]`.

    Args:
        item: A STAC Item.
        ds: An xarray Dataset.

    Returns:
        The input dataset with additional attributes as described above.

    """
    ds.attrs["stac_properties"] = (
        item.properties.copy()
        if "stac_properties" not in ds.attrs
        else {
            **ds.attrs["stac_properties"],
            **item.properties,
        }
    )
    ds.attrs["stac_properties"]["start_datetime"] = ds.attrs["stac_properties"][
        "datetime"
    ]
    ds.attrs["stac_properties"]["end_datetime"] = ds.attrs["stac_properties"][
        "datetime"
    ]
    ds.attrs["stac_geometry"] = item.geometry
    return ds


def use_alternate_s3_href(modifiable: pystac_client.Modifiable) -> None:
    """Fixes the urls in the official USGS Landsat Stac Server
    (https://landsatlook.usgs.gov/stac-server) so the alternate (s3)
    urls are used. Can be used like::

        client = pystac_client.Client.open(
            "https://landsatlook.usgs.gov/stac-server",
            modifier=use_alternate_s3_href,
        )

    Modifies in place, according to best practices from
    https://pystac-client.readthedocs.io/en/stable/usage.html#automatically-modifying-results.
    """
    if isinstance(modifiable, dict):
        if modifiable["type"] == "FeatureCollection":
            new_features = list()
            for item_dict in modifiable["features"]:
                use_alternate_s3_href(item_dict)
                new_features.append(item_dict)
            modifiable["features"] = new_features
        else:
            stac_object = read_dict(modifiable)
            use_alternate_s3_href(stac_object)
            modifiable.update(stac_object.to_dict())
    else:
        for _, asset in modifiable.assets.items():
            asset_dict = asset.to_dict()
            if "alternate" in asset_dict.keys():
                asset.href = asset.to_dict()["alternate"]["s3"]["href"]
