import os
from pathlib import Path
from typing import Iterable, Tuple

from geopandas import read_file, GeoDataFrame
from odc.algo import erase_bad, mask_cleanup
from pystac import ItemCollection
from shapely.geometry import box
from xarray import DataArray, Dataset

from dep_tools.utils import bbox_across_180, fix_winding


def cloud_mask(
    xr: DataArray | Dataset, filters: Iterable[Tuple[str, int]] | None = None
) -> DataArray | Dataset:
    """Get the cloud mask for landsat data.

    Args:
        xr: An xarray object containing Landsat data, including the `qa_pixel` band.
        filters: List of filters to apply to the cloud mask. Each filter is a tuple of
            `(<filter name>, <filter size>)`. Valid filter names are `'opening'` and 
            `'dilation'`. If `None`, no filters will be applied.
            For example: `[("closing", 10),("opening", 2),("dilation", 2)]`.

    Returns:
        The cloud mask defined by the `'qa_pixel'` band, with optional filters applied.

    """
    CLOUD = 3
    CLOUD_SHADOW = 4

    bitmask = 0
    for field in [CLOUD, CLOUD_SHADOW]:
        bitmask |= 1 << field

    try:
        cloud_mask = xr.sel(band="qa_pixel").astype("uint16") & bitmask != 0
    except KeyError:
        cloud_mask = xr.qa_pixel.astype("uint16") & bitmask != 0

    if filters is not None:
        cloud_mask = mask_cleanup(cloud_mask, filters)

    return cloud_mask


def mask_clouds(
    xr: DataArray | Dataset,
    filters: Iterable[Tuple[str, int]] | None = None,
    keep_ints: bool = False,
) -> DataArray | Dataset:
    """
    Mask clouds in Landsat data.

    Args:
        xr: DataArray containing Landsat data including the `qa_pixel` band.
        filters: List of filters to apply to the cloud mask. Each filter is a tuple of
            `(<filter name>, <filter size>)`. Valid filter names are 'opening' and
            'dilation'. If None, no filters will be applied.
            For example: `[("closing", 10),("opening", 2),("dilation", 2)]`.
        keep_ints: If True, return the masked data as integers. Otherwise, return
            the masked data as floats.

    Returns:
        The input, with masks applied.
    """

    mask = cloud_mask(xr, filters)

    if keep_ints:
        return erase_bad(xr, mask)
    else:
        return xr.where(~mask)


def landsat_grid() -> GeoDataFrame:
    """The official Landsat grid filtered to Pacific Island Countries and
    Territories as defined by GADM.

    Returns:
        A GeoDataFrame, indexed by `'PATH'` & `'ROW'`.
    """
    ls_grid_path = Path(__file__).parent / "landsat_grid.gpkg"
    if not ls_grid_path.exists():
        dep_pathrows = read_pathrows_file()
        ls_grid = _pathrows().set_index(["PATH", "ROW"]).loc[dep_pathrows]
        ls_grid.to_file(ls_grid_path)

    return GeoDataFrame(read_file(ls_grid_path).set_index(["PATH", "ROW"]))


def _pathrows() -> GeoDataFrame:
    pathrows = GeoDataFrame(
        read_file(
            "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
        )
    )
    pathrows["geometry"] = pathrows.geometry.apply(fix_winding)
    return pathrows


def read_pathrows_file() -> list[Tuple[int, ...]]:
    """Read list of pathrows containing land within the Digital Earth Pacific aoi
    from a file and return them as a list of tuples.

    Returns:
        A list of tuples of the form `(<path>, <row>)`.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    pathrows_file = os.path.join(cwd, "pathrows.txt")

    with open(pathrows_file, "r") as f:
        lines = f.readlines()
    return [tuple(map(int, line.strip().split("/"))) for line in lines if line.strip()]


def pathrows_in_area(
    area: GeoDataFrame, pathrows: GeoDataFrame | None = None
) -> GeoDataFrame:
    """Get a list of Landsat pathrows in a given area or areas.

    Args:
        area: A :py:class:`geopandas.GeoDataFrame` with polygons.
        pathrows: The pathrows. If unset, the Landsat pathrows are downloaded and
            used.

    Returns:
        A :py:class:`geopandas.GeoDataFrame` of pathrows that intersect the given area.

    """
    if pathrows is None:
        pathrows = _pathrows()

    bbox = bbox_across_180(area)
    if isinstance(bbox, tuple):
        return pathrows[
            pathrows.intersects(box(*bbox[0])) | pathrows.intersects(box(*bbox[1]))
        ]

    return pathrows[pathrows.intersects(box(*bbox))]


def items_in_pathrows(
    items: ItemCollection, some_pathrows: GeoDataFrame
) -> ItemCollection:
    """Filter STAC items to a set of Landsat pathrows.

    This function is likely due for deprecation.

    Args:
        items: A :py:class:`pystac.ItemCollection`. The Landsat pathrow of each
            item is identified by the properties `'landsat:wrs_path'` and
            `'landsat:wrs_row'`.
        some_pathrows:
            The Landsat pathrows of interest. The columns `'PATH'` and `'ROW'`, containing
            integer or string values are used to identify the pathrows.

    Returns:
        The filtered pathrows.

    """
    return ItemCollection(
        some_pathrows.apply(
            lambda row: [
                i
                for i in items
                if i.properties["landsat:wrs_path"] == str(row["PATH"]).zfill(3)
                and i.properties["landsat:wrs_row"] == str(row["ROW"]).zfill(3)
            ],
            axis=1,
        ).sum()
    )


def pathrow_with_greatest_area(shapes: GeoDataFrame) -> Tuple[str, str]:
    """Return the Landsat pathrow with the largest intersection of the given shapes.

    Args:
        shapes: A :py:class:`geopandas.GeoDataFrame` with polygons or
            multipolygons.

    Returns:
        A tuple of the form `(<path>, <row>)`, indicating the Landsat pathrow
        with the greatest overlap.
    """
    pathrows = _pathrows()
    intersection = shapes.to_crs(pathrows.crs).overlay(pathrows, how="intersection")
    row_with_greatest_area = intersection.iloc[[intersection.geometry.area.idxmax()]]
    return (row_with_greatest_area.PATH.item(), row_with_greatest_area.ROW.item())
