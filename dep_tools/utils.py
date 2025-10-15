"""This module contains utility functions which don't belong elsewhere."""

import json
from logging import INFO, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import Dict, List, Union

from antimeridian import bbox as antimeridian_bbox
from antimeridian import (
    fix_multi_polygon,
    fix_polygon,
    fix_line_string,
    fix_multi_line_string,
)
from geopandas import GeoDataFrame
import numpy as np
from odc.geo.geobox import GeoBox as GeoBox
from odc.geo.geom import Geometry as OdcGeometry, unary_intersection
from odc.geo.xr import write_cog
import planetary_computer
from pystac import ItemCollection, Item
import pystac_client
from retry import retry
from shapely import Geometry
from shapely.geometry import (
    LineString,
    MultiLineString,
    GeometryCollection,
    MultiPolygon,
)
from shapely.geometry.polygon import orient
from xarray import DataArray, Dataset

from dep_tools.grids import gadm_union


def join_path_or_url(prefix: Path | str, file: str) -> str:
    """Joins a prefix with a file name, with a slash in-between.

    Args:
        prefix: A folder-like thing, local or remote. Can begin
            with things like ./, https:// and s3://. Can end with a
            forward-slash or not.
        file: A stem-plus-extension file name. Can begin with a
            forward-slash or not.

    Returns:
        A string containing the joined prefix and file, with a
        forward-slash in between.
    """
    return (
        str(prefix / file)
        if isinstance(prefix, Path)
        else prefix.rstrip("/") + "/" + file.lstrip("/")
    )


def mask_to_gadm(xarr: DataArray | Dataset, area: GeoBox) -> DataArray | Dataset:
    """Masks an input xarray object to GADM.

    Args:
        xarr: The input xarray object.
        area: An area used to limit the GADM footprint so the operation doesn't
            take as long.

    Returns:
        The input xarray object, masked to GADM.
    """
    geom = unary_intersection(
        [
            area.boundingbox.polygon,
            OdcGeometry(gadm_union().to_crs(area.crs).iloc[0].geometry, crs=area.crs),
        ]
    )
    return xarr.odc.mask(geom)


def get_logger(prefix: str, name: str) -> Logger:
    """Set up a simple logger.

    Args:
        prefix: The prefix to attach to each message.
        name: The name of the logger.

    Returns:
        A :class:Logger:.
        
    """
    console = StreamHandler()
    time_format = "%Y-%m-%d %H:%M:%S"
    console.setFormatter(
        Formatter(
            fmt=f"%(asctime)s %(levelname)s ({prefix}):  %(message)s",
            datefmt=time_format,
        )
    )

    log = getLogger(name)
    log.addHandler(console)
    log.setLevel(INFO)
    return log


def shift_negative_longitudes(
    geometry: Union[LineString, MultiLineString],
) -> Union[LineString, MultiLineString]:
    """Fixes lines spanning the antimeridian by adding 360 to any negative longitudes.

    Args:
        geometry: The geometry to fix.

    Returns:
        The fixed geometry.
    """
    # This is likely a pacific-specific test.
    if abs(geometry.bounds[2] - geometry.bounds[0]) < 180:
        return geometry

    if isinstance(geometry, MultiLineString):
        return MultiLineString(
            [shift_negative_longitudes(geom) for geom in geometry.geoms]
        )

    # If this doesn't work, shift, split, then unshift
    return LineString([(((pt[0] + 360) % 360), pt[1]) for pt in geometry.coords])


BBOX = list[float]


def bbox_across_180(region: GeoDataFrame | GeoBox) -> BBOX | tuple[BBOX, BBOX]:
    """Calculate a bounding box or boxes for an input region.

    If the given region crosses the antimeridian, the bounding box is split
    into two, which straddle but do not cross it.

    Args:
        region: The region of interest.

    Returns:
        A list of floats, or a two-tuple containing lists of floats. 
        
    """
    if isinstance(region, GeoBox):
        geometry = region.geographic_extent.geom
    else:
        geometry = region.to_crs(4326).geometry.make_valid().explode()
        geometry = geometry[
            geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].union_all()

    geometry = _fix_geometry(geometry)
    bbox = antimeridian_bbox(geometry)
    # Sometimes they still come through with the negative value first, see
    # https://github.com/gadomski/antimeridian/issues/134
    if bbox[0] < 0 and bbox[2] > 0:
        bbox[0], bbox[2] = bbox[2], bbox[0]

    # Now fix some coord issues
    # If the lower left X coordinate is greater than 180 it needs to shift
    if bbox[0] > 180:
        bbox[0] = bbox[0] - 360
        # If the upper right X coordinate is greater than 180 it needs to shift
        # but only if the lower left one did too... otherwise we split it below
        if bbox[2] > 180:
            bbox[2] = bbox[2] - 360

    # These are Pacific specific tests, meaning e.g. we know that these aren't
    # areas that are crossing 0 longitude.
    bbox_crosses_antimeridian = (bbox[0] > 0 and bbox[2] < 0) or (
        bbox[0] < 180 and bbox[2] > 180
    )

    if bbox_crosses_antimeridian:
        # Split into two bboxes across the antimeridian
        xmin, ymin = bbox[0], bbox[1]
        xmax, ymax = bbox[2], bbox[3]

        xmax = xmax - 360 if xmax > 180 else xmax
        left_bbox = BBOX([xmin, ymin, 180, ymax])
        right_bbox = BBOX([-180, ymin, xmax, ymax])
        return (left_bbox, right_bbox)
    else:
        return BBOX(bbox)


def fix_winding(geom: Geometry) -> Geometry:
    """Orient the geometry coordinates to run counter-clockwise.

    This is usually non-essential but resolves a barrage of warnings from
    the antimeridian package.

    Args:
        geom: An input Geometry

    Returns:
        The input Geometry, which orientation fixed if needed.
    """
    if geom.geom_type == "Polygon" and not geom.exterior.is_ccw:
        return orient(geom)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([fix_winding(p) for p in geom.geoms])
    else:
        return geom


def _fix_geometry(geometry):
    if isinstance(geometry, GeometryCollection):
        return GeometryCollection(
            [_fix_geometry(a_geometry) for a_geometry in geometry.geoms]
        )
    match geometry.geom_type:
        case "Polygon":
            geometry = fix_polygon(geometry)
        case "MultiPolygon":
            geometry = fix_multi_polygon(geometry)
        case "LineString":
            geometry = fix_line_string(geometry)
        case "MultiLineString":
            geometry = fix_multi_line_string(geometry)
        case _:
            raise ValueError(f"Unsupported geometry type: {geometry.type}")
    return geometry


# retry is for search timeouts which occasionally occur
@retry(tries=5, delay=1)
def search_across_180(
    region: GeoDataFrame | GeoBox, client: pystac_client.Client | None = None, **kwargs
) -> ItemCollection:
    """Conduct a STAC search that handles data crossing the antimeridian correctly.

    Sometimes search results that cross the antimeridian will make the output data
    span the globe when loaded by e.g. :func:`odc.stac.load`. This works by first 
    calling :func:`bbox_across_180`, and then searching within each bounding box
    and combining the results if there is more than one.
    
    Args:
        region: A GeoDataFrame.
        **kwargs: Arguments besides `bbox` and `intersects` passed to
            :func:`pystac_client.Client.search`.

    Returns:
        An ItemCollection.
    """
    if client is None:
        client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

    bbox = bbox_across_180(region)

    if isinstance(bbox, tuple):
        first_result = list(client.search(bbox=bbox[0], **kwargs).items())
        first_ids = [item.id for item in first_result]

        second_result = list(client.search(bbox=bbox[1], **kwargs).items())
        unique_second_result = [
            item for item in second_result if item.id not in first_ids
        ]

        return ItemCollection(first_result + unique_second_result)
    else:
        return client.search(bbox=bbox, **kwargs).item_collection()


def copy_attrs(
    source: DataArray | Dataset, destination: DataArray | Dataset
) -> DataArray | Dataset:
    """Copy attributes from one xarray object to another.

    See https://corteva.github.io/rioxarray/html/getting_started/manage_information_loss.html This function Doesn't account for situations where the inputs don't have the same
    variables.
    Args:
        source: The source object.
        destination: The destination object.

    Returns:
        The destination object, with source attributes, encoding, and nodata value
        set.
        
    """
    if isinstance(destination, DataArray):
        destination.rio.write_crs(source.rio.crs, inplace=True)
        destination.rio.update_attrs(source.attrs, inplace=True)
        destination.rio.update_encoding(source.encoding, inplace=True)
        return destination
    else:
        for variable in destination:
            destination[variable] = copy_attrs(source[variable], destination[variable])
        return destination


def scale_and_offset(
    da: DataArray | Dataset,
    scale: List[float] = [1],
    offset: float = 0,
    keep_attrs: bool=True,
) -> DataArray | Dataset:
    """Applies a scale and offset to data.

    If the process converts e.g. an integer to a floating point, the dtype will
    change.

    Args:
        da: The input data.
        scale: The scale to apply.
        offset: The offset to apply, after scaling.
        keep_attrs: Whether to retain the input attributes in the output.

    Returns:
        The input data, with scale and offset applied.
    """
        
    output = da * scale + offset
    if keep_attrs:
        output = copy_attrs(da, output)
    return output


def write_to_local_storage(
    d: Union[DataArray, Dataset, GeoDataFrame, Item, str],
    path: Union[str, Path],
    write_args: Dict = dict(),
    overwrite: bool = True,
    use_odc_writer: bool = False,
    **kwargs,  # for compatibiilty only
) -> None:
    """Write something to local storage.

    Args:
        d: What to write.
        path: Where to write it.
        write_args: Additional arguments to the writer.
        overwrite: Whether to overwrite existing data.
        use_odc_writer: Whether to use :func:`write_cog` for xarray objects. Othewise
            :func:`rioxarray.rio.to_raster` is used.
        **kwargs: Not used.

    Raises:
        ValueError: If the input data is not of the available types.
    """
    if isinstance(path, str):
        path = Path(path)

    # Create the target folder if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(d, (DataArray, Dataset)):
        if use_odc_writer:
            del write_args["driver"]
            write_cog(d, path, overwrite=overwrite, **write_args)
        else:
            d.rio.to_raster(path, overwrite=overwrite, **write_args)
    elif isinstance(d, GeoDataFrame):
        d.to_file(path, overwrite=overwrite, **write_args)
    elif isinstance(d, Item):
        d = json.dumps(d.to_dict(), indent=4)
        if not Path(path).exists() or overwrite:
            with open(path, "w") as dst:
                dst.write(d)
    elif isinstance(d, str):
        if overwrite:
            with open(path, "w") as dst:
                dst.write(d)
    else:
        raise ValueError(
            "You can only write an Xarray DataArray or Dataset, Geopandas GeoDataFrame, or string"
        )


def scale_to_int16(
    xr: Union[DataArray, Dataset],
    output_multiplier: int,
    output_nodata: int,
    scale_int16s: bool = False,
) -> Union[DataArray, Dataset]:
    """Scale an xarray object to a 16-bit integer.

    Args:
        xr: The input data.
        output_multiplier: The multiplier to apply to the input data.
        output_nodata: The output nodata value. Any null values in the input
            will be set to this. Additional the `"nodata"` attribute will be
            set to this.
        scale_int16s: Whether to scale data which is already an integer.

    Returns:
        This input data, with scaling applied.
    """

    def scale_da(da: DataArray):
        # I exclude int64 here as it seems to cause issues
        int_types = ["int8", "int16", "uint8", "uint16"]
        if da.dtype not in int_types or scale_int16s:
            da = np.multiply(da, output_multiplier)

        return (
            da.where(da.notnull(), output_nodata)
            .astype("int16")
            .rio.write_nodata(output_nodata)  # for rioxarray
            .assign_attrs(nodata=output_nodata)  # for odc
        )

    if isinstance(xr, Dataset):
        for var in xr:
            xr[var] = scale_da(xr[var])
            # Assuming this needs to be redone?
            xr[var].rio.write_nodata(output_nodata, inplace=True).assign_attrs(
                nodata=output_nodata
            )
    else:
        xr = scale_da(xr)

    return xr


def fix_bad_epsgs(item_collection: ItemCollection) -> None:
    """Repair some band EPSG codes in stac items loaded from the MSPC.

    This function modifies in place.
    See https://github.com/microsoft/PlanetaryComputer/discussions/113 for more
    information.

    Args:
        item_collection: The input items.

    Returns:
        The input items, with any bad EPSG codes fixed.
    """
    for item in item_collection:
        if item.collection_id == "landsat-c2-l2":
            if "proj:epsg" in item.properties:
                epsg = str(item.properties["proj:epsg"])
                item.properties["proj:epsg"] = int(f"{epsg[0:3]}{int(epsg[3:]):02d}")
            elif "proj:code" in item.properties:
                epsg = str(item.ext.proj.epsg)
                item.ext.proj.epsg = int(f"{epsg[0:3]}{int(epsg[3:]):02d}")


def remove_bad_items(item_collection: ItemCollection) -> ItemCollection:
    """Remove some error-causing STAC Items from an item collection.

    Remove really bad items which clobber processes even if `fail_on_error` is
    set to False for :func:`odc.stac.load` or the equivalent for 
    :func:`stackstac.stack`. 
    See https://github.com/microsoft/PlanetaryComputer/discussions/101

    Args:
        item_collection: The items.

    Returns:
        The items, with any with known bad IDs removed.
    """
    bad_ids = [
        "LC08_L2SR_081074_20220514_02_T1",
        "LC08_L2SP_101055_20220612_02_T2",
        "LC08_L2SR_074072_20221105_02_T1",
        "LC09_L2SR_074071_20220708_02_T1",
        "LC08_L2SR_078075_20220712_02_T1",
        "LC08_L2SR_080076_20220726_02_T1",
        "LC08_L2SR_082074_20220724_02_T1",
        "LC09_L2SR_083075_20220402_02_T1",
        "LC08_L2SR_083073_20220917_02_T1",
        "LC08_L2SR_089064_20201007_02_T2",
        "LC08_L2SR_074073_20221105_02_T1",
        "LC09_L2SR_073073_20231109_02_T2",
        "LC08_L2SR_075066_20231030_02_T1",
        "S2B_MSIL2A_20230214T001719_R116_T56MMB_20230214T095023",
        "LC09_L2SR_100050_20231107_02_T1",
        "LC09_L2SR_100051_20231107_02_T1",
        "LE07_L2SP_097065_20221029_02_T1",
    ]
    return ItemCollection([i for i in item_collection if i.id not in bad_ids])
