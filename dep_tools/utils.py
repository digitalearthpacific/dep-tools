from logging import INFO, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rioxarray
from antimeridian import bbox as antimeridian_bbox
from antimeridian import fix_multi_polygon, fix_polygon
from dask.distributed import Client, Lock
from geopandas import GeoDataFrame
from odc.geo.geobox import GeoBox
from odc.geo.xr import to_cog, write_cog
from osgeo import gdal
from pystac import ItemCollection
from retry import retry
from shapely.geometry import LineString, MultiLineString
from xarray import DataArray, Dataset

from .azure import get_container_client

# Set the timeout to five minutes, which is an extremely long time
TIMEOUT_SECONDS = 60 * 5


def get_logger(prefix: str, name: str) -> Logger:
    """Set up a simple logger"""
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
    geometry: Union[LineString, MultiLineString]
) -> Union[LineString, MultiLineString]:
    """
    Fixes lines that span the antimeridian by adding 360 to any negative
    longitudes.
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
    if isinstance(region, GeoBox):
        geometry = region.geographic_extent.geom
    else:
        geometry = region.to_crs(4326).unary_union

    if geometry.geom_type == "Polygon":
        geometry = fix_polygon(geometry)
    elif geometry.geom_type == "MultiPolygon":
        geometry = fix_multi_polygon(geometry)
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.type}")

    bbox = antimeridian_bbox(geometry)

    # Now fix some coord issues
    # If the lower left X coordinate is greater than 180 it needs to shift
    if bbox[0] > 180:
        bbox[0] = bbox[0] - 360
        # If the upper right X coordinate is greater than 180 it needs to shift
        # but only if the lower left one did too... otherwise we split it below
        if bbox[2] > 180:
            bbox[2] = bbox[2] - 360

    # These are Pacific specific tests!
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


# retry is for search timeouts which occasionally occur
@retry(tries=5, delay=1)
def search_across_180(
    region: GeoDataFrame | GeoBox, client: pystac_client.Client | None = None, **kwargs
) -> ItemCollection:
    """
    region: A GeoDataFrame.
    **kwargs: Arguments besides bbox and intersects passed to
        pystac_client.Client.search
    """
    if client is None:
        client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

    bbox = bbox_across_180(region)

    if isinstance(bbox, tuple):
        return ItemCollection(
            list(client.search(bbox=bbox[0], **kwargs).items())
            + list(client.search(bbox=bbox[1], **kwargs).items())
        )
    else:
        return client.search(bbox=bbox, **kwargs).item_collection()


def copy_attrs(
    source: DataArray | Dataset, destination: DataArray | Dataset
) -> DataArray | Dataset:
    # See https://corteva.github.io/rioxarray/html/getting_started/manage_information_loss.html
    # Doesn't account if source and dest don't have the same vars
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
    keep_attrs=True,
) -> DataArray | Dataset:
    """Apply the given scale and offset to the given Xarray object."""
    output = da * scale + offset
    if keep_attrs:
        output = copy_attrs(da, output)
    return output


def write_to_local_storage(
    d: Union[DataArray, Dataset, GeoDataFrame, str],
    path: Union[str, Path],
    write_args: Dict = dict(),
    overwrite: bool = True,
    use_odc_writer: bool = False,
    **kwargs,  # for compatibiilty only
) -> None:
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
    """Multiply the given DataArray by the given multiplier and convert to
    int16 data type, with the given nodata value"""

    def scale_da(da: DataArray):
        # I exclude int64 here as it seems to cause issues
        int_types = ["int8", "int16", "uint8", "uint16"]
        if da.dtype not in int_types or scale_int16s:
            da = np.multiply(da, output_multiplier)

        return (
            da.where(da.notnull(), output_nodata)
            .astype("int16")
            .rio.write_nodata(output_nodata)
        )

    if isinstance(xr, Dataset):
        for var in xr:
            xr[var] = scale_da(xr[var])
            xr[var].rio.write_nodata(output_nodata, inplace=True)
    else:
        xr = scale_da(xr)

    return xr


def raster_bounds(raster_path: Path) -> List:
    """Returns the bounds for a raster file at the given path"""
    with rasterio.open(raster_path) as t:
        return list(t.bounds)


def gpdf_bounds(gpdf: GeoDataFrame) -> List[float]:
    """Returns the bounds for the give GeoDataFrame, and makes sure
    it doesn't cross the antimeridian."""
    bbox = gpdf.to_crs("EPSG:4326").total_bounds
    # Or the opposite!
    bbox_crosses_antimeridian = bbox[0] < 0 and bbox[2] > 0
    if bbox_crosses_antimeridian:
        # This may be overkill, but nothing else was really working
        bbox[0] = -179.9999999999
        bbox[2] = 179.9999999999
    return bbox


def build_vrt(
    bounds: List,
    prefix: str = "",
    suffix: str = "",
) -> Path:
    blobs = [
        f"/vsiaz/output/{blob.name}"
        for blob in get_container_client().list_blobs(name_starts_with=prefix)
        if blob.name.endswith(suffix)
    ]

    local_prefix = Path(prefix).stem
    vrt_file = f"data/{local_prefix}.vrt"
    gdal.BuildVRT(vrt_file, blobs, outputBounds=bounds)
    return Path(vrt_file)


def _local_prefix(prefix: str) -> str:
    return Path(prefix).stem


def _mosaic_file(prefix: str) -> str:
    return f"data/{_local_prefix(prefix)}.tif"


def mosaic_scenes(
    prefix: str,
    bounds: List,
    client: Client,
    scale_factor: float = None,
    overwrite: bool = True,
) -> None:
    mosaic_file = _mosaic_file(prefix)
    if not Path(mosaic_file).is_file() or overwrite:
        vrt_file = build_vrt(prefix, bounds)
        rioxarray.open_rasterio(vrt_file, chunks=True).rio.to_raster(
            mosaic_file,
            compress="LZW",
            lock=Lock("rio", client=client),
        )

        if scale_factor is not None:
            with rasterio.open(mosaic_file, "r+") as dst:
                dst.scales = (scale_factor,)


def fix_bad_epsgs(item_collection: ItemCollection) -> None:
    """Repairs some band epsg codes in stac items loaded from the Planetary
    Computer stac catalog"""
    # ** modifies in place **
    # See https://github.com/microsoft/PlanetaryComputer/discussions/113
    # Will get fixed at some point and we can remove this
    for item in item_collection:
        if item.collection_id == "landsat-c2-l2":
            epsg = str(item.properties["proj:epsg"])
            item.properties["proj:epsg"] = int(f"{epsg[0:3]}{int(epsg[3:]):02d}")


def remove_bad_items(item_collection: ItemCollection) -> ItemCollection:
    """Remove really bad items which clobber processes even if `fail_on_error` is
    set to False for odc.stac.load or the equivalent for stackstac.stack. The
    first one here is a real file that is just an error in html.
    See https://github.com/microsoft/PlanetaryComputer/discussions/101
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
