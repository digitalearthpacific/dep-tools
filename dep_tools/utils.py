import io
from pathlib import Path
from typing import Dict, List, Union

import fiona
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rioxarray
import xarray as xr
from azure.storage.blob import ContainerClient
from dask.distributed import Client, Lock
from geocube.api.core import make_geocube
from geopandas import GeoDataFrame
from odc.geo.xr import to_cog, write_cog
from osgeo import gdal
from pystac import ItemCollection
from retry import retry
from shapely.geometry import LineString, MultiLineString
from xarray import DataArray, Dataset

from .azure import get_container_client

# Set the timeout to five minutes, which is an extremely long time
TIMEOUT_SECONDS = 60 * 5


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


def bbox_across_180(region: GeoDataFrame) -> BBOX | tuple[BBOX, BBOX]:
    # Previously we just used region.to_crs(4326).total_bounds but if the geom
    # is split right at the antimeridian (see landsat pathrow 073072), output
    # will have zero width (since min and max will be -180 and 180)
    # So get the bounds for all geoms in region, remove the 180s and calculate
    # the min and max values ourselves
    region_ll = GeoDataFrame(region.to_crs(4326))
    x_values = (
        region_ll.explode(index_parts=True).bounds.minx.tolist()
        + region_ll.explode(index_parts=True).bounds.maxx.tolist()
    )
    # Possible we just do a direct compare, we shall see if this causes issues
    # with locations _really close_ to 180 but not touching it
    tolerance = 1e-5
    x_values = [i for i in x_values if abs(180 - abs(i)) > tolerance]

    y_values = (
        region_ll.explode(index_parts=True).bounds.miny.tolist()
        + region_ll.explode(index_parts=True).bounds.maxy.tolist()
    )

    bbox = [min(x_values), min(y_values), max(x_values), max(y_values)]

    # Now fix some coord issues
    # If the lower left X coordinate is greater than 180 it needs to shift
    if bbox[0] > 180:
        bbox[0] = bbox[0] - 360
        # If the upper right X coordinate is greater than 180 it needs to shift
        # but only if the lower left one did too... otherwise we split it below
        if bbox[2] > 180:
            bbox[2] = bbox[2] - 360

    # These are Pacific specific tests!
    bbox_crosses_antimeridian = (bbox[0] < 0 and bbox[2] > 0) or (
        bbox[0] < 180 and bbox[2] > 180
    )
    if bbox_crosses_antimeridian:
        # Split into two bboxes across the antimeridian
        xmax_ll, ymin_ll = bbox[0], bbox[1]
        xmin_ll, ymax_ll = bbox[2], bbox[3]

        xmax_ll = xmax_ll - 360 if xmax_ll > 180 else xmax_ll

        left_bbox = BBOX([xmin_ll, ymin_ll, 180, ymax_ll])
        right_bbox = BBOX([-180, ymin_ll, xmax_ll, ymax_ll])
        return (left_bbox, right_bbox)
    else:
        return BBOX(bbox)


# retry is for search timeouts which occasionally occur
@retry(tries=5, delay=1)
def search_across_180(
    region: GeoDataFrame, client: pystac_client.Client | None = None, **kwargs
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


def write_to_blob_storage(
    d: Union[DataArray, Dataset, GeoDataFrame, str],
    path: Union[str, Path],
    overwrite: bool = True,
    use_odc_writer: bool = False,
    client: ContainerClient | None = None,
    **kwargs,
) -> None:
    # Allowing for a shared container client, which might be
    # more efficient. If not provided, get one.
    if client is None:
        client = get_container_client()
    blob_client = client.get_blob_client(str(path))
    if not overwrite and blob_client.exists():
        return

    if isinstance(d, (DataArray, Dataset)):
        if use_odc_writer:
            if "driver" in kwargs:
                del kwargs["driver"]
            binary_data = to_cog(d, **kwargs)
            blob_client.upload_blob(
                binary_data, overwrite=overwrite, connection_timeout=TIMEOUT_SECONDS
            )
        else:
            with io.BytesIO() as buffer:
                # This is needed or rioxarray doesn't know what type it is
                # writing
                if not "driver" in kwargs:
                    kwargs["driver"] = "COG"
                d.rio.to_raster(buffer, **kwargs)
                buffer.seek(0)
                blob_client.upload_blob(
                    buffer, overwrite=overwrite, connection_timeout=TIMEOUT_SECONDS
                )

    elif isinstance(d, GeoDataFrame):
        with fiona.io.MemoryFile() as buffer:
            d.to_file(buffer, **kwargs)
            buffer.seek(0)
            blob_client.upload_blob(buffer, overwrite=overwrite)
    elif isinstance(d, str):
        blob_client.upload_blob(d, overwrite=overwrite, **kwargs)
    else:
        raise ValueError(
            "You can only write an Xarray DataArray or Dataset, or Geopandas GeoDataFrame"
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
    """Repairs soLC08_L2SP_101055_20220612_20220617_02_T2me band epsg codes in stac items loaded from the Planetary
    Computer stac catalog"""
    # ** modifies in place **
    # See https://github.com/microsoft/PlanetaryComputer/discussions/113
    # Will get fixed at some point and we can remove this
    for item in item_collection:
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
