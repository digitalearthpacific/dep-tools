import io
from pathlib import Path
from typing import Dict, List, Union

import fiona
import numpy as np
import planetary_computer
import pyproj
import pystac_client
import rasterio
import rioxarray
import xarray as xr
from dask.distributed import Client, Lock
from geocube.api.core import make_geocube
from geopandas import GeoDataFrame
from osgeo import gdal
from pystac import ItemCollection
from retry import retry
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import transform
from xarray import DataArray, Dataset

from .azure import get_container_client


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


@retry(tries=10, delay=1)
def search_across_180(gpdf: GeoDataFrame, **kwargs) -> ItemCollection:
    """
    gpdf: A GeoDataFrame.
    **kwargs: Arguments besides bbox and intersects passed to
        pystac_client.Client.search
    """

    # pystac_client doesn't appear to be able to handle non-geographic data,
    # either via the `bbox` or `intersects` parameter. The docs don't really say.
    # Here I split the bbox of the given GeoDataFrame on either side of the
    # 180th meridian and concatenate the results.
    # An alternative would be to actually cut the gpdf in two pieces and use
    # intersects, but we can wait to see if that's needed (for the current
    # work I am collecting io-lulc which doesn't have data in areas which
    # aren't near land
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    bbox_4326 = gpdf.to_crs(4326).total_bounds
    bbox_crosses_antimeridian = bbox_4326[0] < 0 and bbox_4326[2] > 0
    if bbox_crosses_antimeridian:
        gpdf_proj = gpdf.to_crs(gpdf.crs)
        projector = pyproj.Transformer.from_crs(
            gpdf_proj.crs, pyproj.CRS("EPSG:4326"), always_xy=True
        ).transform

        xmin, ymin, xmax, ymax = gpdf_proj.total_bounds
        xmin_ll, ymin_ll = transform(projector, Point(xmin, ymin)).coords[0]
        xmax_ll, ymax_ll = transform(projector, Point(xmax, ymax)).coords[0]

        left_bbox = [xmin_ll, ymin_ll, 180, ymax_ll]
        right_bbox = [-180, ymin_ll, xmax_ll, ymax_ll]
        return ItemCollection(
            list(catalog.search(bbox=left_bbox, **kwargs).items())
            + list(catalog.search(bbox=right_bbox, **kwargs).items())
        )

    return catalog.search(bbox=bbox_4326, **kwargs).item_collection()


def scale_and_offset(
    da: xr.DataArray, scale: List[float] = [1], offset: float = 0
) -> xr.DataArray:
    """Apply the given scale and offset to the given DataArray"""
    return da * scale + offset


def make_geocube_dask(
    df: GeoDataFrame, measurements: List[str], like: xr.DataArray, **kwargs
):
    """Dask-enabled geocube.make_geocube. Not completely implemented."""

    def rasterize_block(block):
        return (
            make_geocube(df, measurements=measurements, like=block, **kwargs)
            .to_array(measurements[0])
            .assign_coords(block.coords)
        )

    return like.map_blocks(rasterize_block, template=like)


def write_to_local_storage(
    d: Union[DataArray, Dataset, GeoDataFrame, str],
    path: Union[str, Path],
    write_args: Dict = dict(),
    overwrite: bool = True,
    **kwargs,  # for compatibiilty only
) -> None:
    if isinstance(path, str):
        path = Path(path)

    # Create the target folder if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(d, (DataArray, Dataset)):
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


@retry(tries=2, delay=2)
def write_to_blob_storage(
    d: Union[DataArray, Dataset, GeoDataFrame, str],
    path: Union[str, Path],
    write_args: Dict = dict(),
    overwrite: bool = True,
    **kwargs,
) -> None:
    container_client = get_container_client(**kwargs)

    blob_client = container_client.get_blob_client(str(path))
    if not overwrite and blob_client.exists():
        return

    if isinstance(d, (DataArray, Dataset)):
        with io.BytesIO() as buffer:
            d.rio.to_raster(buffer, **write_args)
            buffer.seek(0)
            blob_client.upload_blob(buffer, overwrite=overwrite)
    elif isinstance(d, GeoDataFrame):
        with fiona.io.MemoryFile() as buffer:
            d.to_file(buffer, **write_args)
            buffer.seek(0)
            blob_client.upload_blob(buffer, overwrite=overwrite)
    elif isinstance(d, str):
        blob_client.upload_blob(d, overwrite=overwrite, **write_args)
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
    print(blobs)
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
        "S2B_MSIL2A_20230214T001719_R116_T56MMB_20230214T095023",
    ]
    return ItemCollection([i for i in item_collection if i.id not in bad_ids])
