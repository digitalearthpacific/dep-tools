import io
from itertools import chain
import os
from pathlib import Path
from typing import Dict, Iterable, List, Union

import azure.storage.blob
from azure.storage.blob import ContainerClient
from dask.distributed import Client, Lock
import fiona
from geopandas import GeoDataFrame
from geocube.api.core import make_geocube
import numpy as np
from osgeo import gdal
import osgeo_utils.gdal2tiles
import planetary_computer
import pyproj
from pystac import ItemCollection
import pystac_client
import rasterio
from retry import retry
import rioxarray
from rio_stac import create_stac_item
from shapely import buffer, difference
from shapely.geometry import MultiLineString, LineString, Point
from shapely.ops import transform
from tqdm import tqdm
import xarray as xr
from xarray import DataArray, Dataset


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
        gpdf_8859 = gpdf.to_crs(8859)
        projector = pyproj.Transformer.from_crs(
            gpdf_8859.crs, pyproj.CRS("EPSG:4326"), always_xy=True
        ).transform

        xmin, ymin, xmax, ymax = gpdf_8859.total_bounds
        xmin_ll, ymin_ll = transform(projector, Point(xmin, ymin)).coords[0]
        xmax_ll, ymax_ll = transform(projector, Point(xmax, ymax)).coords[0]

        left_bbox = [xmin_ll, ymin_ll, 180, ymax_ll]
        right_bbox = [-180, ymin_ll, xmax_ll, ymax_ll]
        return ItemCollection(
            list(catalog.search(bbox=left_bbox, **kwargs).items())
            + list(catalog.search(bbox=right_bbox, **kwargs).items())
        )
    else:
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


def get_container_client(
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    container_name: str = "output",
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
) -> ContainerClient:
    return azure.storage.blob.ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )


def blob_exists(path: Union[str, Path], **kwargs):
    container_client = get_container_client(**kwargs)
    blob_client = container_client.get_blob_client(str(path))
    return blob_client.exists()


def get_blob_path(
    dataset_id, item_id, prefix=None, time=None, variable=None, ext: str = "tif"
) -> str:
    if variable is None:
        variable = dataset_id

    prefix = f"{prefix}/" if prefix is not None else ""
    time = str(time).replace("/", "_") if time is not None else time
    suffix = "_".join([str(i) for i in item_id])
    return (
        f"{prefix}{dataset_id}/{time}/{variable}_{time}_{suffix}.{ext}"
        if time is not None
        else f"{prefix}{dataset_id}/{variable}_{suffix}.{ext}"
    )


def download_blob(
    container_client: ContainerClient,
    dataset: str,
    year: int,
    path: str,
    row: str,
    local_dir: Path,
) -> None:
    remote_path = f"{dataset}/{year}/{dataset}_{year}_{path}_{row}.tif"
    local_path = f"{local_dir}/{dataset}_{year}_{path}_{row}.tif"
    blob_client = container_client.get_blob_client(remote_path)
    if blob_client.exists() and not Path(local_path).exists():
        with open(local_path, "wb") as dst:
            download_stream = blob_client.download_blob()
            dst.write(download_stream.readall())


@retry(tries=20, delay=2)
def write_to_blob_storage(
    d: Union[DataArray, Dataset, GeoDataFrame],
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
    else:
        raise ValueError(
            "You can only write an Xarray DataArray or Dataset, or Geopandas GeoDataFrame"
        )


def copy_to_blob_storage(
    local_path: Path,
    remote_path: Path,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
) -> None:
    container_client = azure.storage.blob.ContainerClient(
        f"https://{storage_account}.blob.core.windows.net",
        container_name=container_name,
        credential=credential,
    )

    with open(local_path, "rb") as src:
        blob_client = container_client.get_blob_client(str(remote_path))
        blob_client.upload_blob(src, overwrite=True)


def scale_to_int16(
    xr: Union[DataArray, Dataset],
    output_multiplier: int,
    output_nodata: int,
    scale_int16s: bool = False,
) -> Union[DataArray, Dataset]:
    """Multiply the given DataArray by the given multiplier and convert to
    int16 data type, with the given nodata value"""

    def scale_da(da: DataArray):
        if da.dtype != "int16" or scale_int16s:
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
    prefix: str,
    bounds: List,
) -> Path:
    blobs = [
        f"/vsiaz/output/{blob.name}"
        for blob in get_container_client().list_blobs()
        if blob.name.startswith(prefix)
    ]

    local_prefix = Path(prefix).stem
    vrt_file = f"data/{local_prefix}.vrt"
    gdal.BuildVRT(vrt_file, blobs, outputBounds=bounds)
    return Path(vrt_file)


def create_tiles(
    color_file: str,
    prefix: str,
    bounds: List,
    remake_mosaic: bool = True,
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"],
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"],
    container_name: str = "output",
):
    if remake_mosaic:
        with Client() as local_client:
            mosaic_scenes(
                prefix=prefix,
                bounds=bounds,
                client=local_client,
                scale_factor=1.0 / 1000,
                overwrite=remake_mosaic,
            )
    dst_vrt_file = f"data/{Path(prefix).stem}_rgb.vrt"
    gdal.DEMProcessing(
        dst_vrt_file,
        str(_mosaic_file(prefix)),
        "color-relief",
        colorFilename=color_file,
        addAlpha=True,
    )
    dst_name = f"data/tiles/{prefix}"
    os.makedirs(dst_name, exist_ok=True)
    max_zoom = 11
    # First arg is just a dummy so the second arg is not removed (see gdal2tiles code)
    # I'm using 512 x 512 tiles so there's fewer files to copy over. likewise
    # for -x
    osgeo_utils.gdal2tiles.main(
        [
            "gdal2tiles.py",
            "--tilesize=512",
            "--processes=4",
            f"--zoom=0-{max_zoom}",
            "-x",
            dst_vrt_file,
            dst_name,
        ]
    )

    for local_path in tqdm(Path(dst_name).rglob("*")):
        if local_path.is_file():
            remote_path = Path("tiles") / "/".join(local_path.parts[4:])
            copy_to_blob_storage(
                local_path, remote_path, storage_account, credential, container_name
            )
            local_path.unlink()


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
    bad_ids = ["LC08_L2SP_101055_20220612_02_T2"]
    return ItemCollection([i for i in item_collection if i.id not in bad_ids])
