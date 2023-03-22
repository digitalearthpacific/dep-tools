from dataclasses import dataclass
import io
import os
from pathlib import Path
from typing import Any, Dict, Union, Callable

from azure.storage.blob import ContainerClient
from dask.distributed import Client, Lock
from dask_gateway import GatewayCluster
from geopandas import read_file, GeoDataFrame
from osgeo import gdal
from osgeo_utils import gdal2tiles
from pystac import ItemCollection
import rasterio
import rioxarray
from stackstac import stack
from time import time
from tqdm import tqdm
from xarray import DataArray

from constants import STORAGE_AOI_PREFIX
from landsat_utils import item_collection_for_pathrow, mask_clouds
from utils import (
    gpdf_bounds,
    raster_bounds,
    scale_to_int16,
    write_to_blob_storage,
    scale_and_offset,
)


@dataclass
class Processor:
    year: int
    scene_processor: Callable
    dataset_id: str
    # Add collection variable - landsat or s2
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"]
    container_name: str = "output"
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"]
    aoi_file: Path = STORAGE_AOI_PREFIX / "aoi.tif"
    # Change these to "tile_file" and "aoi_by_tile_file"
    pathrow_file: Path = STORAGE_AOI_PREFIX / "pathrows_in_aoi.gpkg"
    aoi_by_pathrow_file: Path = STORAGE_AOI_PREFIX / "aoi_split_by_landsat_pathrow.gpkg"
    color_ramp_file: Union[str, None] = None
    output_value_multiplier: int = 10000
    output_nodata: int = -32767
    dask_chunksize: int = 4096
    scene_processor_kwargs: dict = dict()

    def __post_init__(self):
        self.container_client = ContainerClient(
            f"https://{self.storage_account}.blob.core.windows.net",
            container_name=self.container_name,
            credential=self.credential,
        )
        self.prefix = f"{self.dataset_id}/{self.year}/{self.dataset_id}_{self.year}"
        self.bounds = raster_bounds(self.aoi_file)
        self.local_prefix = Path(self.prefix).stem
        self.mosaic_file = f"data/{self.local_prefix}.tif"
        self.pathrows = read_file(self.pathrow_file)
        self.aoi_by_pathrow = read_file(self.aoi_by_pathrow_file)

    # str or int?
    def get_areas(self, path: str, row: str) -> GeoDataFrame:
        return self.aoi_by_pathrow[
            (self.aoi_by_pathrow["PATH"] == path) & (self.aoi_by_pathrow["ROW"] == row)
        ]

    def get_stack(
        self, item_collection: ItemCollection, these_areas: GeoDataFrame
    ) -> DataArray:
        return (
            stack(
                item_collection,
                epsg=8859,
                chunksize=self.dask_chunksize,
                resolution=30,
            )
            .rio.write_crs("EPSG:8859")
            .rio.clip(these_areas.to_crs("EPSG:8859").geometry, all_touched=True)
        )

    def process_by_scene(self) -> None:
        for i, row in self.pathrows.iterrows():
            last_time = time()
            # Change this to get_areas(row), or get_collection(row)
            path = row["PATH"]
            row = row["ROW"]
            these_areas = self.get_areas(path, row)

            item_collection = item_collection_for_pathrow(
                path,
                row,
                # obv change collection here
                dict(
                    collections=["landsat-c2-l2"],
                    datetime=str(self.year),
                    bbox=gpdf_bounds(these_areas),
                ),
            )

            if len(item_collection) == 0:
                print(f"{path:03d}-{row:03d} | ** NO ITEMS **")
                continue

            item_xr = self.get_stack(item_collection, these_areas)
            item_xr = mask_clouds(item_xr)
            scale = 0.0000275
            offset = -0.2
            item_xr = scale_and_offset(item_xr, scale=[scale], offset=offset)

            results = self.scene_processor(item_xr, **self.scene_processor_kwargs)
            results = scale_to_int16(
                results, self.output_value_multiplier, self.output_nodata
            )

            try:
                write_to_blob_storage(
                    results,
                    # Replace {path}_{row} with {self.get_scene_id()}
                    f"{self.prefix}_{path}_{row}.tif",
                    dict(driver="COG", compress="LZW", predictor=2),
                )
            except Exception as e:
                print(e)
            print(
                f"{path:03d}-{row:03d} | {(i+1):03d}/{len(self.pathrows.index)} | {round(time() - last_time)}s"
            )

    def copy_to_blob_storage(self, local_path: Path, remote_path: Path) -> None:
        with open(local_path, "rb") as src:
            blob_client = self.container_client.get_blob_client(str(remote_path))
            blob_client.upload_blob(src, overwrite=True)

    def build_vrt(self) -> Path:
        blobs = [
            f"/vsiaz/{self.container_name}/{blob.name}"
            for blob in self.container_client.list_blobs()
            if blob.name.startswith(self.prefix)
        ]

        vrt_file = f"data/{self.local_prefix}.vrt"
        gdal.BuildVRT(vrt_file, blobs, outputBounds=self.bounds)
        return Path(vrt_file)

    def mosaic_scenes(self, scale_factor: float = None, overwrite: bool = True) -> None:
        if not Path(self.mosaic_file).is_file() or overwrite:
            vrt_file = self.build_vrt()
            with Client() as local_client:
                rioxarray.open_rasterio(vrt_file, chunks=True).rio.to_raster(
                    self.mosaic_file,
                    compress="LZW",
                    predictor=2,
                    lock=Lock("rio", client=local_client),
                )

            if scale_factor is not None:
                with rasterio.open(self.mosaic_file, "r+") as dst:
                    dst.scales = (scale_factor,)

    def create_tiles(self, remake_mosaic: bool = True) -> None:
        if remake_mosaic:
            self.mosaic_scenes(scale_factor=1.0 / 1000, overwrite=True)
        dst_vrt_file = f"data/{self.local_prefix}_rgb.vrt"
        gdal.DEMProcessing(
            dst_vrt_file,
            str(self.mosaic_file),
            "color-relief",
            colorFilename=self.color_ramp_file,
            addAlpha=True,
        )
        dst_name = f"data/tiles/{self.prefix}"
        os.makedirs(dst_name, exist_ok=True)
        max_zoom = 11
        # First arg is just a dummy so the second arg is not removed (see gdal2tiles code)
        # I'm using 512 x 512 tiles so there's fewer files to copy over. likewise
        # for -x
        gdal2tiles.main(
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
                self.copy_to_blob_storage(local_path, remote_path)
                local_path.unlink()


def run_processor(
    year: int,
    scene_processor: Callable,
    color_ramp_file: str = None,
    run_scenes: bool = False,
    mosaic: bool = False,
    tile: bool = False,
    remake_mosaic_for_tiles: bool = True,
    **kwargs,
):
    processor = Processor(
        year, scene_processor, color_ramp_file=color_ramp_file, **kwargs
    )
    if run_scenes:
        cluster = GatewayCluster(worker_cores=1, worker_memory=8)
        cluster.scale(400)
        with cluster.get_client() as client:
            print(client.dashboard_link)
            processor.process_by_scene()

    if mosaic:
        processor.mosaic_scenes()

    if tile:
        processor.create_tiles(remake_mosaic_for_tiles)
