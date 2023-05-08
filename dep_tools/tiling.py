# These pieces were removed from Processor.py. See old
# dep-data code for clearer implementation

import os
from pathlib import Path
from typing import List

from dask.distributed import Client, Lock
from osgeo import gdal
from osgeo_utils import gdal2tiles
import rasterio
import rioxarray
from tqdm import tqdm


def copy_to_blob_storage(self, local_path: Path, remote_path: Path) -> None:
    with open(local_path, "rb") as src:
        blob_client = self.container_client.get_blob_client(str(remote_path))
        blob_client.upload_blob(src, overwrite=True)


def build_vrt(self, bounds: List[float]) -> Path:
    blobs = [
        f"/vsiaz/{self.container_name}/{blob.name}"
        for blob in self.container_client.list_blobs()
        if blob.name.startswith(self.prefix)
    ]

    vrt_file = f"data/{self.local_prefix}.vrt"
    gdal.BuildVRT(vrt_file, blobs, outputBounds=bounds)
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


if mosaic:
    processor.mosaic_scenes()

if tile:
    processor.create_tiles(remake_mosaic_for_tiles)
