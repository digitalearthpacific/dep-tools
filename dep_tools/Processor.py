from dataclasses import dataclass, field

# from importlib.resources import files
import os
from pathlib import Path
from typing import Dict, List, Union, Callable

from azure.storage.blob import ContainerClient
from dask.distributed import Client, Lock
from dask_gateway import GatewayCluster
from geopandas import GeoDataFrame
from osgeo import gdal
from osgeo_utils import gdal2tiles
from pystac import ItemCollection
import rasterio
from rasterio import RasterioIOError
import rioxarray
from stackstac import stack
from tqdm import tqdm
from xarray import DataArray

from .landsat_utils import item_collection_for_pathrow, mask_clouds
from .utils import (
    fix_bad_epsgs,
    gpdf_bounds,
    scale_to_int16,
    write_to_blob_storage,
    scale_and_offset,
)


@dataclass
class Processor:
    """
    Args:
        year(int): The year for which we wish to run computations.
        scene_processor(Callable): A function which accepts at least a parameter
            of type xarray.DataArray and returns the same. Additional arguments
            can be specified using `scene_processor_kwargs`.
        aoi_by_tile(GeoDataFrame): A GeoDataFrame holdinng areas
            of interest (typically land masses), split by landsat tile (and in
            future, Sentinel 2). Each tile must be indexed by the tile id columns
            (e.g. "PATH" and "ROW").
        dataset_id(str): Our name for the dataset, such as "ndvi"
        storage_account(str): The name of the azure storage account we wish to
            use to write outputs. Defaults to the environmental variable
            "AZURE_STORAGE_ACCOUNT"
        container_name(str): The name of the container in the storage account
            to which we wish to write data. Defaults to "output".
        credential(str): The credentials for the storage account we wish to use.
            For valid options, see the help for azure.storage.blob.ContainerClient.
            Defaults to the environmental variable "AZURE_STORAGE_SAS_TOKEN".
    """

    scene_processor: Callable
    dataset_id: str
    year: Union[str, None] = None
    aoi_by_tile: GeoDataFrame = GeoDataFrame.from_file(
        Path(__file__).parent / "aoi_split_by_landsat_pathrow.gpkg"
    ).set_index(["PATH", "ROW"])
    scene_processor_kwargs: Dict = field(default_factory=dict)
    send_area_to_scene_processor: bool = False
    dask_chunksize: int = 4096
    storage_account: str = os.environ["AZURE_STORAGE_ACCOUNT"]
    container_name: str = "output"
    credential: str = os.environ["AZURE_STORAGE_SAS_TOKEN"]
    convert_output_to_int16: bool = True
    output_value_multiplier: int = 10000
    output_nodata: int = -32767
    color_ramp_file: Union[str, None] = None

    def __post_init__(self):
        self.container_client = ContainerClient(
            f"https://{self.storage_account}.blob.core.windows.net",
            container_name=self.container_name,
            credential=self.credential,
        )
        self.prefix = (
            f"{self.dataset_id}/{self.year}/{self.dataset_id}_{self.year}"
            if self.year
            else f"{self.dataset_id}/{self.dataset_id}"
        )
        self.local_prefix = Path(self.prefix).stem
        self.mosaic_file = f"data/{self.local_prefix}.tif"

    def get_stack(
        self, item_collection: ItemCollection, these_areas: GeoDataFrame
    ) -> DataArray:
        return (
            stack(
                item_collection,
                epsg=8859,
                chunksize=self.dask_chunksize,
                resolution=30,
                # Previously it only caught 404s, we are getting other errors
                errors_as_nodata=(RasterioIOError(".*"),),
            )
            .rio.write_crs("EPSG:8859")
            .rio.clip(
                these_areas.to_crs("EPSG:8859").geometry,
                all_touched=True,
                from_disk=True,
            )
        )

    def process_by_scene(self) -> None:
        for index, _ in tqdm(
            self.aoi_by_tile.iterrows(), total=self.aoi_by_tile.shape[0]
        ):
            these_areas = self.aoi_by_tile.loc[[index]]
            index_dict = dict(zip(self.aoi_by_tile.index.names, index))
            item_collection = item_collection_for_pathrow(
                # For S2, would probably just pass index_dict as kwargs
                # to generic function
                path=index_dict["PATH"],
                row=index_dict["ROW"],
                search_args=dict(
                    collections=["landsat-c2-l2"],
                    datetime=self.year,
                    bbox=gpdf_bounds(these_areas),
                ),
            )

            if len(item_collection) == 0:
                continue
            fix_bad_epsgs(item_collection)

            item_xr = self.get_stack(item_collection, these_areas)
            item_xr = mask_clouds(item_xr)
            scale = 0.0000275
            offset = -0.2
            item_xr = scale_and_offset(item_xr, scale=[scale], offset=offset)

            if self.send_area_to_scene_processor:
                self.scene_processor_kwargs.update(dict(area=these_areas))
            results = self.scene_processor(item_xr, **self.scene_processor_kwargs)
            if self.convert_output_to_int16:
                results = scale_to_int16(
                    results, self.output_value_multiplier, self.output_nodata
                )

            if len(results.dims.keys()) > 2:
                results = (
                    results.to_array(dim="variables")
                    #                    .stack(z=["variables", "time"])
                    .stack(z=["time", "variables"])
                    .to_dataset(dim="z")
                    .pipe(
                        lambda ds: ds.rename_vars(
                            {name: "_".join(name) for name in ds.data_vars}
                        )
                    )
                    .drop_vars(["variables", "time"])
                )
            #                    for year in results.coords["time"]:
            #                        these_results = results.sel(time=year)
            #                        name = f"{self.dataset_id}/{year.values.tolist()}_{'_'.join([str(i) for i in index])}.tif"
            #                        write_to_blob_storage(
            #                            these_results, name, dict(driver="COG", compress="LZW")
            #                        )
            #                for var in results:
            #                    these_results = results[var].to_dataset("time")
            #                    name = f"{self.dataset_id}/{var}_{'_'.join([str(i) for i in index])}.tif"
            #                    write_to_blob_storage(
            #                        these_results, name, dict(driver="COG", compress="LZW")
            #                    )

            #            else:
            write_to_blob_storage(
                results,
                f"{self.prefix}_{'_'.join([str(i) for i in index])}.tif",
                dict(driver="COG", compress="LZW"),
            )

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
