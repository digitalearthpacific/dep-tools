from copy import deepcopy
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Dict, List, Union, Callable

from azure.storage.blob import ContainerClient
from dask.distributed import Client
from dask_gateway import GatewayCluster
from geopandas import GeoDataFrame
from odc.stac import load
from pandas import DataFrame
from pystac import ItemCollection
from rasterio import RasterioIOError
from rasterio.enums import Resampling
import rioxarray
from stackstac import stack
from tqdm import tqdm
from xarray import DataArray

from .landsat_utils import mask_clouds
from .utils import (
    fix_bad_epsgs,
    get_container_client,
    scale_and_offset,
    search_across_180,
    scale_to_int16,
    write_to_blob_storage,
)


@dataclass
class Processor:
    """
    Calls `scene_processor` for each row of `aoi_by_tile` and writes the result
    to azure blob storage. This is the core method to scale processing across
    the study area, particularly when processing on the planetary computer using
    kbatch and (optionally, but recommended) the dask gateway.

    Args:
        scene_processor(Callable): A function which accepts at least a parameter
            of type xarray.DataArray or xarray.Dataset and returns the same.
            Additional arguments can be specified using `scene_processor_kwargs`.
        scene_processor_kwargs(Dict): Additional arguments to `scene_processor`.
        send_area_to_scene_processor(bool): Should the features contained in
            `aoi_by_tile` for this scene be sent to `scene_processor` as the
            second argument? Useful if masking or some other custom processing.
            Defaults to false.
        dataset_id(str): Our name for the dataset, such as "ndvi"
        year(int): The year for which we wish to run computations. If None,
            all years are processed. Default is None.
        split_output_by_year(bool): If year is None and there are multiple years
            to process, should a file be made for each year, or should they be
            combined? Useful if computations across all years results in out of
            memory errors. Default is False.
        split_output_by_variable(bool): If scene_processor returns a dataset with
            multiple variables, should outputs be written for each variable?
            Not as useful for memory errors as `split_output_by_year`. Default
            is False.
        overwrite(bool). If True, then overwrite any existing output. Defaults
            to False
        aoi_by_tile(GeoDataFrame): A GeoDataFrame holdinng areas
            of interest (typically land masses), split by landsat tile (and in
            future, Sentinel 2). Each tile must be indexed by the tile id columns
            (e.g. "PATH" and "ROW").
        container_client(azure.storage.blob.ContainerClient): The container
            client for the container to which we wish to write the output.
        scale_and_offset(bool). Should raw landsat data be scaled and offset or
            kept in raw (integer) values. Defaults to True.
        dask_chunksize(int). The (single dimensional) chunk size set when
        convert_output_to_int16(bool). Should output be cast to int16 before
            writing to blob storage? Can save space on disk as well as solve
            memory issues since output files are written to memory before
            being copied to blob storage (as I can't find another way to do
            it; see planetary computer docs). Default is True.
        output_value_multiplier(int). What to multiply output values by before
            casting to int. Default is 10000.
        output_nodata(int). The output nodata value. Seems to stick for DataArrays
            but not Datasets (see discussion in docs for rioxarray.to_raster for more).
            Defaults to -32767.
        scale_int16s(bool). Should arrays or Dataset variables which are already int16
            be rescaled before writing? Defaults to False.
    """

    scene_processor: Callable
    dataset_id: str
    prefix: Union[str, None] = None
    year: Union[str, None] = None
    overwrite: bool = False
    split_output_by_time: bool = False
    split_output_by_variable: bool = False
    aoi_by_tile: GeoDataFrame = field(
        default_factory=lambda: GeoDataFrame.from_file(
            Path(__file__).parent / "aoi_split_by_landsat_pathrow.gpkg"
        ).set_index(["PATH", "ROW"])
    )
    stac_loader_kwargs: Dict = field(
        default_factory=lambda: dict(epsg=8859, resampling=Resampling.nearest)
    )
    load_tile_pathrow_only: bool = False
    scene_processor_kwargs: Dict = field(default_factory=dict)
    scale_and_offset: bool = True
    send_area_to_scene_processor: bool = False
    send_item_collection_to_scene_processor: bool = False
    dask_chunksize: Dict = field(
        default_factory=lambda: dict(band=1, time=1, x=4096, y=4096)
    )
    container_client: ContainerClient = get_container_client()
    convert_output_to_int16: bool = True
    output_value_multiplier: int = 10000
    output_nodata: int = -32767
    extra_attrs: Dict = field(default_factory=dict)
    scale_int16s: bool = False
    logger: Union[Logger, None] = None

    def process_by_scene(self) -> None:
        for index, _ in tqdm(
            self.aoi_by_tile.iterrows(), total=self.aoi_by_tile.shape[0]
        ):
            print(index)
            these_areas = self.aoi_by_tile.loc[[index]]
            input_xr = loader.load(these_areas)
            output_xr = processor.process(input_xr)
            # Happens in coastlines sometimes

            if output_xr is None:
                continue
            writer.write(output_xr)




                logger.info(index, "Successfully written")

    def _get_stack(
        self,
        items,
        these_areas: GeoDataFrame,
        loader: str = "odc",
        epsg: int = 8859,
        **kwargs: Dict,
    ) -> DataArray:
        """
        Returns a DataArray from the given ItemCollection with given epsg
        and clipped to the features in `these_areas`. Uses odc.stac.load.
        """

        fxn = self._get_odc_stack if loader == "odc" else self._get_stackstac_stack
        return fxn(items, these_areas, epsg, **kwargs)

    def _get_stackstac_stack(
        self,
        item_collection: ItemCollection,
        these_areas: GeoDataFrame,
        epsg: int = 8859,
        **kwargs: Dict,
    ) -> DataArray:
        """
        Returns a DataArray from the given ItemCollection with crs set (to 8859)
        and clipped to the features in `these_areas`. Uses stackstac.stack.
        """
        return (
            stack(
                item_collection,
                # stack will take a dict but not keyed by bandname, only numbers
                # I don't know how to find the order of the dimensions from the
                # itemcollection, so this is essentially hoping x & y are always
                # last. If we want to continue to support stack then we need
                # a slicker solution, but for now I'm just testing so I will wait.
                chunksize=self.dask_chunksize.values(),
                # chunksize=4096,
                epsg=epsg,
                resolution=30,
                # Previously it only caught 404s, we are getting other errors
                errors_as_nodata=(RasterioIOError(".*"),),
                **kwargs,
            )
            .rio.write_crs(epsg)
            .rio.clip(
                these_areas.to_crs(epsg).geometry,
                all_touched=True,
                from_disk=True,
            )
        )

    def _get_path(
        self, index, time: Union[str, None] = None, variable: Union[str, None] = None
    ) -> str:
        if variable is None:
            variable = self.dataset_id
        if time is None:
            time = self.year

        prefix = f"{self.prefix}/" if self.prefix is not None else ""
        time = time.replace("/", "_") if time is not None else time
        suffix = "_".join([str(i) for i in index])
        return (
            f"{prefix}{self.dataset_id}/{time}/{variable}_{time}_{suffix}.tif"
            if time is not None
            else f"{prefix}{self.dataset_id}/{variable}_{suffix}.tif"
        )


def run_processor(
    scene_processor: Callable,
    dataset_id: str,
    n_workers: int = 100,
    worker_cores: int = 1,
    worker_memory: int = 8,
    **kwargs,
) -> None:
    """
    Creates a Processor object and calls process_by_scene with the given
    arguments. Tries to do so using a dask GatewayCluster with the given
    number of workers. If one is not available, uses a local dask client.
    """
    processor = Processor(scene_processor, dataset_id, **kwargs)
    try:
        cluster = GatewayCluster(worker_cores=worker_cores, worker_memory=worker_memory)
        cluster.scale(n_workers)
        with cluster.get_client() as client:
            print(client.dashboard_link)
            processor.process_by_scene()
    except ValueError:
        with Client() as client:
            print(client.dashboard_link)
            processor.process_by_scene()
