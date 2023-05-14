from dataclasses import dataclass, field

# from importlib.resources import files
import json
from pathlib import Path
from typing import Dict, List, Union, Callable

from azure.storage.blob import ContainerClient
from dask.distributed import Client
from dask_gateway import GatewayCluster
from geopandas import GeoDataFrame
from pystac import ItemCollection
from rasterio import RasterioIOError
import rioxarray
from stackstac import stack
from tqdm import tqdm
from xarray import DataArray

from .landsat_utils import item_collection_for_pathrow, mask_clouds
from .utils import (
    fix_bad_epsgs,
    gpdf_bounds,
    scale_and_offset,
    search_across_180,
    scale_to_int16,
    write_to_blob_storage,
)


@dataclass
class Processor:
    """
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
    container_client: ContainerClient
    year: Union[str, None] = None
    overwrite: bool = False
    split_output_by_year: bool = False
    split_output_by_variable: bool = False
    aoi_by_tile: GeoDataFrame = GeoDataFrame.from_file(
        Path(__file__).parent / "aoi_split_by_landsat_pathrow.gpkg"
    ).set_index(["PATH", "ROW"])
    scene_processor_kwargs: Dict = field(default_factory=dict)
    scale_and_offset: bool = True
    send_area_to_scene_processor: bool = False
    dask_chunksize: int = 4096
    convert_output_to_int16: bool = True
    output_value_multiplier: int = 10000
    output_nodata: int = -32767
    scale_int16s: bool = False

    def __post_init__(self):
        self.prefix = (
            f"{self.dataset_id}/{self.year}/{self.dataset_id}_{self.year}"
            if self.year
            else f"{self.dataset_id}/{self.dataset_id}"
        )
        self.local_prefix = Path(self.prefix).stem
        self.mosaic_file = f"data/{self.local_prefix}.tif"

    def process_by_scene(self) -> None:
        for index, _ in tqdm(
            self.aoi_by_tile.iterrows(), total=self.aoi_by_tile.shape[0]
        ):
            print(index)
            these_areas = self.aoi_by_tile.loc[[index]]
            index_dict = dict(zip(self.aoi_by_tile.index.names, index))
            item_collection = search_across_180(
                these_areas,
                collections=["landsat-c2-l2"],
                datetime=self.year,
            )
            #            item_collection = item_collection_for_pathrow(
            #                # For S2, would probably just pass index_dict as kwargs
            #                # to generic function
            #                path=index_dict["PATH"],
            #                row=index_dict["ROW"],
            #                search_args=dict(
            #                    collections=["landsat-c2-l2"],
            #                    datetime=self.year,
            #                    bbox=gpdf_bounds(these_areas),
            #                ),
            #            )

            # If there are not items in this collection for _this_ pathrow,
            # we don't want to process, since they will be captured in
            # other pathrows (or are areas not covered by our aoi)
            item_collection_for_this_pathrow = [
                i
                for i in item_collection
                if i.properties["landsat:wrs_path"] == f"{index_dict['PATH']:03d}"
                and i.properties["landsat:wrs_row"] == f"{index_dict['ROW']:03d}"
            ]

            if len(item_collection_for_this_pathrow) == 0:
                continue
            fix_bad_epsgs(item_collection)

            item_xr = self._get_stack(item_collection, these_areas)
            item_xr = mask_clouds(item_xr)

            if self.scale_and_offset:
                # These values only work for SR bands of landsat. Ideally we could
                # read from metadata. _Really_ ideally we could just pass "scale"
                # to rioxarray but apparently that doesn't work.
                scale = 0.0000275
                offset = -0.2
                item_xr = scale_and_offset(item_xr, scale=[scale], offset=offset)

            if self.send_area_to_scene_processor:
                self.scene_processor_kwargs.update(dict(area=these_areas))
            results = self.scene_processor(item_xr, **self.scene_processor_kwargs)
            # Happens in coastlines sometimes
            if results is None:
                continue

            if self.convert_output_to_int16:
                results = scale_to_int16(
                    results,
                    output_multiplier=self.output_value_multiplier,
                    output_nodata=self.output_nodata,
                    scale_int16s=self.scale_int16s,
                )

            # If we want to create an output for each year, split results
            # into a list of da/ds
            if self.split_output_by_year:
                results = [results.sel(time=year) for year in results.coords["time"]]

            # If we want to create an output for each variable, split or further
            # split results into a list of da/ds for each variable
            # or variable x year
            if self.split_output_by_variable:
                results = (
                    [
                        result.to_array().sel(variable=var)
                        for result in results
                        for var in result
                    ]
                    if self.split_output_by_year
                    else [results.to_array().sel(variable=var) for var in results]
                )

            if isinstance(results, List):
                for result in results:
                    # preferable to to results.coords.get('time') but that returns
                    # a dataarray rather than a string
                    time = (
                        result.coords["time"].values.tolist()
                        if "time" in result.coords
                        else None
                    )
                    variable = (
                        result.coords["variable"].values.tolist()
                        if "variable" in result.coords
                        else None
                    )

                    write_to_blob_storage(
                        result,
                        path=self._get_path(index, time, variable),
                        write_args=dict(driver="COG", compress="LZW"),
                        overwrite=self.overwrite,
                    )
            else:
                # We cannot write outputs with > 2 dimensions using rio.to_raster,
                # so we create new variables for each year x variable combination
                # Note this requires time to represent year, so we should consider
                # doing that here as well (rather than earlier).
                if len(results.dims.keys()) > 2:
                    results = (
                        results.to_array(dim="variables")
                        .stack(z=["time", "variables"])
                        .to_dataset(dim="z")
                        .pipe(
                            lambda ds: ds.rename_vars(
                                {name: "_".join(name) for name in ds.data_vars}
                            )
                        )
                        .drop_vars(["variables", "time"])
                    )
                write_to_blob_storage(
                    # Squeeze here in case we have e.g. a single time reading
                    results.squeeze(),
                    path=self._get_path(index),
                    write_args=dict(driver="COG", compress="LZW"),
                    overwrite=self.overwrite,
                )

    def _get_stack(
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

    def _get_path(
        self, index, year: Union[str, None] = None, variable: Union[str, None] = None
    ) -> str:
        if variable is None:
            variable = self.dataset_id
        if year is None:
            year = self.year

        year = year.replace("/", "_")
        suffix = "_".join([str(i) for i in index])
        return (
            f"{self.dataset_id}/{year}/{variable}_{year}_{suffix}.tif"
            if year is not None
            else f"{self.dataset_id}/{variable}_{suffix}.tif"
        )


def run_processor(
    scene_processor: Callable,
    dataset_id: str,
    n_workers: int = 400,
    **kwargs,
) -> None:
    processor = Processor(scene_processor, dataset_id, **kwargs)
    # Try to run on a gateway cluster, if one is not configured,
    # fall back to local cluster
    try:
        cluster = GatewayCluster(worker_cores=1, worker_memory=8)
        cluster.scale(n_workers)
        with cluster.get_client() as client:
            print(client.dashboard_link)
            processor.process_by_scene()
    except ValueError:
        with Client() as client:
            print(client.dashboard_link)
            processor.process_by_scene()
