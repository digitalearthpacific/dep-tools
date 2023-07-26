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
