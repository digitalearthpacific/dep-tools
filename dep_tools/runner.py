from logging import getLogger, Logger

from dask.distributed import Client
from dask_gateway import GatewayCluster
from geopandas import GeoDataFrame
from rasterio.errors import RasterioError, NotGeoreferencedWarning
from tqdm import tqdm

from .exceptions import EmptyCollectionError
from .loaders import Loader
from .processors import Processor
from .writers import Writer


def run_by_area(
    areas: GeoDataFrame,
    loader: Loader,
    processor: Processor,
    writer: Writer,
    logger: Logger = getLogger(),
) -> None:
    for index, _ in tqdm(areas.iterrows(), total=areas.shape[0]):
        these_areas = areas.loc[[index]]

        try:
            input_data = loader.load(these_areas)
        except EmptyCollectionError:
            logger.debug([index, "no items for areas"])
            continue

        processor_kwargs = (
            dict(area=these_areas) if processor.send_area_to_processor else dict()
        )
        output_data = processor.process(input_data, **processor_kwargs)

        if output_data is None:
            logger.debug([index, "no output from processor"])
            continue

        try:
            paths = writer.write(output_data, index)
        except RasterioError as e:
            logger.error([index, "r/w error", "", e])
            continue

        logger.info([index, "complete", paths])


def run_by_area_dask(
    n_workers: int = 100, worker_cores: int = 1, worker_memory: int = 8, **kwargs
) -> None:
    try:
        cluster = GatewayCluster(worker_cores=worker_cores, worker_memory=worker_memory)
        cluster.scale(n_workers)
        with cluster.get_client() as client:
            print(client.dashboard_link)
            run_by_area(**kwargs)
    except ValueError:
        with Client():
            run_by_area(**kwargs)
