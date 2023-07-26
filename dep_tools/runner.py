from dataclasses import dataclass
from logging import getLogger, Logger

from dask.distributed import Client
from dask_gateway import GatewayCluster
from geopandas import GeoDataFrame
from tqdm import tqdm

from .exceptions import EmptyCollectionError
from .loaders import Loader
from .processors import Processor
from .writers import Writer


@dataclass
class Runner:
    areas: GeoDataFrame
    loader: Loader
    processor: Processor
    writer: Writer
    logger: Logger = getLogger()

    def run_by_area(self) -> None:
        for index, _ in tqdm(self.areas.iterrows(), total=self.areas.shape[0]):
            print(index)
            these_areas = self.areas.loc[[index]]

            try:
                input_xr = self.loader.load(these_areas)
            except EmptyCollectionError:
                self.logger.debug([index, "no items for areas"])
                continue

            processor_kwargs = (
                dict(area=these_areas)
                if self.processor.send_area_to_processor
                else dict()
            )
            output_xr = self.processor.process(input_xr, **processor_kwargs)

            if output_xr is None:
                self.logger.debug([index, "no output from processor"])
                continue

            #           self.writer.write(output_xr, index)
            self.logger.info([index, "complete"])
            breakpoint()


def run(
    areas: GeoDataFrame,
    loader: Loader,
    processor: Processor,
    writer: Writer,
    logger: Logger = getLogger(),
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
    runner = Runner(
        areas=areas,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
        **kwargs,
    )
    try:
        cluster = GatewayCluster(worker_cores=worker_cores, worker_memory=worker_memory)
        cluster.scale(n_workers)
        with cluster.get_client() as client:
            print(client.dashboard_link)
            runner.run_by_area()
    except ValueError:
        with Client() as client:
            print(client.dashboard_link)
            runner.run_by_area()
