from abc import ABC, abstractmethod
from logging import Logger, getLogger

from dask.distributed import Client
from dask_gateway import GatewayCluster
from geopandas import GeoDataFrame
from tqdm import tqdm

from .exceptions import EmptyCollectionError, NoOutputError
from .loaders import Loader
from .processors import Processor
from .writers import Writer


class Runner(ABC):
    def __init__(
        self,
        tasks,
        loader: Loader,
        processor: Processor,
        writer: Writer,
        logger: Logger = getLogger(),
    ):
        self.tasks = tasks
        self.loader = loader
        self.processor = processor
        self.writer = writer
        self.logger = logger

    @abstractmethod
    def run(self):
        pass


class AreasRunner(Runner):
    def __init__(self, tasks: GeoDataFrame, continue_on_error: bool = True, **kwargs):
        super().__init__(tasks, **kwargs)
        self.continue_on_error = continue_on_error

    def run_one(self, index):
        these_areas = self.tasks.loc[[index]]

        # Search for items and lazy load them
        input_data = self.loader.load(these_areas)

        processor_kwargs = (
            dict(area=these_areas)
            if self.processor.send_area_to_processor
            else dict()
        )
        # Run the data processing
        output_data = self.processor.process(input_data, **processor_kwargs)

        if output_data is None:
            raise NoOutputError("No data was returned from the processor")

        # Write the data to disk/blob store/whatever
        return self.writer.write(output_data, index)

    def run(self):
        for index, _ in tqdm(self.tasks.iterrows(), total=self.tasks.shape[0]):
            try:
                # Here's where the work happens
                paths = self.run_one(index)
                self.logger.info([index, "complete", paths])
            except EmptyCollectionError:
                self.logger.debug([index, "no items for areas"])
                continue
            except Exception as e:
                if not self.continue_on_error:
                    self.logger.debug([index, f"ignoring error, {e}"])
                    raise e
                else:
                    self.logger.error([index, "error", "", e])


def run_by_area(
    areas: GeoDataFrame,
    loader: Loader,
    processor: Processor,
    writer: Writer,
    logger: Logger = getLogger(),
    **kwargs
) -> None:
    AreasRunner(
        areas,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
        **kwargs
    ).run()


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
        run_by_area_dask_local(**kwargs)


def run_by_area_dask_local(local_cluster_kwargs=dict(), **kwargs):
    with Client(**local_cluster_kwargs):
        run_by_area(**kwargs)
