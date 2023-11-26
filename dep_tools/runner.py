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

    def run(self):
        for index, _ in tqdm(self.tasks.iterrows(), total=self.tasks.shape[0]):
            these_areas = self.tasks.loc[[index]]

            try:
                input_data = self.loader.load(these_areas)
            except EmptyCollectionError:
                self.logger.debug([index, "no items for areas"])
                # not raising this one
                continue

            except Exception as e:
                self.logger.debug([index, "load error", e])
                if self.continue_on_error:
                    continue
                raise e

            processor_kwargs = (
                dict(area=these_areas)
                if self.processor.send_area_to_processor
                else dict()
            )
            try:
                output_data = self.processor.process(input_data, **processor_kwargs)
            except Exception as e:
                self.logger.debug([index, "processor error", e])
                if self.continue_on_error:
                    continue
                raise e

            if output_data is None:
                self.logger.debug([index, "no output from processor"])
                if self.continue_on_error:
                    continue
                raise NoOutputError()
            try:
                paths = self.writer.write(output_data, index)
            except Exception as e:
                # I just put "error" here because it could be more than
                # a write error due to dask.
                self.logger.error([index, "error", "", e])
                if self.continue_on_error:
                    continue
                raise e

            self.logger.info([index, "complete", paths])


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
