from abc import ABC, abstractmethod
from logging import Logger, getLogger

from geopandas import GeoDataFrame

from .exceptions import EmptyCollectionError, NoOutputError
from .loaders import Loader
from .processors import Processor
from .writers import Writer


class Task(ABC):
    def __init__(
        self, loader: Loader, processor: Processor, writer: Writer, logger: Logger
    ):
        self.loader = loader
        self.processor = processor
        self.writer = writer
        self.logger = logger

    @abstractmethod
    def run(self):
        pass


class AreaTask(Task):
    def __init__(
        self,
        id: str,
        area: GeoDataFrame,
        loader: Loader,
        processor: Processor,
        writer: Writer,
        logger: Logger = getLogger(),
    ):
        super().__init__(loader, processor, writer, logger)
        self.area = area
        self.id = id

    def run(self):
        input_data = self.loader.load(self.area)

        processor_kwargs = (
            dict(area=self.area) if self.processor.send_area_to_processor else dict()
        )
        output_data = self.processor.process(input_data, **processor_kwargs)
        paths = self.writer.write(output_data, self.id)
        return paths


class ErrorCategoryAreaTask(AreaTask):
    def run(self):
        try:
            input_data = self.loader.load(self.area)
        except EmptyCollectionError as e:
            self.logger.debug([self.id, "no items for areas"])
            raise e

        except Exception as e:
            self.logger.debug([self.id, "load error", e])
            raise e

        processor_kwargs = (
            dict(area=self.area) if self.processor.send_area_to_processor else dict()
        )
        try:
            output_data = self.processor.process(input_data, **processor_kwargs)
        except Exception as e:
            self.logger.debug([self.id, "processor error", e])
            raise e

        if output_data is None:
            self.logger.debug([self.id, "no output from processor"])
            raise NoOutputError()
        try:
            paths = self.writer.write(output_data, self.id)
            # Return the list of paths that were written
        except Exception as e:
            self.logger.error([self.id, "error", "", e])
            raise e

        self.logger.debug([self.id, "complete", paths])

        return paths


class SimpleLoggingAreaTask(AreaTask):
    def run(self, min_timesteps: int = 0):
        self.logger.info("Preparing to load data")
        input_data = self.loader.load(self.area)
        self.logger.info(f"Found {len(input_data.time)} timesteps to load")

        if len(input_data.time) < min_timesteps:
            self.logger.warning(
                f"Found {len(input_data.time)} timesteps, which is less than the minimum of {min_timesteps}"
            )
            return []

        self.logger.info("Preparing to process data")
        processor_kwargs = {}
        if self.processor.send_area_to_processor:
            processor_kwargs["area"] = self.area
        output_data = self.processor.process(input_data, **processor_kwargs)
        self.logger.info(
            f"Processed data will have a result of shape: {[output_data.dims[d] for d in ['x', 'y']]}"
        )

        self.logger.info("Processing and writing data...")
        paths = self.writer.write(output_data, self.id)
        self.logger.info(f"Succesfully wrote data to {len(paths)} paths")

        return paths
