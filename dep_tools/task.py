from abc import ABC, abstractmethod
from logging import getLogger, Logger


from geopandas import GeoDataFrame

from .exceptions import EmptyCollectionError, NoOutputError
from .loaders import Loader
from .processors import Processor
from .writers import Writer


class Task(ABC):
    def __init__(
        self,
        loader: Loader,
        processor: Processor,
        writer: Writer,
        logger: Logger = getLogger(),
    ):
        self.loader = loader
        self.processor = processor
        self.writer = writer
        self.logger = logger

    @abstractmethod
    def run(self):
        pass


class AreaTask(Task):
    def run(self, area: GeoDataFrame):
        index = area.index[0]
        try:
            input_data = self.loader.load(area)
        except EmptyCollectionError as e:
            self.logger.debug([index, "no items for areas"])
            raise e

        except Exception as e:
            self.logger.debug([index, "load error", e])
            raise e

        processor_kwargs = (
            dict(area=area) if self.processor.send_area_to_processor else dict()
        )
        try:
            output_data = self.processor.process(input_data, **processor_kwargs)
        except Exception as e:
            self.logger.debug([index, "processor error", e])
            raise e

        if output_data is None:
            self.logger.debug([index, "no output from processor"])
            raise NoOutputError()
        try:
            paths = self.writer.write(output_data, index)
        except Exception as e:
            # I just put "error" here because it could be more than
            # a write error due to dask.
            self.logger.error([index, "error", "", e])
            raise e

        self.logger.info([index, "complete", paths])
