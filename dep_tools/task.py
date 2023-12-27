from abc import ABC, abstractmethod
from logging import Logger, getLogger

from geopandas import GeoDataFrame

from .exceptions import EmptyCollectionError, NoOutputError
from .loaders import Loader
from .processors import Processor
from .writers import Writer

task_id = str


class Task(ABC):
    def __init__(
        self,
        task_id: task_id,
        loader: Loader,
        processor: Processor,
        writer: Writer,
        logger: Logger,
    ):
        self.id = task_id
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
        id: task_id,
        area: GeoDataFrame,
        loader: Loader,
        processor: Processor,
        writer: Writer,
        logger: Logger = getLogger(),
    ):
        super().__init__(id, loader, processor, writer, logger)
        self.area = area

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
            self.logger.info([self.id, "no items for areas"])
            raise e

        except Exception as e:
            self.logger.info([self.id, "load error", e])
            raise e

        processor_kwargs = (
            dict(area=self.area) if self.processor.send_area_to_processor else dict()
        )
        try:
            output_data = self.processor.process(input_data, **processor_kwargs)
        except Exception as e:
            self.logger.info([self.id, "processor error", e])
            raise e

        if output_data is None:
            self.logger.info([self.id, "no output from processor"])
            raise NoOutputError()
        try:
            paths = self.writer.write(output_data, self.id)
        except Exception as e:
            self.logger.error([self.id, "error", e])
            raise e

        self.logger.info([self.id, "complete", paths])

        return paths


class MultiAreaTask(ABC):
    def __init__(
        self,
        ids: list[task_id],
        areas: GeoDataFrame,
        task_class: type[AreaTask],
        loader: Loader,
        processor: Processor,
        writer: Writer,
        logger: Logger = getLogger(),
        fail_on_error: bool = True,
    ):
        self.ids = ids
        self.areas = areas
        self.loader = loader
        self.processor = processor
        self.writer = writer
        self.logger = logger
        self.task_class = task_class
        self.fail_on_error = fail_on_error

    def run(self):
        for id in self.ids:
            try:
                self.task_class(
                    id,
                    self.areas.loc[[id]],
                    self.loader,
                    self.processor,
                    self.writer,
                    self.logger,
                ).run()
            except Exception as e:
                if self.fail_on_error:
                    raise e
                continue
