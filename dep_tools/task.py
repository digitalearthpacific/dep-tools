from abc import ABC, abstractmethod
from logging import Logger, getLogger

from geopandas import GeoDataFrame
from pystac import Item

from .exceptions import EmptyCollectionError, NoOutputError
from .loaders import Loader, StacLoader
from .processors import Processor
from .namers import S3ItemPath
from .searchers import Searcher
from .stac_utils import set_stac_properties, StacCreator, copy_stac_properties
from .writers import Writer, AwsDsCogWriter, AwsStacWriter

TaskID = str


class Task(ABC):
    def __init__(
        self,
        task_id: TaskID,
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
        id: TaskID,
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


class StacTask(AreaTask):
    def __init__(
        self,
        id: TaskID,
        area: GeoDataFrame,
        searcher: Searcher,
        loader: StacLoader,
        processor: Processor,
        writer: Writer,
        post_processor: Processor | None = None,
        stac_creator: StacCreator | None = None,
        stac_writer: Writer | None = None,
        logger: Logger = getLogger(),
    ):
        """Implementation of a typical DEP product pipeline as a series of
        generic operations. For a defined location (identified by an id and
        a spatial region), search for and then load appropriate input data,
        process that data to create an output, optionally post-process that
        output to prep for writing, then write. Also allows for
        optional creation and writing of a stac item document.
        """
        super().__init__(id, area, loader, processor, writer, logger)
        self.id = id
        self.searcher = searcher
        self.post_processor = post_processor
        self.stac_creator = stac_creator
        self.stac_writer = stac_writer

    def run(self):
        items = self.searcher.search(self.area)
        input_data = self.loader.load(items, self.area)

        processor_kwargs = (
            dict(area=self.area) if self.processor.send_area_to_processor else dict()
        )

        output_data = set_stac_properties(
            input_data, self.processor.process(input_data, **processor_kwargs)
        )

        if self.post_processor is not None:
            output_data = self.post_processor.process(output_data)

        paths = self.writer.write(output_data, self.id)

        if self.stac_creator is not None and self.stac_writer is not None:
            stac_item = self.stac_creator.process(output_data, self.id)
            self.stac_writer.write(stac_item, self.id)

        return paths


class AwsStacTask(StacTask):
    def __init__(
        self,
        itempath: S3ItemPath,
        id: TaskID,
        area,
        searcher: Searcher,
        loader: StacLoader,
        processor: Processor,
        post_processor: Processor | None = None,
        logger: Logger = getLogger(),
        **kwargs,
    ):
        """A StacTask with typical parameters to write to s3 storage."""
        writer = kwargs.pop("writer", AwsDsCogWriter(itempath))
        stac_creator = kwargs.pop("stac_creator", StacCreator(itempath))
        stac_writer = kwargs.pop("stac_writer", AwsStacWriter(itempath))
        super().__init__(
            id=id,
            area=area,
            searcher=searcher,
            loader=loader,
            processor=processor,
            post_processor=post_processor,
            writer=writer,
            stac_creator=stac_creator,
            stac_writer=stac_writer,
            logger=logger,
            **kwargs,
        )


class ItemStacTask(Task):
    def __init__(
        self,
        id: TaskID,
        item: Item,
        loader: StacLoader,
        processor: Processor,
        writer: Writer,
        post_processor: Processor | None = None,
        stac_creator: StacCreator | None = None,
        stac_writer: Writer | None = None,
        logger: Logger = getLogger(),
    ):
        """A task for a single stac item. Used for example, to create output for
        every Landsat or Sentinel-2 scene. The two differences from the "usual"
        processing via `StacTask` or `AreaTask` is that thre is no `area` parameter,
        and that all properties from the input stac item are copied to the output
        xarray."""
        super().__init__(
            task_id=id, loader=loader, processor=processor, writer=writer, logger=logger
        )
        self.post_processor = post_processor
        self.stac_creator = stac_creator
        self.stac_writer = stac_writer
        self.item = item

    def run(self):
        input_data = self.loader.load([self.item])

        output_data = copy_stac_properties(
            self.item, self.processor.process(input_data)
        )

        if self.post_processor is not None:
            output_data = self.post_processor.process(output_data)

        paths = self.writer.write(output_data, self.id)

        if self.stac_creator is not None and self.stac_writer is not None:
            stac_item = self.stac_creator.process(output_data, self.id)
            self.stac_writer.write(stac_item, self.id)

        return paths


class ErrorCategoryAreaTask(AreaTask):
    def run(self):
        try:
            input_data = self.loader.load(self.area)
        except EmptyCollectionError as e:
            self.logger.error([self.id, "no items for areas"])
            raise e

        except Exception as e:
            self.logger.error([self.id, "load error", e])
            raise e

        processor_kwargs = (
            dict(area=self.area) if self.processor.send_area_to_processor else dict()
        )
        try:
            output_data = self.processor.process(input_data, **processor_kwargs)
        except Exception as e:
            self.logger.error([self.id, "processor error", e])
            raise e

        if output_data is None:
            self.logger.error([self.id, "no output from processor"])
            raise NoOutputError()
        try:
            paths = self.writer.write(output_data, self.id)
        except Exception as e:
            self.logger.error([self.id, "error", e])
            raise e

        self.logger.info([self.id, "complete", paths])

        return paths


class MultiAreaTask:
    def __init__(
        self,
        ids: list[TaskID],
        areas: GeoDataFrame,
        logger,
        task_class: type[AreaTask],
        fail_on_error: bool = True,
        **kwargs,
    ):
        self.ids = ids
        self.areas = areas
        self.task_class = task_class
        self.fail_on_error = fail_on_error
        self.logger = logger
        self._kwargs = kwargs

    def run(self):
        for id in self.ids:
            try:
                paths = self.task_class(id, self.areas.loc[[id]], **self._kwargs).run()
                self.logger.info([id, "complete", paths])
            except Exception as e:
                if self.fail_on_error:
                    raise e
                self.logger.error([id, "error", [], e])
                continue


class SimpleLoggingAreaTask(AreaTask):
    def run(self):
        self.logger.info("Preparing to load data")
        input_data = self.loader.load(self.area)
        self.logger.info(f"Found {len(input_data.time)} timesteps to load")

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
