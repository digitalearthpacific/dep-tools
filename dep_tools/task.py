"""Tasks form the core of the DEP scaling procedure. They orchestrate tasks
by loading data, processing it, and writing the output. Tasks can be generic
but are sometimes fine-tuned for specific processing.
"""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import Any

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
    """The abstract base for Task objects.

    Task objects load data, process it, and write output. They are
    reusable for the same task operating on new data.

    Args:
        id: An identifier for a particular task.
        loader: A loader loads data, usually based on the id.
        processor: A processor processes loaded data.
        writer: A writer writes output from the processor.
        logger: A logger records processing information.
    """

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
    def run(self) -> Any:
        """The run method orchestrates the processing. Usually a typical
        workflow would be load -> process -> write.

        Returns:
            Typically a task will return artifacts from the processing (like
            a file path).
        """
        pass


class AreaTask(Task):
    """An AreaTask adds an `area` property to a basic :py:class:`Task`.

    Most other arguments are as for :py:class:`Task`.

    Args:
        area: An area for use by the loader and/or processor. For instance,
            it can be used to clip data.
    """

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

    def run(self) -> str | list[str] | None:
        """Run the task.

        Returns:
            The output of the writer, typically a list of paths as strings.
        """
        input_data = self.loader.load(self.area)

        processor_kwargs = (
            dict(area=self.area) if self.processor.send_area_to_processor else dict()
        )
        output_data = self.processor.process(input_data, **processor_kwargs)
        paths = self.writer.write(output_data, self.id)
        return paths


class StacTask(AreaTask):
    """A StacTask extends :py:class:`AreaTask` by adding a searcher and optional
    post-processor, STAC creator, and STAC writer.

    Most arguments (id, area, loader, processor, writer, logger) are as for
    :py:class:`AreaTask`.

    Args:
        searcher: The searcher searches for data, typically on the basis of
            the id and/or the area.
        post_processor: A :py:class:`Processor` that can prep data for writing,
            for example scaling or data type conversions.
        stac_creator: Creates a STAC Item from the data.
        stac_writer: Writes the STAC Item to storage.
    """

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
        """ """
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
    """A convenience class with values of `writer`, `stac_creator`, and
    `stac_writer` set to sensible defaults for writing to S3.

    By default, an :py:class:`AwsDsCogWriter` is used as the primary writer,
    an :py:class:`AwsStacWriter` is used to write STAC Items, and the base
    :py:class:`StacCreator` is used to create the STAC object.

    All other arguments are as for :py:class:`StacTask`.

    Args:
        **kwargs: Additional arguments passed to :py:class:`StacTask`.
    """

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
    """A task for a single STAC item.

    Most arguments are as for :py:class:`StacTask`, except `area` is dropped.

    Args:
        item: A :py:class:`pystac.Item` representing the input data.
    """

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
    """An AreaTask with extra logging.

    Errors logged include:
        - :py:exc:`EmptyCollectionError` from loader: logged as "no items for areas"
        - Other :py:exc:`Exception` from loader: logged as "load error"
        - :py:exc:`Exception` from processor: logged as "processor error"
        - Empty processor output: logged as "no output from processor"
        - Writer error: logged as "error" (could be from writer or something
          beforehand if using dask)
    """

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
    """A "Task" object that iterates over multiple IDs and runs a task for each.

    This class is useful when running multiple short tasks where the time
    to build the run environment (for instance, if running on a pod) adds
    considerably to overall processing time.

    Args:
        ids: A list of IDs.
        areas: A :py:class:`geopandas.GeoDataFrame` with index corresponding to the IDs.
        task_class: The :py:class:`AreaTask` subclass to use for each task.
        fail_on_error: If True, will exit on error. Otherwise, will log the full
            exception and continue.
        logger: A logger.
        **kwargs: Additional arguments to the `task_class` constructor.
    """

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
    """An AreaTask with basic logging.

    All arguments are as for :py:class:`AreaTask`.
    """

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
