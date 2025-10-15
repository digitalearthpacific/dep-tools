"""Base class and implementation of :class:`Namer` objects, which are used to
define paths for output data and accompanying files, such as STAC Items.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from .aws import get_s3_bucket_region
from .utils import join_path_or_url


class ItemPath(ABC):
    """An ItemPath returns a path."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def path(self, *args, **kwargs) -> str:
        return ""


class GenericItemPath(ItemPath):
    """A GenericItemPath represents a path on a network or local drive."""

    def __init__(
        self,
        sensor: str,
        dataset_id: str,
        version: str,
        time: str,
        prefix: str = "dep",
        zero_pad_numbers: bool = True,
        full_path_prefix: str | Path | None = None,
    ):
        """
        Args:
            sensor: The satellite sensor, typically "ls" for Landsat or "s2"
                for Sentinel-2.
            dataset_id: The identifier for this product, such as "geomad".
            version: A version identified.
            time: A time, such as "2012".
            prefix: The prefix for the product, typically representing the
                data producer, such as "dep" for Digital Earth Pacific.
            zero_pad_numbers: Whether to pad numbers in the item id (see below)
                to three digits.
            full_path_prefix: Something to put at the beginning of the full path
                representation of the path, such as "/home/".
        """
        self.sensor = sensor
        self.dataset_id = dataset_id
        self.version = version
        self.time = time
        self.prefix = prefix
        self.zero_pad_numbers = zero_pad_numbers
        self.version = self.version.replace(".", "-")
        self.full_path_prefix = full_path_prefix
        self._folder_prefix = (
            f"{self.prefix}_{self.sensor}_{self.dataset_id}/{self.version}"
        )
        self.item_prefix = (
            f"{self.prefix}_{self.sensor}_{self.dataset_id.replace('/','_')}"
        )

    def _format_item_id(
        self, item_id: list[str | int] | tuple[str | int] | str, join_str="/"
    ) -> str:
        """Zero pads to 3 characters anything (string or int) that is numeric-like
        and joins list/tuple items or items of a comma-separated string with `join_str`
        """
        if isinstance(item_id, list | tuple):
            item_parts = item_id
        elif len(item_id.split(",")) > 1:
            item_parts = item_id.split(",")
        else:
            item_parts = [item_id]

        item_parts = [
            str(i).zfill(3) if str(i).isnumeric() and self.zero_pad_numbers else str(i)
            for i in item_parts
        ]
        return join_str.join(item_parts)

    def _folder(self, item_id) -> str:
        return f"{self._folder_prefix}/{self._format_item_id(item_id)}/{self.time}"

    def basename(self, item_id: list[str | int] | tuple[str | int] | str) -> str:
        """The stem of the file name, without any parent folders.

        Args:
            item_id: The item id. If a list, items are converted to string, optionally
                zero-padded (according to the class parameter `zero_pad_numbers`),
                and joined using an underscore. Otherwise, the value is used directly.

        Returns:
            A string.

        """
        return f"{self.item_prefix}_{self._format_item_id(item_id, join_str='_')}_{self.time}"

    def path(
        self,
        item_id: list[str | int] | tuple[str | int] | str,
        asset_name: str | None = None,
        ext: str = ".tif",
        absolute: bool = False,
    ) -> str:
        """

        Args:
            item_id: The item id.
            asset_name: If the thing we are naming has multiple assets / bands, this
                is the name of the asset.
            ext: The extension for the path.
            absolute: Whether to return the absolute path.

        Returns:
            The path as a string.

        """
        relative_path = (
            f"{self._folder(item_id)}/{self.basename(item_id)}_{asset_name}{ext}"
            if asset_name is not None
            else f"{self._folder(item_id)}/{self.basename(item_id)}{ext}"
        )

        return (
            join_path_or_url(self.full_path_prefix, relative_path)
            if absolute and self.full_path_prefix is not None
            else relative_path
        )

    def stac_path(
        self, item_id: list[str | int] | tuple[str | int] | str, **kwargs
    ) -> str:
        """The path to the STAC item.

        Args:
            item_id: The item id.
            **kwargs: Additional arguments to :py:func:`GenericItemPath.path`.

        Returns:
            The path as a string.

        """
        return self.path(item_id, ext=".stac-item.json", **kwargs)

    def log_path(self) -> str:
        """The path to the log file.

        The path represents a csv file and is not used in all processing.

        Returns:
            The path as a string.

        """
        return f"{self._folder_prefix}/logs/{self.item_prefix}_{self.time}_log.csv"


class DepItemPath(GenericItemPath):
    """A DepItemPath is just a renamed GenericItemPath, for backwards compatibility."""

    pass


class S3ItemPath(GenericItemPath):
    """An ItemPath for something on Amazon S3 storage.

    Attributes:
        bucket: The name of the bucket.
    """
    def __init__(
        self,
        bucket: str,
        sensor: str,
        dataset_id: str,
        version: str,
        time: str,
        prefix: str = "dep",
        zero_pad_numbers: bool = True,
        full_path_prefix: str | None = None,
        make_hrefs_https: bool = True,
    ):
        super().__init__(
            sensor=sensor,
            dataset_id=dataset_id,
            version=version,
            time=time,
            prefix=prefix,
            zero_pad_numbers=zero_pad_numbers,
        )
        self.bucket = bucket
        if full_path_prefix is not None:
            self.full_path_prefix = full_path_prefix
        elif make_hrefs_https:
            aws_region = get_s3_bucket_region(bucket)
            # E.g., https://dep-public-prod.s3.us-west-2.amazonaws.com/
            self.full_path_prefix = f"https://{bucket}.s3.{aws_region}.amazonaws.com/"
        else:
            self.full_path_prefix = f"s3://{bucket}/"


class DailyItemPath(S3ItemPath):
    """An S3ItemPath which produces approprate paths for "daily" items where the time
    is a datetime parseable by datetime.datetime.fromisoformat.
    Folders will have format YYYY/mm/dd, and stems will have dates in the
    format YYYY-mm-dd.
    Example: a time value of `"2025-06-13 15:56:54.012509"`
    would produce
    dep_ls_wofl/99/77/2025/06/13/dep_ls_wofl_99_77_2025-06-13.tif
    """

    def __init__(self, time: str | datetime, **kwargs):
        super().__init__(time=str(time), **kwargs)
        self.datetime = (
            datetime.fromisoformat(time) if not isinstance(time, datetime) else time
        )
    """
        Args:
            time: The time.
            **kwargs: Additonal arguments to :py:class:`S3ItemPath`.
    """

    def _folder(self, item_id) -> str:
        return f"{self._folder_prefix}/{self._format_item_id(item_id)}/{self.datetime:%Y/%m/%d}"

    def basename(self, item_id) -> str:
        return f"{self.item_prefix}_{self._format_item_id(item_id, join_str='_')}_{self.datetime:%Y-%m-%d}"


class LocalPath(DepItemPath):
    def __init__(self, local_folder: str, prefix: str = "dep", **kwargs):
        super().__init__(full_path_prefix=local_folder, prefix=prefix, **kwargs)

    def path(self, item_id, asset_name=None, ext=".tif") -> str:
        return super().path(item_id, asset_name=asset_name, ext=ext, absolute=True)
