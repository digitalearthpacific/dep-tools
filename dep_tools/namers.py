from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from odc.loader._aws import auto_find_region

from dep_tools.utils import join_path_or_url


class ItemPath(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def path(self, *args, **kwargs) -> str:
        return ""


class GenericItemPath(ItemPath):
    def __init__(
        self,
        sensor: str,
        dataset_id: str,
        version: str,
        time: str,
        prefix: str = "dep",
        zero_pad_numbers: bool = True,
        full_path_prefix: str | Path | None = Path("./"),
    ):
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

    def basename(self, item_id) -> str:
        return f"{self.item_prefix}_{self._format_item_id(item_id, join_str='_')}_{self.time}"

    def path(self, item_id, asset_name=None, ext=".tif", absolute: bool = False) -> str:
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

    def stac_path(self, item_id, **kwargs):
        return self.path(item_id, ext=".stac-item.json", **kwargs)

    def log_path(self) -> str:
        return f"{self._folder_prefix}/logs/{self.item_prefix}_{self.time}_log.csv"


class DepItemPath(GenericItemPath):
    pass


class S3ItemPath(GenericItemPath):
    def __init__(
        self,
        bucket: str,
        sensor: str,
        dataset_id: str,
        version: str,
        time: str,
        prefix: str = "dep",
        zero_pad_numbers: bool = True,
        make_hrefs_https: bool = True,
        full_path_prefix: str | None = None,
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
        if make_hrefs_https:
            if self.full_path_prefix is not None:
                self.full_path_prefix = full_path_prefix
            else:
                # E.g., https://dep-public-prod.s3.us-west-2.amazonaws.com/
                aws_region = auto_find_region()
                self.full_path_prefix = (
                    f"https://{self.bucket}.s3.{aws_region}.amazonaws.com/"
                )
        else:
            self.full_path_prefix = f"s3://{self.bucket}/"


class DailyItemPath(S3ItemPath):
    """An ItemPath which produces approprate paths for "daily" items where the time
    is a datetime parseable by datetime.datetime.fromisoformat.
    Folders will have format YYYY/mm/dd, and stems will have dates in the
    format YYYY-mm-dd.
    Example: a time value of "2025-06-13 15:56:54.012509"
    would produce
    dep_ls_wofl/99/77/2025/06/13/dep_ls_wofl_99_77_2025-06-13.tif
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.datetime = datetime.fromisoformat(self.time)

    def _folder(self, item_id) -> str:
        return f"{self._folder_prefix}/{self._format_item_id(item_id)}/{self.datetime:%Y/%m/%d}"

    def basename(self, item_id) -> str:
        return f"{self.item_prefix}_{self._format_item_id(item_id, join_str='_')}_{self.datetime:%Y-%m-%d}"


class LocalPath(DepItemPath):
    def __init__(self, local_folder: str, prefix: str = "dep", **kwargs):
        super().__init__(full_path_prefix=local_folder, prefix=prefix, **kwargs)

    def path(self, item_id, asset_name=None, ext=".tif") -> str:
        return super().path(item_id, asset_name=asset_name, ext=ext, absolute=True)
