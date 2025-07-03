from abc import ABC, abstractmethod


class ItemPath(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def path(self) -> str:
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
        folder: str = None,
    ):
        self.sensor = sensor
        self.dataset_id = dataset_id
        self.version = version
        self.time = time
        self.prefix = prefix
        self.folder = folder
        self.zero_pad_numbers = zero_pad_numbers
        self.version = self.version.replace(".", "-")
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
        return (
            f"{self._folder_prefix}/{self._format_item_id(item_id)}/{self.time}"
            if self.folder is None
            else f"{self.folder}/{self._folder_prefix}/{self._format_item_id(item_id)}/{self.time}"
        )
    def basename(self, item_id) -> str:
        return f"{self.item_prefix}_{self._format_item_id(item_id, join_str='_')}_{self.time}"

    def path(self, item_id, asset_name=None, ext=".tif") -> str:
        return (
            f"{self._folder(item_id)}/{self.basename(item_id)}_{asset_name}{ext}"
            if asset_name is not None
            else f"{self._folder(item_id)}/{self.basename(item_id)}{ext}"
        )

    def stac_path(self, item_id):
        return self.path(item_id, ext=".stac-item.json")

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
        bucket_prefix: str = None
    ):
        super().__init__(
            sensor=sensor,
            dataset_id=dataset_id,
            version=version,
            time=time,
            prefix=prefix,
            zero_pad_numbers=zero_pad_numbers,
            folder=bucket_prefix,
        )
        self.bucket = bucket


class LocalPath(DepItemPath):
    def __init__(self, local_folder: str, prefix: str = "dep", **kwargs):
        super().__init__(**kwargs)
        self._folder_prefix = (
            f"{local_folder}/{prefix}_{self.sensor}_{self.dataset_id}/{self.version}"
        )
