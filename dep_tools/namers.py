from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


# Should probably be renamed to AssetPath at earliest opportunity
# to avoid confusion with stac items
class ItemPath(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def path(self) -> str:
        return ""


@dataclass
class DepItemPath(ItemPath):
    sensor: str
    dataset_id: str
    version: str
    time: str

    def __post_init__(self):
        self.version = self.version.replace(".", "-")
        self._folder_prefix = f"dep_{self.sensor}_{self.dataset_id}/{self.version}"
        self.item_prefix = f"dep_{self.sensor}_{self.dataset_id}"

    def _format_item_id(self, item_id) -> str:
        if isinstance(item_id, List) or isinstance(item_id, Tuple):
            # Assuming we have a list like ('66,23', 'FJ')
            tile_id = item_id[0]
            region_id = item_id[1]
            item_id = f"{self._format_item_id(tile_id)}_{region_id}"
        if len(item_id.split(",")) == 2:
            # Create a zero padded len of 3 string separated by a _
            # e.g. 1_2 becomes 001_002
            item_id = "_".join([f"{int(i):03d}" for i in item_id.split(",")])
        return item_id

    def _folder(self, item_id) -> str:
        return f"{self._folder_prefix}/{self._format_item_id(item_id)}/{self.time}"

    def basename(self, item_id) -> str:
        return f"{self.item_prefix}_{self._format_item_id(item_id)}_{self.time}"  # _{asset_name}"

    def path(self, item_id, asset_name=None, ext=".tif") -> str:
        return (
            f"{self._folder(item_id)}/{self.basename(item_id)}_{asset_name}{ext}"
            if asset_name is not None
            else f"{self._folder(item_id)}/{self.basename(item_id)}{ext}"
        )

    def log_path(self) -> str:
        return f"{self._folder_prefix}/logs/{self.item_prefix}_{self.time}_log.csv"


class LocalPath(DepItemPath):
    def __init__(self, local_folder: str, **kwargs):
        # Need to create an abc for DepItemPath and drop this
        super().__init__(**kwargs)
        self._folder_prefix = local_folder

    def _folder(self, item_id) -> str:
        return self._folder_prefix
