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
        self._folder_prefix = f"dep_{self.sensor}_{self.dataset_id}\\{self.version}"
        self.item_prefix = f"dep_{self.sensor}_{self.dataset_id}"

    def _folder(self, item_id) -> str:
        if isinstance(item_id, List) or isinstance(item_id, Tuple):
            item_id = "\\".join(item_id)
        return f"{self._folder_prefix}\\{item_id}\\{self.time}"

    def basename(self, item_id, asset_name) -> str:
        if isinstance(item_id, List) or isinstance(item_id, Tuple):
            item_id = "_".join(item_id)
        return f"{self.item_prefix}_{item_id}_{self.time}_{asset_name}"

    def path(self, item_id, asset_name, ext=".tif") -> str:
        return f"{self._folder(item_id)}\\{self.basename(item_id, asset_name)}{ext}"

    def log_path(self) -> str:
        return f"{self._folder_prefix}\\{self.item_prefix}_log.csv"
