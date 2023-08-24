from abc import ABC, abstractmethod
from dataclasses import dataclass


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

    def _folder(self, item_id) -> str:
        return f"dep_{self.sensor}_{self.dataset_id}\\{self.version}\\{item_id}\\{self.time}"

    def _basename(self, item_id, asset_name) -> str:
        return f"dep_{self.sensor}_{self.dataset_id}_{item_id}_{self.time}_{asset_name}"

    def path(self, item_id, asset_name) -> str:
        return f"{self._folder(item_id)}\\{self._basename(item_id, asset_name)}"
