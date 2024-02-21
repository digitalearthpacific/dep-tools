from abc import ABC, abstractmethod
from dataclasses import dataclass


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
    zero_pad_numbers: bool = False

    def __post_init__(self):
        self.version = self.version.replace(".", "-")
        self._folder_prefix = f"dep_{self.sensor}_{self.dataset_id}/{self.version}"
        self.item_prefix = f"dep_{self.sensor}_{self.dataset_id.replace('/','_')}"

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


class LocalPath(DepItemPath):
    def __init__(self, local_folder: str, **kwargs):
        # Need to create an abc for DepItemPath and drop this
        super().__init__(**kwargs)
        self._folder_prefix = (
            f"{local_folder}/dep_{self.sensor}_{self.dataset_id}/{self.version}"
        )
