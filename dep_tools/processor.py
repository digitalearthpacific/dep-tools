from abc import ABC, abstractmethod
from xarray import DataArray

from .landsat_utils import mask_clouds
from .utils import scale_and_offset


class Processor(ABC):
    def __init__(self, send_area_to_processor: bool = False):
        self.send_area_to_processor = send_area_to_processor

    @abstractmethod
    def process(self, xr: DataArray) -> DataArray:
        pass


class LandsatProcessor(Processor):
    def __init__(
        self, send_area_to_processor: bool = False, scale_and_offset: bool = True
    ) -> None:
        super().__init__(send_area_to_processor)
        self.scale_and_offset = scale_and_offset

    def process(self, xr: DataArray) -> DataArray:
        xr = mask_clouds(xr)
        if self.scale_and_offset:
            # These values only work for SR bands of landsat. Ideally we could
            # read from metadata. _Really_ ideally we could just pass "scale"
            # to rioxarray/stack/odc.stac.load but apparently that doesn't work.
            scale = 0.0000275
            offset = -0.2
            xr = scale_and_offset(xr, scale=[scale], offset=offset)

        return xr
