from abc import ABC, abstractmethod

from xarray import DataArray, Dataset

from .landsat_utils import mask_clouds as mask_clouds_landsat
from .s2_utils import harmonize_to_old
from .s2_utils import mask_clouds as mask_clouds_s2
from .utils import scale_and_offset


class Processor(ABC):
    def __init__(self, send_area_to_processor: bool = False):
        self.send_area_to_processor = send_area_to_processor

    @abstractmethod
    def process(self, input_data):
        pass


class LandsatProcessor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = True,
        mask_clouds: bool = True,
        mask_clouds_kwargs: dict = dict(),
    ) -> None:
        super().__init__(send_area_to_processor)
        self.scale_and_offset = scale_and_offset
        self.mask_clouds = mask_clouds
        self.mask_kwargs = mask_clouds_kwargs

    def process(self, xr: DataArray | Dataset) -> DataArray | Dataset:
        if self.mask_clouds:
            xr = mask_clouds_landsat(xr, **self.mask_kwargs)
        if self.scale_and_offset:
            # These values only work for SR bands of landsat. Ideally we could
            # read from metadata. _Really_ ideally we could just pass "scale"
            # to rioxarray/stack/odc.stac.load but apparently that doesn't work.
            scale = 0.0000275
            offset = -0.2
            xr = scale_and_offset(xr, scale=[scale], offset=offset)

        return xr


class S2Processor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = True,
        mask_clouds: bool = True,
        mask_clouds_kwargs: dict = dict(),
    ) -> None:
        super().__init__(send_area_to_processor)
        self.scale_and_offset = scale_and_offset
        self.mask_clouds = mask_clouds
        self.mask_kwargs = mask_clouds_kwargs

    def process(self, xr: DataArray) -> DataArray:
        if self.mask_clouds:
            xr = mask_clouds_s2(xr, **self.mask_kwargs)

        if self.scale_and_offset:
            xr = harmonize_to_old(xr)

        return xr
