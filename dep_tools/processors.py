from abc import ABC, abstractmethod

from xarray import DataArray, Dataset

from .landsat_utils import mask_clouds as mask_clouds_landsat
from .s2_utils import harmonize_to_old
from .s2_utils import mask_clouds as mask_clouds_s2
from .utils import scale_and_offset, scale_to_int16


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
        harmonize_to_old: bool = True,
        scale_and_offset: bool = True,
        mask_clouds: bool = True,
        mask_clouds_kwargs: dict = dict(),
    ) -> None:
        super().__init__(send_area_to_processor)
        self.harmonize_to_old = harmonize_to_old
        self.scale_and_offset = scale_and_offset
        self.mask_clouds = mask_clouds
        self.mask_kwargs = mask_clouds_kwargs

    def process(self, xr: DataArray) -> DataArray:
        if self.mask_clouds:
            xr = mask_clouds_s2(xr, **self.mask_kwargs)

        if self.scale_and_offset and not self.harmonize_to_old:
            print(
                "Warning: scale and offset is dangerous when used without harmonize_to_old"
            )

        if self.harmonize_to_old:
            xr = harmonize_to_old(xr)

        if self.scale_and_offset:
            scale = 1 / 10000
            offset = 0
            xr = scale_and_offset(xr, scale=[scale], offset=offset)

        return xr


class XrPostProcessor(Processor):
    def __init__(
        self,
        convert_to_int16: bool = True,
        output_value_multiplier: int = 10000,
        scale_int16s: bool = False,
        output_nodata: int = -32767,
        extra_attrs: dict = {},
    ):
        self._convert_to_int16 = convert_to_int16
        self._output_value_multiplier = output_value_multiplier
        self._scale_int16s = scale_int16s
        self._output_nodata = output_nodata
        self._extra_attrs = extra_attrs

    def process(self, xr: DataArray | Dataset):
        xr.attrs.update(self._extra_attrs)
        if self._convert_to_int16:
            xr = scale_to_int16(
                xr,
                output_multiplier=self._output_value_multiplier,
                output_nodata=self._output_nodata,
                scale_int16s=self._scale_int16s,
            )
        return xr
