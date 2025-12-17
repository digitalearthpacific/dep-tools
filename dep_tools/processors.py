"""Definition of base class and implementations of :class:`Processor` objects.

:class:`Processor` objects process input data to produce output data. As such,
they are the most likely to be written for custom processing.
"""

from abc import ABC, abstractmethod

from xarray import DataArray, Dataset

from .landsat_utils import mask_clouds as mask_clouds_landsat
from .s2_utils import mask_clouds as mask_clouds_s2
from .utils import scale_and_offset, scale_to_int16


class Processor(ABC):
    """A Processor converts input data to output data.

    Args:
        send_area_to_processor: Whether to send the input area
            (typically used by a loader to load appropriate data)
            to the processor.
    """

    def __init__(self, send_area_to_processor: bool = False):
        self.send_area_to_processor = send_area_to_processor

    @abstractmethod
    def process(self, input_data):
        """Process the data.

        Args:
            input_data (Any): Any data.
        """
        pass


class LandsatProcessor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = True,
        mask_clouds: bool = True,
        mask_clouds_kwargs: dict = dict(),
    ) -> None:
        """A :class:`Processor` for use with Landsat data.

        Typically this Processor will be subclassed when working with
        Landsat data

        Args:
            scale_and_offset:
                Whether to scale and offset the input data.
                Landsat data is typically stored in 16-bit integers, this applies
                the standard scale and offset values to each band for surface
                reflectance data and (as a side effect) converts the data type
                to floating point.
            mask_clouds:
                Whether to mask_clouds,
                using :func:`dep_tools.landsat_utils.mask_clouds_landsat`.
            mask_clouds_kwargs:
                Additional arguments to
                :func:`dep_tools.landsat_utils.mask_clouds_landsat`.
        """
        super().__init__(send_area_to_processor)
        self.scale_and_offset = scale_and_offset
        self.mask_clouds = mask_clouds
        self.mask_kwargs = mask_clouds_kwargs

    def process(self, xr: DataArray | Dataset) -> DataArray | Dataset:
        """Process the data.

        Args:
            xr: Any input data, but to benefit from the functionality of this
                class, input should be Landsat surface reflectance data,
                typically with the `"QA_PIXEL"` band as well.
        Returns:
            The input data, optionally with clouds masked and/or scale and
            offset applied.
        """
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
        scale_and_offset: bool = False,
        mask_clouds: bool = True,
        mask_clouds_kwargs: dict = dict(),
    ) -> None:
        """A :class:`Processor` for use with Sentinel-2 data.

        Typically this Processor will be subclassed when working with
        Sentinel-2 data.

        Args:
            scale_and_offset:
                Whether to scale and offset the input data.
                Landsat data is typically stored in 16-bit integers, this applies
                the standard scale and offset values to each band for surface
                reflectance data and (as a side effect) converts the data type
                to floating point.
            mask_clouds:
                Whether to mask_clouds,
                using :func:`dep_tools.s2_utils.mask_clouds`.
            mask_clouds_kwargs:
                Additional arguments to
                :func:`dep_tools.s2_utils.mask_clouds`.
        """
        super().__init__(send_area_to_processor)
        self.scale_and_offset = scale_and_offset
        self.mask_clouds = mask_clouds
        self.mask_clouds_kwargs = mask_clouds_kwargs

    def process(self, xr: DataArray) -> DataArray:
        """Process the data.

        Args:
            xr: Any input data, but to benefit from the functionality of this
                class, input should be Sentinel-2 data, typically including the
                `"SCL"` band.
        Returns:
            The input data, optionally with clouds masked and/or scale and
            offset applied.
        """
        if self.mask_clouds:
            xr = mask_clouds_s2(xr, **self.mask_clouds_kwargs)

        if self.scale_and_offset and not self.harmonize_to_old:
            print(
                "Warning: scale and offset is dangerous when used without harmonize_to_old"
            )

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
        """A Processor with typical things to do to output data.

        Some :class:`Task` objects allow for use of a Processor to prep
        data for writing after the actual processing. This is mostly a wrapper
        around :func:`scale_to_int16`.

        Args:
            convert_to_int16: Whether to convert output data to 16-bit (signed)
                integer.
            output_value_multiplier: A multiplier to apply to the input data.
            scale_int16s: Whether data which is already 16-bit signed integer should
                be scaled using `output_value_multiplier`.
            output_nodata: The `nodata` value to be declared in the output.
            extra_attrs: Extra attributes to add to the output data.
        """
        self._convert_to_int16 = convert_to_int16
        self._output_value_multiplier = output_value_multiplier
        self._scale_int16s = scale_int16s
        self._output_nodata = output_nodata
        self._extra_attrs = extra_attrs

    def process(self, xr: DataArray | Dataset):
        """Process the data.

        Args:
            xr: Any input data.

        Returns:
            The input data, with scaling, type-conversion and other adjustments
            applied.
        """
        xr.attrs.update(self._extra_attrs)
        if self._convert_to_int16:
            xr = scale_to_int16(
                xr,
                output_multiplier=self._output_value_multiplier,
                output_nodata=self._output_nodata,
                scale_int16s=self._scale_int16s,
            )
        return xr
