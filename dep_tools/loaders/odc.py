class BaseOdcLoader(StacLoader):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def load(self, items, area) -> DataArray | Dataset:
        return odc.stac.load(
            items,
            geopolygon=area.to_crs(4326),
            **self._kwargs,
        )


class OdcLoader(BaseOdcLoader):
    def __init__(
        self,
        clip_to_area: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._clip_to_area = clip_to_area

    def load(self, items, area) -> Dataset | DataArray:
        ds = super().load(items, area)

        if self._clip_to_area:
            ds = ds.rio.clip(
                area.to_crs(ds.odc.crs).geometry, all_touched=True, from_disk=True
            )

        return ds
