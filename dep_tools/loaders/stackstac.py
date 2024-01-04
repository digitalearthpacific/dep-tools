class StackStacLoader(StacLoader):
    def __init__(
        self, stack_kwargs=dict(resolution=30), resamplers_and_assets=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.stack_kwargs = stack_kwargs
        self.resamplers_and_assets = resamplers_and_assets

    def load(
        self,
        items,
        areas: GeoDataFrame,
    ) -> DataArray:
        areas_proj = areas.to_crs(self._current_epsg)
        if self.resamplers_and_assets is not None:
            s = concat(
                [
                    stack(
                        items,
                        chunksize=self.dask_chunksize,
                        epsg=self._current_epsg,
                        errors_as_nodata=(RasterioError(".*"),),
                        assets=resampler_and_assets["assets"],
                        resampling=resampler_and_assets["resampler"],
                        bounds=areas_proj.total_bounds.tolist(),
                        band_coords=False,  # needed or some coords are often missing
                        # from qa pixel and we get an error. Make sure it doesn't
                        # mess anything up else where (e.g. rio.crs)
                        **self.stack_kwargs,
                    )
                    for resampler_and_assets in self.resamplers_and_assets
                ],
                dim="band",
            )
        else:
            s = stack(
                items,
                chunksize=self.dask_chunksize,
                epsg=self._current_epsg,
                errors_as_nodata=(RasterioIOError(".*"),),
                **self.stack_kwargs,
            )

        return s.rio.write_crs(self._current_epsg).rio.clip(
            areas_proj.geometry,
            all_touched=True,
            from_disk=True,
        )
