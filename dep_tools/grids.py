from odc.geo import XY
from odc.geo.gridspec import GridSpec

# This EPSG code is what we're using for now
# but it's not ideal, as its not an equal area projection...
PACIFIC_EPSG = "EPSG:3832"

# The origin is in the projected CRS. This works for Landsat.
PACIFIC_GRID_30 = GridSpec(
    crs=PACIFIC_EPSG,
    tile_shape=(3200, 3200),
    resolution=30,
    origin=XY(-3000000, -4000000),
)

# This grid is for Sentinel-2 and has the same footprint
PACIFIC_GRID_10 = GridSpec(
    crs=PACIFIC_EPSG,
    tile_shape=(9600, 9600),
    resolution=10,
    origin=XY(-3000000, -4000000),
)
