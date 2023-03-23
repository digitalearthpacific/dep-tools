import fiona
import geopandas as gpd

fiona.drvsupport.supported_drivers["KML"] = "rw"
fiona.drvsupport.supported_drivers["kml"] = "rw"

aoi_gpdf = gpd.read_file("data/aoi.gpkg")

tiles = gpd.read_file(
    "https://hls.gsfc.nasa.gov/wp-content/uploads/2016/03/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml",
    driver="KML",
)

breakpoint()
tiles_in_aoi = tiles[tiles.intersects(aoi_gpdf.unary_union)]
tiles_in_aoi.to_file("data/s2_tiles_in_aoi.gpkg")

# test this
gpd.clip(tiles_in_aoi, aoi_gpdf).to_file("data/aoi_split_by_s2_tiles.gpkg")
