import geopandas as gpd

pathrows = gpd.read_file(
    "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"
)

aoi = gpd.read_file("data/aoi.gpkg").unary_union

# This takes hours but it works
pathrows[pathrows.intersects(aoi)].to_file("data/pathrows_in_aoi.gpkg")
