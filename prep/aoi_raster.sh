#!/usr/bin/env bash

gdal_rasterize -burn 1 -a_nodata 0 -co COMPRESS=CCITTFAX4 -co TILED=YES -co NBITS=1 -co SPARSE_OK=TRUE -tr 30 30 -ot Byte data/aoi.gpkg data/aoi.tif
