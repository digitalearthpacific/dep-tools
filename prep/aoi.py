import geopandas as gpd

# List of countries from the needs assessment
# Note that the existing pic_XXX have different countries.
# See https://github.com/jessjaco/dep-data/issues/1
countries = [
    "American Samoa",
    "Cook Islands",
    "Fiji",
    "French Polynesia",
    "Guam",
    "Kiribati",
    "Marshall Islands",
    "Micronesia",
    "Nauru",
    "New Caledonia",
    "Niue",
    "Northern Mariana Islands",
    "Palau",
    "Papua New Guinea",
    "Pitcairn Islands",
    "Solomon Islands",
    "Samoa",
    "Tokelau",
    "Tonga",
    "Tuvalu",
    "Vanuatu",
    "Wallis and Futuna",
]

all_polys = (
      gpd.read_file("https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-gpkg.zip").query("NAME_0 in @countries")
    #   ^^^ untested, but should work (slow, etc)
#    gpd.read_file("data/gadm_410-gpkg/gadm_410.gpkg")
)

# Note that there are some polys with TYPE_1 / ENGTYPE_1 codes of "Atol" and
# "Atoll", if that's how Sachin has in mine to define those areas (for lc
# mapping). BUT, there are also Atolls (or things that look very similar to
# them in other areas. Anyway, check out this file if you want to see
all_polys.to_file("data/aoi_split.gpkg")

all_polys.dissolve("NAME_0").to_file("data/aoi.gpkg")
