from collections import Counter

from geopandas import GeoDataFrame

from dep_tools.landsat_utils import landsat_grid, read_pathrows_file


def test_landsat_grid():
    grid = landsat_grid()
    pathrows = read_pathrows_file()
    assert isinstance(grid, GeoDataFrame)
    assert Counter(grid.index.tolist()) == Counter(pathrows)


def test_read_pathrows_file():
    """Test reading pathrows from the file."""
    pathrows = read_pathrows_file()
    assert isinstance(pathrows, list)
    assert all(isinstance(pr, tuple) and len(pr) == 2 for pr in pathrows)
    assert all(isinstance(pr[0], int) and isinstance(pr[1], int) for pr in pathrows)
    assert len(pathrows) > 0  # Ensure the file is not empty
