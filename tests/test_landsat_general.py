from dep_tools.landsat_utils import read_pathrows_file


def test_read_pathrows_file():
    """Test reading pathrows from the file."""
    pathrows = read_pathrows_file()
    assert isinstance(pathrows, list)
    assert all(isinstance(pr, tuple) and len(pr) == 2 for pr in pathrows)
    assert all(isinstance(pr[0], int) and isinstance(pr[1], int) for pr in pathrows)
    assert len(pathrows) > 0  # Ensure the file is not empty
