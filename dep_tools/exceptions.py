class EmptyCollectionError(Exception):
    """Raised when a STAC search returns and empty :py:class:`pystac.ItemCollection."""

    pass


class NoOutputError(Exception):
    """Raised when something does not produce any output."""

    pass
