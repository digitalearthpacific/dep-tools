class EmptyCollectionError(Exception):
    """Indicates an empty collection.

    Typically raised when a STAC search returns an empty
    :py:class:`pystac.ItemCollection`.
    """

    pass


class NoOutputError(Exception):
    """Raised when something does not produce any output."""

    pass
