"""This module contains various parsers for converting data.

These are primarily useful as convenience functions when using the typer module,
particularly in combination with argo workflows, which only allows certain
parameter types.

Example usage might be something like::

    from typing import Annotated
    import typer

    def main(flag: Annotated[str, typer.Option(parser=bool_parser)] = "True"):
        print(flag.__class__)

    if __name__ == "__main__":
        typer.run(main())

Calling this code like::

    $ python file.py --flag True

or::

    $ python file.py

will produce::

    <class 'bool'>

That is, the input string argument has automatically been converted to a bool.

the parameter must have a name and a value.
"""


def datetime_parser(datetime: str) -> list[str]:
    """Parse a string representing a year or multiple years.

    Args:
        datetime: A string of the format `'<year>'` or `'<year x>_<year y>'`.
            If the latter, the years must be in order.

    Returns:
        If `datetime` is a single year, it is returned as a single item list.
        Otherwise a generator producing integer values in the range
        `[<year x>, <year y> + 1]` is returned.
    """
    years = datetime.split("_")
    if len(years) == 2:
        years = [str(year) for year in range(int(years[0]), int(years[1]) + 1)]
    elif len(years) > 2:
        ValueError(f"{datetime} is not a valid value for --datetime")
    return years


def bool_parser(raw: str) -> bool:
    """Parse a string representing a true or false condition.

    Args:
        raw: Either "False" or something else.

    Returns:
        False if `raw` is `"False"'` and true otherwise.
    """
    return False if raw == "False" else True
