"""This module contains various parsers, or functions that data in one format and convert it to another. These are primarily useful as convenience functions when using the typer module, particularly in combination with argo workflows.

example usage might be something like

```
from typing import Annotated
ltyper

def main(flag: Annotated[str, typer.Option(parser=bool_parser)] = "True"):
    print(flag.__class__)

if __name__ == "__main__":
    typer.run(main())
```

Calling this code like

$ python file.py --flag True
or
$ python file.py

will produce

<class 'bool'>

That is, the input string argument has automatically been converted to a bool.

Why not just define the type as a bool? Argo can't have flag options. So
the parameter must have a name and a value.
"""


def datetime_parser(datetime: str):
    """Parse a string in the format <year> or <year 1>_<year 2>. If a
    single year, the input is returned. Otherwise a generator producing integer
    values in the range [year1, year2 + 1] is returned.
    """
    years = datetime.split("_")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{datetime} is not a valid value for --datetime")
    return years


def bool_parser(raw: str) -> bool:
    """Convert the input string into a boolean, which is False if the input is
    "False" and true otherwise.
    """
    return False if raw == "False" else True
