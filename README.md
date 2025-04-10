# dep-tools python package

This repository hosts a python package containing a set of tools to help load,
process, write, and keep track of geospatial products created as part of the
Digital Earth Pacific project. It was conceived to provide a framework to aid
scaling products across the Pacific.

## Installation

## What is here

### Grid definition

[dep_tools/grids.py](dep_tools/grids.py) contains definitions of commonly used
tiling schemes for DE Pacific products at 10 or 30 meter resolution.

### Utility functions

There are general and specific functions that can be applied
primarily to Xarray datasets and arrays related to Azure (dep_tools/azure.py),
Landsat (dep_tools/landsat_utils.py), Sentinel-2 (dep_tools/s2_utils.py) and
spatial temporal asset catalogs (i.e. STAC, dep_tools./stac_utils.py).

### Naming

[dep_tools/namers.py](dep_tools/namers.py) contains the definition
implementation to represent the local or remote (network) paths for dep products
for given arguments such as satellite sensor, dataset name, version, date, etc.

### Task framework

The DEP tools task framework is an attempt to break down the typical product
workflow into a series of common steps. It is an experimental approach whose
main goals are to avoid code redundancy and make processing steps more easily
understanding and modifiable.

The task framework consists of five primary classes:

- [Searcher](dep_tools/searchers.py): A Searcher searches for data for a given
  area or location id, typically via a STAC endpoint.
- [Loader](dep_tools/loaders.py): A Loader loads data for a given area, or other
  identifier.
- [Processor](dep_tools/processors.py): A Processor processes data to produce a
  given output. This is typically the portion of the workflow that needs
  implementation for a given product.
- [Writer](dep_tools/writers.py): A Writer takes output data and writes it
  somewhere.
- [Task](dep_tools/task.py): A Task orchestrates the steps above.
