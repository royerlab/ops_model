"""Central base paths for ops_model data/model/analysis roots.

Every storage location in this package derives from ``BASE_PATH``, which must be
set via the ``OPS_BASE_PATH`` env var. There is no default: the correct root
depends on where you have staged the dataset, so an unset value is an error
rather than a silent fallback. See the README ("Data availability") for how to
obtain the data.
"""
import os
from pathlib import Path

_BASE_PATH_ENV = "OPS_BASE_PATH"

BASE_PATH = os.environ.get(_BASE_PATH_ENV)
if not BASE_PATH:
    raise RuntimeError(
        f"{_BASE_PATH_ENV} is not set. Point it at the root of your OPS data tree, "
        f'e.g. `export {_BASE_PATH_ENV}="/path/to/ops_data"`. '
        'See the README section "Data availability" for how to obtain the dataset.'
    )
