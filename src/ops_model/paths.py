"""Central base paths for ops_model data/model/analysis roots.

Every hardcoded storage location derives from ``BASE_PATH``; override the whole
tree with the ``OPS_BASE_PATH`` env var. The default is the current shared store.
"""
import os
from pathlib import Path

BASE_PATH = os.environ.get("OPS_BASE_PATH", "/hpc/projects/icd.fast.ops")
