"""Locations of the result files (``data/``) and of the regenerated figures (``figures/``).

The repository root is, in this order, the directory named by the environment variable ``QMSC_ROOT``, the
parent of the package when it contains ``data/`` (an editable install or the source tree on the path), or
the first directory at or above the current working directory that contains both ``data/`` and
``scripts/``.
"""
import os


def _find_root():
    env = os.environ.get("QMSC_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isdir(os.path.join(here, "data")):
        return here
    d = os.getcwd()
    while True:
        if os.path.isdir(os.path.join(d, "data")) and os.path.isdir(os.path.join(d, "scripts")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return here
        d = parent


ROOT = _find_root()
DATA_DIR = os.path.join(ROOT, "data")
FIGURES_DIR = os.path.join(ROOT, "figures")
