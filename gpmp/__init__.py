# gpmp/__init__.py

from . import config
from . import num
from . import kernel
from . import dataloader
from . import misc
from .core import Model

import os

__all__ = ["num", "kernel", "Model", "__version__"]

# Read version from VERSION file at the project root
_pkg_dir = os.path.dirname(__file__)
_version_file = os.path.join(os.path.dirname(_pkg_dir), "VERSION")

try:
    with open(_version_file, "r", encoding="utf-8") as f:
        __version__ = f.read().strip()
except FileNotFoundError:
    __version__ = "0.0.0"
