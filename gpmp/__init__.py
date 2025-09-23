# gpmp/__init__.py

from . import config
from . import num
from . import core
from . import kernel
from . import dataloader
from . import misc
from .core import Model
import os

__all__ = ["num", "kernel", "Model", "__version__"]

# Read version from VERSION file at project root
_version_file = os.path.join(os.path.dirname(__file__), "..", "VERSION")
try:
    with open(_version_file, "r") as f:
        __version__ = f.read().strip()
except FileNotFoundError:
    __version__ = "0.0.0"
