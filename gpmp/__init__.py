# gpmp/__init__.py
"""
GPmp package.

Defines
-------
Model
__version__

Notes
-----
gpmp.config is imported eagerly to initialize configuration and backend selection.
Other subpackages are loaded lazily on first access.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

from __future__ import annotations

import importlib
import os
from typing import Final

from . import config as config  # eager
from .core import Model

__all__ = [
    "Model",
    "__version__",
    "config",
    "num",
    "kernel",
    "dataloader",
    "modeldiagnosis",
    "misc",
    "plot",
]

_DEFAULT_VERSION: Final[str] = "0.0.0"
_LAZY_SUBMODULES: Final[set[str]] = {
    "num",
    "kernel",
    "dataloader",
    "modeldiagnosis",
    "misc",
    "plot",
}


def _read_version() -> str:
    pkg_dir = os.path.dirname(__file__)
    version_file = os.path.join(os.path.dirname(pkg_dir), "VERSION")
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            v = f.read().strip()
        return v if v else _DEFAULT_VERSION
    except OSError:
        return _DEFAULT_VERSION


__version__ = _read_version()


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | _LAZY_SUBMODULES)
