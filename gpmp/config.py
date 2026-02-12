"""
config.py — Global configuration for GPmp.

This module defines and stores a process-wide configuration object for GPmp.

Responsibilities
----------------
- Store runtime configuration (backend, dtype, device, seed, caches, logger).
- Detect and initialize the numerical backend before importing gpmp.num.
- Expose a small API to query/update configuration values.

Backend selection
-----------------
The backend is determined in the following order:
1) Explicit environment variable GPMP_BACKEND (accepted: 'torch', 'numpy').
2) Automatic detection: use 'torch' if available, otherwise fallback to 'numpy'.

The selected backend is stored in GPMP_BACKEND to make downstream imports
consistent within the process.

Dtype handling
--------------
GPmp uses a portable dtype specification stored in GPMP_DTYPE ('float32' or
'float64'). This is resolved inside gpmp.num at import time into a backend-native
dtype object stored in `config.dtype_resolved`.

Notes
-----
- Functions that change backend or dtype must be called before importing gpmp.num
  to take effect for that process.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022–2026, CentraleSupélec
License: GPLv3 (see LICENSE)

"""

import os
import logging
from importlib.util import find_spec

# Read version from VERSION file
_version_file = os.path.join(os.path.dirname(__file__), "..", "VERSION")
try:
    with open(os.path.abspath(_version_file), "r") as f:
        __version__ = f.read().strip()
except FileNotFoundError:
    __version__ = "0.0.0"


def _normalize_dtype_spec(dtype) -> str:
    """
    Normalize a dtype specification to 'float32' or 'float64'.

    Accepts strings ('float32', 'float64', 'torch.float32', 'np.float64', ...),
    python float, numpy/torch dtype objects (by string form), etc.
    """
    if dtype is None:
        return "float64"
    if dtype is float:
        return "float64"

    s = dtype.lower() if isinstance(dtype, str) else str(dtype).lower()
    if "float32" in s or s.endswith("f4") or s.endswith("32"):
        return "float32"
    if "float64" in s or "double" in s or s.endswith("f8") or s.endswith("64"):
        return "float64"
    raise ValueError("dtype must resolve to float32 or float64")


class _GPMPConfig:
    def __init__(self):
        self.version = __version__
        self.backend = None
        # Portable dtype spec (resolved inside gpmp.num).
        self.dtype = _normalize_dtype_spec(os.environ.get("GPMP_DTYPE", "float64"))
        # Backend-native dtype object resolved at gpmp.num import time.
        self.dtype_resolved = None
        self.device = "cpu"
        self.seed = 1234
        self.caches = {}
        # logger lives in config
        self.logger = logging.getLogger("gpmp")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

    def __str__(self):
        return (
            f"GPMPConfig("
            f"version={self.version}, "
            f"backend={self.backend}, "
            f"dtype={self.dtype}, "
            f"dtype_resolved={self.dtype_resolved}, "
            f"device={self.device}, "
            f"seed={self.seed}, "
            f"caches={list(self.caches.keys())})"
        )

    def __repr__(self):
        return (
            f"<GPMPConfig "
            f"version={self.version!r}, "
            f"backend={self.backend!r}, "
            f"dtype={self.dtype!r}, "
            f"dtype_resolved={self.dtype_resolved!r}, "
            f"device={self.device!r}, "
            f"seed={self.seed!r}, "
            f"caches={list(self.caches.keys())}>"
        )

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def clear_caches(self, name=None):
        if name is None:
            self.caches.clear()
        else:
            self.caches.pop(name, None)


_config = _GPMPConfig()


def get_config():
    return _config


def _detect_backend():
    env = os.environ.get("GPMP_BACKEND")
    if env == "jax":
        raise RuntimeError(
            "The 'jax' backend is no longer supported. "
            "Please set GPMP_BACKEND to 'numpy' or 'torch'."
        )
    if env in ("numpy", "torch"):
        return env
    if find_spec("torch") is not None:
        return "torch"
    return "numpy"


def init_backend():
    """Idempotent. Detect and store backend, set env for downstream imports."""
    if _config.backend is None:
        backend = _detect_backend()
        _config.backend = backend
        os.environ["GPMP_BACKEND"] = backend
    return _config.backend


def set_backend(backend: str):
    """Force a backend ('numpy'|'torch') before importing gpmp.num."""
    if backend == "jax":
        raise ValueError("backend must be 'numpy' or 'torch' (jax is not supported)")
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be 'numpy' or 'torch'")
    _config.backend = backend
    os.environ["GPMP_BACKEND"] = backend


def get_backend():
    """Return current backend; triggers detection if not set."""
    return _config.backend or init_backend()


def set_dtype(dtype):
    """
    Set portable dtype spec ('float32' or 'float64').

    Must be called before importing gpmp.num to take effect for that process.
    """
    spec = _normalize_dtype_spec(dtype)
    _config.dtype = spec
    _config.dtype_resolved = None
    os.environ["GPMP_DTYPE"] = spec


def set_device(device):
    _config.device = device


def clear_caches(name=None):
    _config.clear_caches(name)


def get_logger():
    return _config.logger


def set_log_level(level):
    _config.logger.setLevel(level)
