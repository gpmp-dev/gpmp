# gpmp/config.py
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

class _GPMPConfig:
    def __init__(self):
        self.version = __version__
        self.backend = None
        self.dtype = float
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
    if env in ("numpy", "torch", "jax"):
        return env
    if find_spec("torch") is not None:
        return "torch"
    if find_spec("jax") is not None:
        return "jax"
    return "numpy"

def init_backend():
    """Idempotent. Detect and store backend, set env for downstream imports."""
    if _config.backend is None:
        backend = _detect_backend()
        _config.backend = backend
        os.environ["GPMP_BACKEND"] = backend
    return _config.backend

def set_backend(backend: str):
    """Force a backend ('numpy'|'torch'|'jax') before importing gpmp.num."""
    if backend not in ("numpy", "torch", "jax"):
        raise ValueError("backend must be 'numpy', 'torch', or 'jax'")
    _config.backend = backend
    os.environ["GPMP_BACKEND"] = backend

def get_backend():
    """Return current backend; triggers detection if not set."""
    return _config.backend or init_backend()

def set_dtype(dtype):
    _config.dtype = dtype  # torch default dtype is set from gpmp.num after backend selection

def set_device(device):
    _config.device = device

def clear_caches(name=None):
    _config.clear_caches(name)

def get_logger():
    return _config.logger

def set_log_level(level):
    _config.logger.setLevel(level)
