# gpmp/num/__init__.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""Numerical backend dispatcher for GPmp."""

from gpmp.config import init_backend

from . import shared as _shared

_gpmp_backend_ = init_backend()

if _gpmp_backend_ == "numpy":
    from . import numpy_backend as _backend
elif _gpmp_backend_ == "torch":
    from . import torch_backend as _backend
else:
    raise RuntimeError(
        "Please set the GPMP_BACKEND environment variable to 'numpy' or 'torch'."
    )

# Re-export backend API.
for _name in dir(_backend):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_backend, _name)

# Re-export backend-independent helpers from shared.py.
get_dtype = _shared.get_dtype
compute_gammaln = _shared.compute_gammaln
derivative_finite_diff = _shared.derivative_finite_diff
try_with_postmortem = _shared.try_with_postmortem
