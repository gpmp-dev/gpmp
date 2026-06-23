# gpmp/parameter/__init__.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Structured parameter objects for GPmp.

This package is a helper layer for naming, normalizing, and displaying
parameter vectors. The lower-level :mod:`gpmp.core` and :mod:`gpmp.kernel`
APIs operate on plain arrays and do not depend on ``gpmp.parameter``.
"""

from .param import (
    Normalization,
    Param,
    make_anisotropic_param,
    param_from_covparam_anisotropic,
    param_from_covparam_anisotropic_noisy,
)

__all__ = [
    "Normalization",
    "Param",
    "make_anisotropic_param",
    "param_from_covparam_anisotropic",
    "param_from_covparam_anisotropic_noisy",
]
