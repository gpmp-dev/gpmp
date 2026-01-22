# gpmp/core/__init__.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------

"""
Core components of the gpmp package.

This subpackage contains the core numerical routines for Gaussian
Process modeling, including kriging predictors, LOO validation,
likelihood and Fisher information computations, sampling methods,
and supporting linear algebra utilities.

Public API
----------
Model : class
    Main Gaussian Process model façade combining all core routines.
"""

from .model import Model

__all__ = ["Model"]
