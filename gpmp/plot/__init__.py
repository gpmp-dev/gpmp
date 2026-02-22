# gpmp/plot/__init__.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
GPmp plotting utilities.
"""

from .plotutils import Figure, crosssections, plot_loo

__all__ = ["Figure", "crosssections", "plot_loo", "plotutils"]

# Keep plotutils module accessible for backward compatibility
from . import plotutils
