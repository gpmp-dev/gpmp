# gpmp/modeldiagnosis/__init__.py
"""
Model diagnosis utilities for GPmp.

Defines
-------
This package groups helpers for:
- parameter statistics on selection criteria
- predictive performance metrics
- report construction and display
- plotting helpers
- small utilities for printing and data description

Public API
----------
The most common entry points are re-exported at package level.
Importing `gpmp.modeldiagnosis` does not import matplotlib. Plotting
functions are imported lazily via the `plotting` submodule.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

from __future__ import annotations

from .distributions import Unnormalized1DDistribution
from .param_stats import (
    fast_univariate_stats,
    make_single_param_criterion_function,
    selection_criterion_statistics,
    selection_criterion_statistics_fast,
)
from .performance import compute_performance, perf
from .report import diag, model_diagnosis_disp, modeldiagnosis_init
from .utils import describe_array, pretty_print_dictionnary, pretty_print_dictionary, sigma_rho_from_covparam

__all__ = [
    "Unnormalized1DDistribution",
    "fast_univariate_stats",
    "make_single_param_criterion_function",
    "selection_criterion_statistics",
    "selection_criterion_statistics_fast",
    "compute_performance",
    "perf",
    "diag",
    "modeldiagnosis_init",
    "model_diagnosis_disp",
    "sigma_rho_from_covparam",
    "describe_array",
    "pretty_print_dictionary",
    "pretty_print_dictionnary",
]

# Lazy access to plotting functions to avoid importing matplotlib on import.
_PLOTTING_EXPORTS = {
    "plot_pit_ecdf",
    "plot_selection_criterion_crosssections",
    "plot_selection_criterion_2d",
    "plot_selection_criterion_sigma_rho",
}


def __getattr__(name: str):
    if name in _PLOTTING_EXPORTS:
        from . import plotting as _plotting

        obj = getattr(_plotting, name)
        globals()[name] = obj  # cache for subsequent lookups
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(__all__) + list(_PLOTTING_EXPORTS))
