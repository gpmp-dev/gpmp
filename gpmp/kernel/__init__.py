# gpmp/kernel/__init__.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Gaussian Process kernels and related utilities.

This subpackage provides covariance functions and parameter
initialization tools for Gaussian Process (GP) modeling.

Modules
-------
exponential
    Exponential kernel.
matern
    Matérn family of kernels with half-integer regularity.
bounds
    Empirical parameter bounds (variance, length scales).
init
    Initialization heuristics for covariance parameters.
parameter_selection
    Parameter optimization (ML, REML, REMAP) and selection criteria.
priors
    Log-prior formulations (Jeffreys, power-law, reference).
utils
    Internal helper functions for data preparation and validation.

Public API
-----------
- Exponential kernel:
    exponential_kernel
- Matérn kernels:
    matern32_kernel, maternp_kernel, maternp_covariance
- Utilities for parameter selection:
    anisotropic_parameters_initial_guess
    select_parameters_with_reml
    select_parameters_with_remap
"""

from .exponential import exponential_kernel
from .matern import matern32_kernel, maternp_kernel, maternp_covariance

# Optional imports from kernel tools (keep lazy import style for heavy ones)
from .init import (
    anisotropic_parameters_initial_guess,
    anisotropic_parameters_initial_guess_constant_mean,
    anisotropic_parameters_initial_guess_zero_mean,
)
from .parameter_selection import (
    negative_log_likelihood_zero_mean,
    negative_log_likelihood,
    negative_log_restricted_likelihood,
    make_selection_criterion_with_gradient,
    autoselect_parameters,
    select_parameters_with_criterion,
    update_parameters_with_criterion,
    select_parameters_with_reml,
    update_parameters_with_reml,
    select_parameters_with_remap,
    update_parameters_with_remap,
)
from .priors import (
    log_prior_jeffrey_variance,
    log_prior_power_law,
    log_prior_reference,
    neg_log_restricted_posterior_with_power_law_prior,
)

__all__ = [
    # Kernels
    "exponential_kernel",
    "matern32_kernel",
    "maternp_kernel",
    "maternp_covariance",
    # Initialization
    "anisotropic_parameters_initial_guess",
    "anisotropic_parameters_initial_guess_constant_mean",
    "anisotropic_parameters_initial_guess_zero_mean",
    # Parameter selection
    "negative_log_likelihood_zero_mean",
    "negative_log_likelihood",
    "negative_log_restricted_likelihood",
    "make_selection_criterion_with_gradient",
    "autoselect_parameters",
    "select_parameters_with_reml",
    "update_parameters_with_reml",
    "select_parameters_with_remap",
    "update_parameters_with_remap",
    # Priors
    "log_prior_jeffrey_variance",
    "log_prior_power_law",
    "log_prior_reference",
    "neg_log_restricted_posterior_with_power_law_prior",
]
