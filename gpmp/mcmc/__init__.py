# gpmp/mcmc/__init__.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Markov chain Monte Carlo (MCMC) and particle-based samplers for GPmp.

This subpackage gathers:
- Metropolis-Hastings tools
- NUTS sampling
- Sequential Monte Carlo (SMC) utilities

Public API
----------
MHOptions, MetropolisHastings
    Metropolis-Hastings configuration and sampler.
nuts_sample, nuts_transition
    NUTS transition kernel and sampling driver.
NUTSOptions
    Configuration for NUTS warmup and adaptation policies.
ParticlesSetConfig, SMCConfig, ParticlesSet, SMC
    SMC configuration and core classes.
run_smc_sampling, run_subset_simulation
    High-level SMC/subset-simulation entry points.
sample_from_selection_criterion_mh, sample_from_selection_criterion_nuts, sample_from_selection_criterion_smc
    Posterior parameter sampling helpers.
"""
from __future__ import annotations

import importlib

__all__ = [
    "MHOptions",
    "MetropolisHastings",
    "sample_multivariate_normal_with_jitter",
    "nuts_sample",
    "nuts_transition",
    "NUTSOptions",
    "plot_nuts_diagnostics",
    "ParticlesSetConfig",
    "SMCConfig",
    "ParticlesSet",
    "SMC",
    "run_smc_sampling",
    "log_indicator_density",
    "run_subset_simulation",
    "sample_from_selection_criterion_mh",
    "sample_from_selection_criterion_nuts",
    "sample_from_selection_criterion_smc",
]

_EXPORT_TO_MODULE = {
    # Metropolis-Hastings
    "MHOptions": "mh",
    "MetropolisHastings": "mh",
    "sample_multivariate_normal_with_jitter": "mh",
    # NUTS
    "nuts_sample": "nuts",
    "nuts_transition": "nuts",
    "NUTSOptions": "nuts",
    "plot_nuts_diagnostics": "nuts",
    # SMC
    "ParticlesSetConfig": "smc",
    "SMCConfig": "smc",
    "ParticlesSet": "smc",
    "SMC": "smc",
    "run_smc_sampling": "smc",
    "log_indicator_density": "smc",
    "run_subset_simulation": "smc",
    # Posterior parameter sampling
    "sample_from_selection_criterion_mh": "param_posterior",
    "sample_from_selection_criterion_nuts": "param_posterior",
    "sample_from_selection_criterion_smc": "param_posterior",
}


def __getattr__(name: str):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{module_name}")
    obj = getattr(module, name)
    globals()[name] = obj
    return obj


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
