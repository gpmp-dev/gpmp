# gpmp/kernel/prior_defaults.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Default hyperparameters for kernel prior models.

Functions
---------
get_default_prior_hyperparameters
    Return the current prior-default hyperparameters.
set_default_prior_hyperparameters
    Update one or more prior-default hyperparameters.
set_default_prior_hyperparameters_from_kwargs
    Update prior defaults from ``kwargs`` in place (recognized keys are popped).
"""

from dataclasses import dataclass


@dataclass
class _PriorDefaults:
    gamma: float = 1.5
    sigma2_coverage: float = 0.95
    alpha: float = 1.0
    rho_min_range_factor: float = 1 / 20.0


_PRIOR_DEFAULTS = _PriorDefaults()


def _validate_xi_shape(xi):
    if xi is not None and hasattr(xi, "shape"):
        shape = tuple(int(s) for s in xi.shape)
        if len(shape) != 2:
            raise ValueError("xi must have shape (n, d).")


def get_default_prior_hyperparameters(xi=None):
    """
    Return default prior hyperparameters.

    Parameters
    ----------
    xi : array_like, optional
        Observation points. If provided, must have shape ``(n, d)``.
        Current defaults are dataset-agnostic; this argument is reserved for
        future dataset-conditioned policies.

    Returns
    -------
    dict
        Dictionary with keys ``gamma``, ``sigma2_coverage``, ``alpha``,
        ``rho_min_range_factor``.
    """
    _validate_xi_shape(xi)
    return {
        "gamma": _PRIOR_DEFAULTS.gamma,
        "sigma2_coverage": _PRIOR_DEFAULTS.sigma2_coverage,
        "alpha": _PRIOR_DEFAULTS.alpha,
        "rho_min_range_factor": _PRIOR_DEFAULTS.rho_min_range_factor,
    }


def set_default_prior_hyperparameters(
    *,
    gamma=None,
    sigma2_coverage=None,
    alpha=None,
    rho_min_range_factor=None,
):
    """
    Update one or more default prior hyperparameters.

    Parameters
    ----------
    gamma : float, optional
        Multiplicative factor for ``sigma2`` prior calibration. Must be > 1.
    sigma2_coverage : float, optional
        Central coverage used for log-variance Gaussian prior calibration.
        Must be in ``(0, 1)``.
    alpha : float, optional
        Linear tail slope parameter for ``logrho`` prior. Must be > 0.
    rho_min_range_factor : float, optional
        Range-based safeguard factor for ``logrho_min`` construction.
        Must be > 0.
    """
    if gamma is not None:
        gamma = float(gamma)
        if gamma <= 1.0:
            raise ValueError("gamma must be > 1.")
        _PRIOR_DEFAULTS.gamma = gamma

    if sigma2_coverage is not None:
        sigma2_coverage = float(sigma2_coverage)
        if not (0.0 < sigma2_coverage < 1.0):
            raise ValueError("sigma2_coverage must be in (0, 1).")
        _PRIOR_DEFAULTS.sigma2_coverage = sigma2_coverage

    if alpha is not None:
        alpha = float(alpha)
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0.")
        _PRIOR_DEFAULTS.alpha = alpha

    if rho_min_range_factor is not None:
        rho_min_range_factor = float(rho_min_range_factor)
        if rho_min_range_factor <= 0.0:
            raise ValueError("rho_min_range_factor must be > 0.")
        _PRIOR_DEFAULTS.rho_min_range_factor = rho_min_range_factor


def set_default_prior_hyperparameters_from_kwargs(kwargs):
    """
    Update prior defaults from ``kwargs`` in place.

    Recognized keys are:
    ``prior_logsigma2_gamma``, ``prior_logsigma2_coverage``,
    ``prior_logrho_alpha``, ``prior_logrho_min_range_factor``.
    Recognized keys are removed from ``kwargs``.
    """
    if "prior_logsigma2_gamma" in kwargs:
        set_default_prior_hyperparameters(gamma=kwargs.pop("prior_logsigma2_gamma"))
    if "prior_logsigma2_coverage" in kwargs:
        set_default_prior_hyperparameters(
            sigma2_coverage=kwargs.pop("prior_logsigma2_coverage")
        )
    if "prior_logrho_alpha" in kwargs:
        set_default_prior_hyperparameters(alpha=kwargs.pop("prior_logrho_alpha"))
    if "prior_logrho_min_range_factor" in kwargs:
        set_default_prior_hyperparameters(
            rho_min_range_factor=kwargs.pop("prior_logrho_min_range_factor")
        )


def resolve_prior_defaults_for_selection(
    xi=None,
    dataloader=None,
    gamma=None,
    sigma2_coverage=None,
    alpha=None,
    rho_min_range_factor=None,
):
    """
    Resolve prior defaults from configured values using available observation points.
    """
    xi_for_defaults = xi
    if (
        xi_for_defaults is None
        and dataloader is not None
        and hasattr(dataloader, "dataset")
    ):
        ds = dataloader.dataset
        if hasattr(ds, "x_list"):
            # gnp is intentionally imported lazily here to keep this module lightweight.
            import gpmp.num as gnp  # local import

            xi_for_defaults = (
                gnp.concatenate(ds.x_list, axis=0)
                if isinstance(ds.x_list, list)
                else ds.x_list
            )

    defaults = get_default_prior_hyperparameters(xi_for_defaults)
    if gamma is None:
        gamma = defaults["gamma"]
    if sigma2_coverage is None:
        sigma2_coverage = defaults["sigma2_coverage"]
    if alpha is None:
        alpha = defaults["alpha"]
    if rho_min_range_factor is None:
        rho_min_range_factor = defaults["rho_min_range_factor"]
    return gamma, sigma2_coverage, alpha, rho_min_range_factor
