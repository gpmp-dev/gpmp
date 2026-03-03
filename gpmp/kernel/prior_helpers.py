# gpmp/kernel/prior_helpers.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Helpers for prior-informed parameter selection procedures.
"""

import warnings

import gpmp.num as gnp

from .init import anisotropic_parameters_initial_guess
from .prior_defaults import (
    get_default_prior_hyperparameters,
    resolve_prior_defaults_for_selection,
)


def _minimum_nonzero_gap_distance_1d(xj):
    """Smallest positive spacing among points in 1D (inf if none)."""
    xj = gnp.asarray(xj).reshape(-1)
    if xj.shape[0] < 2:
        return gnp.inf
    xs = gnp.sort(xj)
    diffs = gnp.diff(xs)
    diffs = diffs[diffs > 0.0]
    return gnp.min(diffs) if diffs.shape[0] > 0 else gnp.inf


def _componentwise_logrho_min_from_xi(xi):
    """
    Compute componentwise logrho lower bounds from gaps and component ranges.

    Parameters
    ----------
    xi : array_like of shape (n, d)
        Observation points.

    Returns
    -------
    logrho_min_from_gap : array_like of shape (d,)
        Componentwise ``log(min_nonzero_gap)``; ``-inf`` where no finite gap exists.
    x_range : array_like of shape (d,)
        Componentwise range ``max(x[:, j]) - min(x[:, j])``.
    """
    xi = gnp.asarray(xi)
    _n, d = xi.shape
    vals = []
    ranges = []
    for j in range(d):
        xj = xi[:, j]
        min_gap = _minimum_nonzero_gap_distance_1d(xi[:, j])
        vals.append(gnp.log(min_gap) if gnp.isfinite(min_gap) else -gnp.inf)
        ranges.append(gnp.max(xj) - gnp.min(xj))
    return gnp.asarray(vals), gnp.asarray(ranges)


def compute_logrho_min_from_xi(xi, prior_rho_min_range_factor=None):
    """
    Compute safeguarded componentwise ``prior_logrho_min`` from observation points.

    The bound combines two componentwise lower bounds and keeps the tightest (largest)
    admissible one:

    1. ``log(min nonzero gap)``
    2. ``log(range * prior_rho_min_range_factor)``

    Parameters
    ----------
    xi : array_like of shape (n, d)
        Observation points.
    prior_rho_min_range_factor : float, optional
        Safeguard factor for the range-based lower bound.
        If None, the default configured in ``gpmp.kernel.prior_defaults`` is used.

    Returns
    -------
    prior_logrho_min : array_like of shape (d,)
        Safeguarded componentwise lower bound for ``logrho``.
    """
    if prior_rho_min_range_factor is None:
        prior_rho_min_range_factor = get_default_prior_hyperparameters(xi)[
            "rho_min_range_factor"
        ]
    if prior_rho_min_range_factor <= 0:
        raise ValueError("prior_rho_min_range_factor must be strictly positive.")
    logrho_min_gap, x_range = _componentwise_logrho_min_from_xi(xi)
    min_rho_from_range = x_range * float(prior_rho_min_range_factor)
    positive_mask = min_rho_from_range > 0.0
    min_rho_safe = gnp.where(positive_mask, min_rho_from_range, 1.0)
    logrho_min_range = gnp.where(positive_mask, gnp.log(min_rho_safe), -gnp.inf)
    return gnp.maximum(logrho_min_gap, logrho_min_range)


def resolve_covparam0_prior_and_init(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    *,
    covparam0=None,
    covparam0_prior=None,
    covparam0_init=None,
):
    """
    Resolve prior anchor and optimization start for covariance parameters.

    Parameters
    ----------
    covparam0 : array_like, optional
        Shared fallback covariance parameters used when either
        ``covparam0_prior`` or ``covparam0_init`` is not provided.
        If all three are None, an anisotropic initial guess is computed once.
    covparam0_prior : array_like, optional
        Prior-anchor covariance parameters.
    covparam0_init : array_like, optional
        Initial covariance parameters for the optimizer.

    Returns
    -------
    covparam0_prior : ndarray
        Covariance parameters used as prior anchor.
    covparam0_init : ndarray
        Covariance parameters used as optimizer start.
    """
    covparam_initial_guess = None
    if covparam0_init is None:
        if covparam0 is not None:
            covparam0_init = covparam0
        else:
            covparam_initial_guess = anisotropic_parameters_initial_guess(
                model, xi, zi, dataloader
            )
            covparam0_init = covparam_initial_guess

    if covparam0_prior is None:
        if covparam0 is not None:
            covparam0_prior = covparam0
        elif covparam_initial_guess is not None:
            covparam0_prior = covparam_initial_guess
        else:
            covparam0_prior = anisotropic_parameters_initial_guess(
                model, xi, zi, dataloader
            )

    return covparam0_prior, covparam0_init


def resolve_covparam0_roles_for_update(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    *,
    covparam0=None,
    covparam0_prior=None,
    covparam0_init=None,
    warn_covparam0_prior=True,
):
    """
    Resolve ``covparam0_prior`` and ``covparam0_init`` for update procedures.

    Resolution policy
    -----------------
    1. ``covparam0_prior``:
       - explicit ``covparam0_prior`` if provided;
       - else ``covparam0`` if provided (with warning);
       - else ``model.covparam`` if available (with warning);
       - else anisotropic initial guess.
    2. ``covparam0_init``:
       - explicit ``covparam0_init`` if provided;
       - else ``covparam0`` if provided;
       - else ``model.covparam`` if available;
       - else anisotropic initial guess.
    """
    covparam_initial_guess = None
    if covparam0_init is None:
        if covparam0 is not None:
            covparam0_init = covparam0
        elif model.covparam is not None:
            covparam0_init = model.covparam
        else:
            covparam_initial_guess = anisotropic_parameters_initial_guess(
                model, xi, zi, dataloader
            )
            covparam0_init = covparam_initial_guess

    if covparam0_prior is None:
        if covparam0 is not None:
            if warn_covparam0_prior:
                warnings.warn(
                    "covparam0 provided without covparam0_prior in update procedure; "
                    "using covparam0 as covparam0_prior. "
                    "Pass covparam0_prior explicitly to avoid this coupling.",
                    stacklevel=2,
                )
            covparam0_prior = covparam0
        elif model.covparam is not None:
            if warn_covparam0_prior:
                warnings.warn(
                    "covparam0 and covparam0_prior not provided in update procedure; "
                    "using model.covparam as covparam0_prior. "
                    "Pass covparam0_prior explicitly to avoid this coupling.",
                    stacklevel=2,
                )
            covparam0_prior = model.covparam
        elif covparam_initial_guess is not None:
            covparam0_prior = covparam_initial_guess
        else:
            covparam0_prior = anisotropic_parameters_initial_guess(
                model, xi, zi, dataloader
            )

    return covparam0_prior, covparam0_init


def resolve_logsigma2_logrho_prior_args(
    *,
    covparam0_prior,
    xi=None,
    dataloader=None,
    prior_gamma=None,
    prior_sigma2_coverage=None,
    prior_alpha=None,
    prior_rho_min_range_factor=None,
    prior_log_sigma2_0=None,
    prior_logrho_0=None,
    prior_logrho_min=None,
):
    """
    Resolve defaults and reference values for gaussian-logsigma2 + logrho prior.

    Returns
    -------
    tuple
        ``(prior_gamma, prior_sigma2_coverage, prior_alpha,
        prior_rho_min_range_factor, prior_log_sigma2_0, prior_logrho_0,
        prior_logrho_min)``.
    """
    prior_gamma, prior_sigma2_coverage, prior_alpha, prior_rho_min_range_factor = (
        resolve_prior_defaults_for_selection(
            xi=xi,
            dataloader=dataloader,
            gamma=prior_gamma,
            sigma2_coverage=prior_sigma2_coverage,
            alpha=prior_alpha,
            rho_min_range_factor=prior_rho_min_range_factor,
        )
    )

    prior_log_sigma2_0 = (
        covparam0_prior[0] if prior_log_sigma2_0 is None else prior_log_sigma2_0
    )
    prior_logrho_0 = -covparam0_prior[1:] if prior_logrho_0 is None else prior_logrho_0
    prior_logrho_0 = gnp.asarray(prior_logrho_0)

    if prior_logrho_min is None:
        if xi is not None:
            xi_for_min = xi
        elif dataloader is not None and hasattr(dataloader, "dataset"):
            ds = dataloader.dataset
            if hasattr(ds, "x_list"):
                xi_for_min = (
                    gnp.concatenate(ds.x_list, axis=0)
                    if isinstance(ds.x_list, list)
                    else ds.x_list
                )
            else:
                raise ValueError(
                    "dataloader.dataset must provide x_list when prior_logrho_min is None."
                )
        else:
            raise ValueError(
                "xi or dataloader.dataset.x_list must be provided when prior_logrho_min is None."
            )
        prior_logrho_min = compute_logrho_min_from_xi(
            xi_for_min, prior_rho_min_range_factor=prior_rho_min_range_factor
        )
    prior_logrho_min = gnp.asarray(prior_logrho_min)

    return (
        prior_gamma,
        prior_sigma2_coverage,
        prior_alpha,
        prior_rho_min_range_factor,
        prior_log_sigma2_0,
        prior_logrho_0,
        prior_logrho_min,
    )
