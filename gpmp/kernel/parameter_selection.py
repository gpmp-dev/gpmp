# gpmp/kernel/parameter_selection.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Kernel parameter selection criteria and optimization helpers.
"""

import time
import numpy as np
from scipy.optimize import minimize
import gpmp.num as gnp

from .utils import check_xi_zi_or_loader
from .init import anisotropic_parameters_initial_guess
from .priors import (
    neg_log_restricted_posterior_power_laws_prior,
    neg_log_restricted_posterior_gaussian_logsigma2_prior,
    neg_log_restricted_posterior_gaussian_logsigma2_and_logrho_prior,
)


# ---------------------- criterion + gradient maker --------------------
def make_selection_criterion_with_gradient(
    model,
    selection_criterion,
    xi=None,
    zi=None,
    dataloader=None,
    batches_per_eval=0,
    parameterized_mean=False,
    meanparam_len=1,
):
    """
    Build criterion wrappers for value/gradient optimization and diagnostics.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance passed to ``selection_criterion``.
    selection_criterion : callable
        Criterion function.
        Expected signatures:
        - ``f(model, covparam, xi, zi)`` when ``parameterized_mean=False``
        - ``f(model, meanparam, covparam, xi, zi)`` when
          ``parameterized_mean=True``.
    xi, zi : array_like, optional
        Observation arrays used for criterion evaluation.
    dataloader : iterable, optional
        Batch loader used instead of ``xi, zi``.
        Batches must be yielded as ``(xb, zb)``.
    batches_per_eval : int, default=0
        Number of batches used per criterion call when ``dataloader`` is
        provided.
        - ``0``: iterate over the full loader each evaluation
        - ``>0``: use exactly that many batches per call (iterator cycles).
    parameterized_mean : bool, default=False
        Whether the criterion depends on explicit mean parameters.
    meanparam_len : int, default=1
        Number of leading parameters in the optimization vector assigned to the
        mean model.

    Returns
    -------
    evaluate : callable
        Value function with gradient-enabled behavior from backend wrapper.
    evaluate_pre_grad : callable
        Value function intended to be called just before ``gradient`` in
        optimization loops.
    evaluate_no_grad : callable
        Criterion evaluation function without gradient tracking.
    gradient : callable
        Gradient function with respect to optimization parameters.

    Notes
    -----
    Data source contract:
    exactly one of ``(xi, zi)`` or ``dataloader`` must be provided.

    Internally, this function wraps ``selection_criterion`` into an adapter
    accepting either:
    - covariance parameters only, or
    - concatenated ``[meanparam, covparam]`` parameters.

    For array data it uses ``gnp.DifferentiableSelectionCriterion``.
    For loader data it uses ``gnp.BatchDifferentiableSelectionCriterion``.

    The four returned callables are complementary:
    - ``evaluate`` and ``evaluate_pre_grad`` for optimizer value calls,
    - ``gradient`` for optimizer gradient calls,
    - ``evaluate_no_grad`` for diagnostics and sampling paths where gradients
      are not required.
    """
    data_source = check_xi_zi_or_loader(xi, zi, dataloader)

    if parameterized_mean:

        def crit_(param, xi, zi):
            meanparam = param[:meanparam_len]
            covparam = param[meanparam_len:]
            return selection_criterion(model, meanparam, covparam, xi, zi)

    else:

        def crit_(covparam, xi, zi):
            return selection_criterion(model, covparam, xi, zi)

    if data_source == "arrays":
        xi_ = gnp.asarray(xi)
        zi_ = gnp.asarray(zi)
        crit = gnp.DifferentiableSelectionCriterion(crit_, xi_, zi_)
    else:
        crit = gnp.BatchDifferentiableSelectionCriterion(
            crit_, dataloader, batches_per_eval=batches_per_eval
        )
    return crit.evaluate, crit.evaluate_pre_grad, crit.evaluate_no_grad, crit.gradient


# ------------------------------ optimizer -----------------------------
def autoselect_parameters(
    p0,
    criterion,
    gradient,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    silent=True,
    info=False,
    method="SLSQP",
    method_options=None,
):
    """
    Minimize a scalar selection criterion with SciPy.

    Parameters
    ----------
    p0 : array_like
        Initial parameter vector.
    criterion : callable
        Objective function ``criterion(p) -> scalar``.
    gradient : callable
        Gradient function ``gradient(p) -> array_like``.
    bounds : sequence of tuple, optional
        Bounds passed to SciPy in normalized parameter space.
    bounds_auto : bool, default=True
        If True and ``bounds`` is None, construct local bounds around ``p0``
        using ``bounds_delta`` and internal safety limits.
    bounds_delta : float, default=10.0
        Half-width used for automatic local bounds.
    silent : bool, default=True
        If False, enable solver output.
    info : bool, default=False
        If True, return the full SciPy result object.
    method : {"SLSQP", "L-BFGS-B"}, default="SLSQP"
        Optimization method.
    method_options : dict, optional
        Additional options passed to SciPy ``minimize``.

    Returns
    -------
    p_opt : array_like
        Best parameter vector found.
    info_ret : scipy.optimize.OptimizeResult or None
        Optimization diagnostics if ``info=True``, else None.

    Notes
    -----
    Optimization wrapper behavior:

    1. Builds SciPy options from method-specific defaults and user
       ``method_options``.
    2. Tracks full optimization history (parameter vectors and criterion values).
    3. If the final SciPy result is worse than the best visited point, replaces
       the returned solution by the best seen one and sets
       ``best_value_returned=False`` in the result object.

    Exception handling:
    criterion evaluation exceptions caused by linear-algebra failures are mapped
    to ``+inf`` inside ``criterion_with_history`` so optimization can continue.
    Other exceptions are re-raised.

    Added fields in returned ``OptimizeResult`` (when ``info=True``):
    ``history_params``, ``history_criterion``, ``initial_params``,
    ``final_params``, ``bounds``, ``selection_criterion``, ``total_time``,
    and ``best_value_returned``.
    """
    if method_options is None:
        method_options = {}
    tic = time.time()

    # local tube if needed
    safe_lower, safe_upper = -500, 500  # FIXME
    if bounds is None and bounds_auto:
        bounds = [
            (
                max(param - bounds_delta, safe_lower),
                min(param + bounds_delta, safe_upper),
            )
            for param in p0
        ]

    history_params, history_criterion = [], []
    best_params, best_criterion = None, float("inf")

    def record(p, J):
        nonlocal best_params, best_criterion
        history_params.append(p.copy())
        history_criterion.append(J)
        if J < best_criterion:
            best_criterion, best_params = J, p.copy()

    _is_linalg_exception = getattr(gnp, "_is_linalg_exception", None)

    def criterion_with_history(p):
        try:
            J = criterion(p)
        except Exception as exc:
            if callable(_is_linalg_exception) and _is_linalg_exception(exc):
                J = np.inf
            else:
                raise
        record(p, J)
        return J

    options = {"disp": not silent}
    if method == "L-BFGS-B":
        options.update(
            dict(
                maxcor=20,
                ftol=1e-6,
                gtol=1e-5,
                eps=1e-8,
                maxfun=15000,
                maxiter=15000,
                maxls=40,
                iprint=-1,
            )
        )
    elif method == "SLSQP":
        options.update(dict(ftol=1e-6, eps=1e-8, maxiter=15000))
    else:
        raise ValueError("Optimization method not implemented.")
    options.update(method_options)

    r = minimize(
        criterion_with_history,
        p0,
        method=method,
        jac=gradient,
        bounds=bounds,
        options=options,
    )

    # ensure returning best seen
    if r.fun > best_criterion:
        r.x, r.fun, r.best_value_returned = best_params, best_criterion, False
    else:
        r.best_value_returned = True

    r.history_params = history_params
    r.history_criterion = history_criterion
    r.initial_params = p0
    r.final_params = r.x
    r.bounds = bounds
    r.selection_criterion = criterion
    r.total_time = time.time() - tic

    return (r.x, r) if info else (r.x, None)


# -------------------- high-level parameter selection procedures  ------------
def select_parameters_with_criterion(
    model,
    criterion,
    xi=None,
    zi=None,
    dataloader=None,
    meanparam0=None,
    covparam0=None,
    parameterized_mean=False,
    meanparam_len=1,
    info=False,
    verbosity=0,
    *,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    batches_per_eval=0,
    method="SLSQP",
    method_options=None,
):
    """
    Optimize model parameters using a user-supplied selection criterion.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model whose parameters are optimized.
    criterion : callable
        Criterion minimized by SciPy.
        Expected signatures:
        - ``criterion(model, covparam, xi, zi)`` when ``parameterized_mean=False``
        - ``criterion(model, meanparam, covparam, xi, zi)`` when
          ``parameterized_mean=True``.
    xi, zi : array_like, optional
        Dataset arrays. Must be provided together unless ``dataloader`` is
        used instead.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``. Must yield batches
        ``(xb, zb)`` compatible with ``criterion``.
    meanparam0, covparam0 : array_like, optional
        Initial parameters in normalized space.
        If ``covparam0`` is None, an anisotropic initial guess is computed from
        the provided data source.
    parameterized_mean : bool, default False
        If True, optimize both mean and covariance parameters jointly using the
        concatenated vector ``[meanparam, covparam]``.
    meanparam_len : int, default 1
        Number of leading entries in the concatenated vector corresponding to
        mean parameters.
    info : bool, default False
        If True, return optimization diagnostics.
    verbosity : int, default 0
        0: silent, 1: short progress message, 2: SciPy solver output.
    bounds, bounds_auto, bounds_delta :
        Bounds configuration in normalized parameter space, forwarded to
        ``autoselect_parameters``.
    batches_per_eval : int, default 0
        Number of loader batches per objective call when using ``dataloader``.
        ``0`` means one full pass over loader per criterion evaluation.
        ``>0`` means evaluate on exactly that many batches (with iterator cycling).
    method : str, default "SLSQP"
        Optimization method ("SLSQP" or "L-BFGS-B").
    method_options : dict, optional
        Extra options passed to SciPy ``minimize``.

    Returns
    -------
    model : gpmp.core.Model
        Model with updated parameters.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.

    Notes
    -----
    Data source contract:
    exactly one of ``(xi, zi)`` or ``dataloader`` must be provided.

    Internally, this function constructs four complementary criterion
    callables from ``make_selection_criterion_with_gradient`` required by
    optimization and diagnostics, then optimizes with ``autoselect_parameters``.

    When ``info=True``, the returned diagnostics include optimization metadata
    (history, timing, parameters) and both callable criteria:
    ``selection_criterion`` and ``selection_criterion_nograd``.
    """
    if method_options is None:
        method_options = {}

    tic = time.time()
    _source = check_xi_zi_or_loader(xi, zi, dataloader)

    if covparam0 is None:
        covparam0 = anisotropic_parameters_initial_guess(model, xi, zi, dataloader)

    if parameterized_mean:
        if meanparam0 is None:
            raise ValueError(
                "meanparam0 must be provided when parameterized_mean=True."
            )
        param0 = gnp.concatenate([meanparam0, covparam0])
    else:
        param0 = covparam0

    crit, crit_pre_grad, crit_no_grad, crit_grad = (
        make_selection_criterion_with_gradient(
            model,
            criterion,
            xi,
            zi,
            dataloader,
            batches_per_eval=batches_per_eval,
            parameterized_mean=parameterized_mean,
            meanparam_len=meanparam_len,
        )
    )

    silent = not (verbosity == 2)
    if verbosity == 1:
        print("Parameter selection using custom criterion...")

    param_opt, info_ret = autoselect_parameters(
        param0,
        crit_pre_grad,
        crit_grad,
        bounds=bounds,
        bounds_auto=bounds_auto,
        bounds_delta=bounds_delta,
        silent=silent,
        info=True,
        method=method,
        method_options=method_options,
    )

    if verbosity == 1:
        print("done.")

    # split back
    if parameterized_mean:
        meanparam_opt = param_opt[:meanparam_len]
        covparam_opt = param_opt[meanparam_len:]
        model.meanparam = gnp.asarray(meanparam_opt)
    else:
        meanparam_opt = None
        covparam_opt = param_opt
    model.covparam = gnp.asarray(covparam_opt)

    if info:
        info_ret["meanparam0"] = gnp.to_np(meanparam0) if parameterized_mean else None
        info_ret["covparam0"] = gnp.to_np(covparam0)
        info_ret["meanparam"] = meanparam_opt
        info_ret["covparam"] = covparam_opt
        info_ret["selection_criterion"] = crit
        info_ret["selection_criterion_nograd"] = crit_no_grad
        info_ret["time"] = time.time() - tic
        return model, info_ret
    return model, None


def update_parameters_with_criterion(
    model,
    criterion,
    xi=None,
    zi=None,
    dataloader=None,
    parameterized_mean=False,
    meanparam_len=1,
    info=False,
    *,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    method="SLSQP",
    method_options=None,
):
    """
    Update model parameters using current model parameters as initialization.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance to update.
    criterion : callable
        Selection criterion to minimize.
    xi, zi : array_like, optional
        Dataset arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    parameterized_mean : bool, default=False
        Whether mean parameters are optimized jointly.
    meanparam_len : int, default=1
        Number of mean parameters in concatenated vectors.
    info : bool, default=False
        If True, return optimization diagnostics.
    bounds, bounds_auto, bounds_delta :
        Bounds configuration in normalized parameter space.
    method : {"SLSQP", "L-BFGS-B"}, default="SLSQP"
        Optimization method.
    method_options : dict, optional
        Extra options passed to SciPy ``minimize``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.
    """
    return select_parameters_with_criterion(
        model,
        criterion,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        meanparam0=model.meanparam if parameterized_mean else None,
        covparam0=model.covparam,
        parameterized_mean=parameterized_mean,
        meanparam_len=meanparam_len,
        info=info,
        verbosity=0,
        bounds=bounds,
        bounds_auto=bounds_auto,
        bounds_delta=bounds_delta,
        method=method,
        method_options=method_options,
    )


# ------------------------- objective wrappers -------------------------
def negative_log_likelihood_zero_mean(model, covparam, xi, zi):
    """
    Evaluate the negative log-likelihood for a zero-mean GP model.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    covparam : array_like
        Covariance parameter vector.
    xi, zi : array_like
        Observation points and observed values.

    Returns
    -------
    scalar
        Negative log-likelihood value.
    """
    return model.negative_log_likelihood_zero_mean(covparam, xi, zi)


def negative_log_likelihood(model, meanparam, covparam, xi, zi):
    """
    Evaluate the negative log-likelihood for a GP model with mean parameters.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    meanparam : array_like
        Mean-function parameter vector.
    covparam : array_like
        Covariance parameter vector.
    xi, zi : array_like
        Observation points and observed values.

    Returns
    -------
    scalar
        Negative log-likelihood value.
    """
    return model.negative_log_likelihood(meanparam, covparam, xi, zi)


def negative_log_restricted_likelihood(model, covparam, xi, zi):
    """
    Evaluate the negative restricted log-likelihood (REML criterion).

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    covparam : array_like
        Covariance parameter vector.
    xi, zi : array_like
        Observation points and observed values.

    Returns
    -------
    scalar
        Negative restricted log-likelihood value.
    """
    return model.negative_log_restricted_likelihood(covparam, xi, zi)


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


def compute_logrho_min_from_xi(xi, rho_min_range_factor=20.0):
    """
    Compute safeguarded componentwise ``logrho_min`` from observation points.

    The bound combines two componentwise lower bounds and keeps the tightest
    admissible one:

    1. ``log(min nonzero gap)``
    2. ``log(range / rho_min_range_factor)``

    Parameters
    ----------
    xi : array_like of shape (n, d)
        Observation points.
    rho_min_range_factor : float, default=20.0
        Safeguard factor for the range-based lower bound.

    Returns
    -------
    logrho_min : array_like of shape (d,)
        Safeguarded componentwise lower bound for ``logrho``.
    """
    if rho_min_range_factor <= 0:
        raise ValueError("rho_min_range_factor must be strictly positive.")
    logrho_min_gap, x_range = _componentwise_logrho_min_from_xi(xi)
    min_rho_from_range = x_range / float(rho_min_range_factor)
    positive_mask = min_rho_from_range > 0.0
    min_rho_safe = gnp.where(positive_mask, min_rho_from_range, 1.0)
    logrho_min_range = gnp.where(positive_mask, gnp.log(min_rho_safe), -gnp.inf)
    return gnp.maximum(logrho_min_gap, logrho_min_range)


# ------------------------------------------------------------------------
#
#                   specific parameter selection procedures
#
# ------------------------------------------------------------------------


# --------------------------------- REML ---------------------------------
def select_parameters_with_reml(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    covparam0=None,
    info=False,
    verbosity=0,
    *,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    method="SLSQP",
    method_options=None,
):
    """
    Select covariance parameters with REML.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    xi, zi : array_like, optional
        Dataset arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    covparam0 : array_like, optional
        Initial covariance parameters. If None, an anisotropic initial guess
        is computed.
    info : bool, default=False
        If True, return optimization diagnostics.
    verbosity : int, default=0
        Verbosity level forwarded to generic criterion selection.
    bounds, bounds_auto, bounds_delta :
        Bounds configuration in normalized parameter space.
    method : {"SLSQP", "L-BFGS-B"}, default="SLSQP"
        Optimization method.
    method_options : dict, optional
        Extra options passed to SciPy ``minimize``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.
    """
    return select_parameters_with_criterion(
        model,
        negative_log_restricted_likelihood,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        covparam0=covparam0,
        info=info,
        verbosity=verbosity,
        bounds=bounds,
        bounds_auto=bounds_auto,
        bounds_delta=bounds_delta,
        method=method,
        method_options=method_options,
    )


def update_parameters_with_reml(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    info=False,
    *,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    method="SLSQP",
    method_options=None,
):
    """
    Update covariance parameters with REML from current model parameters.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    xi, zi : array_like, optional
        Observation arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    info : bool, default=False
        If True, return optimization diagnostics.
    bounds, bounds_auto, bounds_delta :
        Bounds configuration in normalized parameter space.
    method : {"SLSQP", "L-BFGS-B"}, default="SLSQP"
        Optimization method.
    method_options : dict, optional
        Extra options passed to SciPy ``minimize``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.
    """
    return update_parameters_with_criterion(
        model,
        negative_log_restricted_likelihood,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        info=info,
        bounds=bounds,
        bounds_auto=bounds_auto,
        bounds_delta=bounds_delta,
        method=method,
        method_options=method_options,
    )


# ---------------------------- REMAP (default prior) ----------------------------
def select_parameters_with_remap(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    covparam0=None,
    info=False,
    verbosity=0,
    **kwargs,
):
    """
    Alias of ``select_parameters_with_remap_gaussian_logsigma2_and_logrho_prior``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    xi, zi : array_like, optional
        Observation arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    covparam0 : array_like, optional
        Initial covariance parameters.
    info : bool, default=False
        If True, return optimization diagnostics.
    verbosity : int, default=0
        Verbosity level forwarded to the target function.
    **kwargs
        Additional keyword arguments forwarded to
        ``select_parameters_with_remap_gaussian_logsigma2_and_logrho_prior``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.
    """
    return select_parameters_with_remap_gaussian_logsigma2_and_logrho_prior(
        model,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        covparam0=covparam0,
        info=info,
        verbosity=verbosity,
        **kwargs,
    )


def update_parameters_with_remap(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    info=False,
    verbosity=0,
    **kwargs,
):
    """
    Alias of ``update_parameters_with_remap_gaussian_logsigma2_and_logrho_prior``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    xi, zi : array_like, optional
        Observation arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    info : bool, default=False
        If True, return optimization diagnostics.
    verbosity : int, default=0
        Verbosity level forwarded to the target function.
    **kwargs
        Additional keyword arguments forwarded to
        ``update_parameters_with_remap_gaussian_logsigma2_and_logrho_prior``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.
    """
    return update_parameters_with_remap_gaussian_logsigma2_and_logrho_prior(
        model,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        info=info,
        verbosity=verbosity,
        **kwargs,
    )


# --------------------- REMAP with power laws prior  ----------------------
def select_parameters_with_remap_with_power_laws_prior(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    covparam0=None,
    info=False,
    verbosity=0,
    *,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    method="SLSQP",
    method_options=None,
):
    """
    Select covariance parameters with REMAP and power-laws prior.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    xi, zi : array_like, optional
        Observation arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    covparam0 : array_like, optional
        Initial covariance parameters. If None, an anisotropic initial guess
        is computed.
    info : bool, default=False
        If True, return optimization diagnostics.
    verbosity : int, default=0
        Verbosity level forwarded to generic criterion selection.
    bounds, bounds_auto, bounds_delta :
        Bounds configuration in normalized parameter space.
    method : {"SLSQP", "L-BFGS-B"}, default="SLSQP"
        Optimization method.
    method_options : dict, optional
        Extra options passed to SciPy ``minimize``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.
    """
    return select_parameters_with_criterion(
        model,
        neg_log_restricted_posterior_power_laws_prior,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        covparam0=covparam0,
        info=info,
        verbosity=verbosity,
        bounds=bounds,
        bounds_auto=bounds_auto,
        bounds_delta=bounds_delta,
        method=method,
        method_options=method_options,
    )


def update_parameters_with_remap_with_power_laws_prior(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    info=False,
    *,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    method="SLSQP",
    method_options=None,
):
    """
    Update covariance parameters with REMAP and power-laws prior.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    xi, zi : array_like, optional
        Observation arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    info : bool, default=False
        If True, return optimization diagnostics.
    bounds, bounds_auto, bounds_delta :
        Bounds configuration in normalized parameter space.
    method : {"SLSQP", "L-BFGS-B"}, default="SLSQP"
        Optimization method.
    method_options : dict, optional
        Extra options passed to SciPy ``minimize``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.
    """
    return update_parameters_with_criterion(
        model,
        neg_log_restricted_posterior_power_laws_prior,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        info=info,
        bounds=bounds,
        bounds_auto=bounds_auto,
        bounds_delta=bounds_delta,
        method=method,
        method_options=method_options,
    )


# --------------------- REMAP with gaussian prior on logsigma2  --------------------
def select_parameters_with_remap_gaussian_logsigma2(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    covparam0=None,
    info=False,
    verbosity=0,
    *,
    gamma=2.0,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    method="SLSQP",
    method_options=None,
):
    """
    Select covariance parameters with REMAP and Gaussian prior on ``log(sigma^2)``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    xi, zi : array_like, optional
        Observation arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    covparam0 : array_like, optional
        Initial covariance parameters. If None, an anisotropic initial guess
        is computed.
    info : bool, default=False
        If True, return optimization diagnostics.
    verbosity : int, default=0
        Verbosity level forwarded to generic criterion selection.
    gamma : float, default=2.0
        Prior spread control for ``log(sigma^2)``.
    bounds, bounds_auto, bounds_delta :
        Bounds configuration in normalized parameter space.
    method : {"SLSQP", "L-BFGS-B"}, default="SLSQP"
        Optimization method.
    method_options : dict, optional
        Extra options passed to SciPy ``minimize``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.

    Notes
    -----
    The Gaussian prior center ``log_sigma2_0`` is taken from ``covparam0[0]``.
    """
    if covparam0 is None:
        covparam0 = anisotropic_parameters_initial_guess(model, xi, zi, dataloader)
    log_sigma2_0 = covparam0[0]

    def criterion(m, covparam, x, z):
        return neg_log_restricted_posterior_gaussian_logsigma2_prior(
            m,
            covparam,
            x,
            z,
            log_sigma2_0=log_sigma2_0,
            gamma=gamma,
        )

    return select_parameters_with_criterion(
        model,
        criterion,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        covparam0=covparam0,
        info=info,
        verbosity=verbosity,
        bounds=bounds,
        bounds_auto=bounds_auto,
        bounds_delta=bounds_delta,
        method=method,
        method_options=method_options,
    )


def update_parameters_with_remap_gaussian_logsigma2(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    info=False,
    verbosity=0,
    *,
    gamma=4.0,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    method="SLSQP",
    method_options=None,
):
    """
    Update covariance parameters with REMAP and Gaussian prior on ``log(sigma^2)``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    xi, zi : array_like, optional
        Observation arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    info : bool, default=False
        If True, return optimization diagnostics.
    verbosity : int, default=0
        Verbosity level forwarded to the selector function.
    gamma : float, default=4.0
        Prior spread control for ``log(sigma^2)``.
    bounds, bounds_auto, bounds_delta :
        Bounds configuration in normalized parameter space.
    method : {"SLSQP", "L-BFGS-B"}, default="SLSQP"
        Optimization method.
    method_options : dict, optional
        Extra options passed to SciPy ``minimize``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.
    """
    covparam0 = model.covparam
    if covparam0 is None:
        covparam0 = anisotropic_parameters_initial_guess(model, xi, zi, dataloader)
    return select_parameters_with_remap_gaussian_logsigma2(
        model,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        covparam0=covparam0,
        info=info,
        verbosity=verbosity,
        gamma=gamma,
        bounds=bounds,
        bounds_auto=bounds_auto,
        bounds_delta=bounds_delta,
        method=method,
        method_options=method_options,
    )


# ------------------ REMAP with priors on logsigma2 and logrho  -------------------
def select_parameters_with_remap_gaussian_logsigma2_and_logrho_prior(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    covparam0=None,
    info=False,
    verbosity=0,
    *,
    gamma=1.5,
    rho_min_range_factor=20.0,
    logrho_min=None,
    logrho_0=None,
    alpha=10.0,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    method="SLSQP",
    method_options=None,
):
    """
    Select covariance parameters with REMAP and priors on ``log(sigma^2)`` and ``logrho``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    xi, zi : array_like, optional
        Observation arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    covparam0 : array_like, optional
        Initial covariance parameters. If None, an anisotropic initial guess
        is computed.
    info : bool, default=False
        If True, return optimization diagnostics.
    verbosity : int, default=0
        Verbosity level forwarded to generic criterion selection.
    gamma : float, default=2.0
        Prior spread control for ``log(sigma^2)``.
    rho_min_range_factor : float, default=20.0
        Safeguard factor used when ``logrho_min`` is inferred from data:
        a componentwise lower bound
        ``log(range(x[:, j]) / rho_min_range_factor)`` is
        applied in addition to the minimum-gap bound.
    logrho_min : array_like, optional
        Lower bounds for ``logrho`` prior support.
    logrho_0 : array_like, optional
        Reference values for ``logrho`` prior.
    alpha : float, default=10.0
        Scale parameter of the ``logrho`` barrier-linear prior.
    bounds, bounds_auto, bounds_delta :
        Bounds configuration in normalized parameter space.
    method : {"SLSQP", "L-BFGS-B"}, default="SLSQP"
        Optimization method.
    method_options : dict, optional
        Extra options passed to SciPy ``minimize``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.

    Notes
    -----
    If ``logrho_min`` is None, it is computed componentwise from minimum nonzero
    gaps in ``xi`` and safeguarded with ``log(range / rho_min_range_factor)``. If
    ``xi`` is not provided, ``dataloader.dataset.x_list`` is used when available.
    """
    if covparam0 is None:
        covparam0 = anisotropic_parameters_initial_guess(model, xi, zi, dataloader)

    log_sigma2_0 = covparam0[0]
    logrho_0 = -covparam0[1:] if logrho_0 is None else logrho_0
    if logrho_min is None:
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
                    "dataloader.dataset must provide x_list when logrho_min is None."
                )
        else:
            raise ValueError(
                "xi or dataloader.dataset.x_list must be provided when logrho_min is None."
            )
        logrho_min = compute_logrho_min_from_xi(
            xi_for_min, rho_min_range_factor=rho_min_range_factor
        )
    logrho_min = gnp.asarray(logrho_min)
    logrho_0 = gnp.asarray(logrho_0)

    def criterion(m, covparam, x, z):
        return neg_log_restricted_posterior_gaussian_logsigma2_and_logrho_prior(
            m,
            covparam,
            x,
            z,
            log_sigma2_0=log_sigma2_0,
            gamma=gamma,
            logrho_min=logrho_min,
            logrho_0=logrho_0,
            alpha=alpha,
        )

    return select_parameters_with_criterion(
        model,
        criterion,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        covparam0=covparam0,
        info=info,
        verbosity=verbosity,
        bounds=bounds,
        bounds_auto=bounds_auto,
        bounds_delta=bounds_delta,
        method=method,
        method_options=method_options,
    )


def update_parameters_with_remap_gaussian_logsigma2_and_logrho_prior(
    model,
    xi=None,
    zi=None,
    dataloader=None,
    info=False,
    verbosity=0,
    *,
    gamma=4.0,
    rho_min_range_factor=20.0,
    logrho_min=None,
    logrho_0=None,
    alpha=1.0,
    bounds=None,
    bounds_auto=True,
    bounds_delta=10.0,
    method="SLSQP",
    method_options=None,
):
    """
    Update covariance parameters with REMAP and priors on ``log(sigma^2)`` and ``logrho``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    xi, zi : array_like, optional
        Observation arrays.
    dataloader : iterable, optional
        Batch loader alternative to ``xi, zi``.
    info : bool, default=False
        If True, return optimization diagnostics.
    verbosity : int, default=0
        Verbosity level forwarded to the selector function.
    gamma : float, default=4.0
        Prior spread control for ``log(sigma^2)``.
    rho_min_range_factor : float, default=20.0
        Safeguard factor used when ``logrho_min`` is inferred from data:
        a componentwise lower bound
        ``log(range(x[:, j]) / rho_min_range_factor)`` is
        applied in addition to the minimum-gap bound.
    logrho_min : array_like, optional
        Lower bounds for ``logrho`` prior support.
    logrho_0 : array_like, optional
        Reference values for ``logrho`` prior.
    alpha : float, default=1.0
        Scale parameter of the ``logrho`` barrier-linear prior.
    bounds, bounds_auto, bounds_delta :
        Bounds configuration in normalized parameter space.
    method : {"SLSQP", "L-BFGS-B"}, default="SLSQP"
        Optimization method.
    method_options : dict, optional
        Extra options passed to SciPy ``minimize``.

    Returns
    -------
    model : gpmp.core.Model
        Updated model.
    info_ret : dict | None
        Diagnostics dictionary if ``info=True``, else None.
    """
    covparam0 = model.covparam
    if covparam0 is None:
        covparam0 = anisotropic_parameters_initial_guess(model, xi, zi, dataloader)
    return select_parameters_with_remap_gaussian_logsigma2_and_logrho_prior(
        model,
        xi=xi,
        zi=zi,
        dataloader=dataloader,
        covparam0=covparam0,
        info=info,
        verbosity=verbosity,
        gamma=gamma,
        rho_min_range_factor=rho_min_range_factor,
        logrho_min=logrho_min,
        logrho_0=logrho_0,
        alpha=alpha,
        bounds=bounds,
        bounds_auto=bounds_auto,
        bounds_delta=bounds_delta,
        method=method,
        method_options=method_options,
    )
