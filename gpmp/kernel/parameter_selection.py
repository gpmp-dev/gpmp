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
from .priors import neg_log_restricted_posterior_with_power_law_prior  # for REMAP


# ------------------------- objective wrappers -------------------------
def negative_log_likelihood_zero_mean(model, covparam, xi, zi):
    """Wrapper to model.negative_log_likelihood_zero_mean."""
    return model.negative_log_likelihood_zero_mean(covparam, xi, zi)


def negative_log_likelihood(model, meanparam, covparam, xi, zi):
    """Wrapper to model.negative_log_likelihood."""
    return model.negative_log_likelihood(meanparam, covparam, xi, zi)


def negative_log_restricted_likelihood(model, covparam, xi, zi):
    """Wrapper to model.negative_log_restricted_likelihood."""
    return model.negative_log_restricted_likelihood(covparam, xi, zi)


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
    """Build differentiable selection criterion for array or loader data."""
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
    """Minimize selection criterion with SciPy."""
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

    def criterion_with_history(p):
        try:
            J = criterion(p)
        except Exception:
            J = np.inf
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
    """Optimize model parameters using the given selection criterion."""
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
    """Update model parameters using the given selection criterion."""
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
    """Optimize covariance parameters with REML."""
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
    """Update covariance parameters with REML."""
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


def select_parameters_with_remap(
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
    """Optimize covariance parameters with REMAP (REML + prior)."""
    return select_parameters_with_criterion(
        model,
        neg_log_restricted_posterior_with_power_law_prior,
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


def update_parameters_with_remap(
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
    """Update covariance parameters with REMAP (REML + prior)."""
    return update_parameters_with_criterion(
        model,
        neg_log_restricted_posterior_with_power_law_prior,
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
