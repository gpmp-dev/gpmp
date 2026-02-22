# gpmp/modeldiagnosis/param_stats.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Parameter statistics from 1D criterion profiles.

This module summarizes a scalar selection criterion as a function of one
covariance parameter at a time, with all other parameters frozen. It provides:

- Grid-based summaries using a pseudo density w(x) = exp(-criterion(x)).
- Integration-based summaries using a 1D pseudo distribution built from
  log p(x) = -criterion(x).
- Fisher information via model.fisher_information.

Defined functions
-----------------
fast_univariate_stats
    Grid-based mean/variance/quantiles/mode of w(x) on a fixed interval.
make_single_param_criterion_function
    Closure g(x) that evaluates the criterion with one parameter set to x.
selection_criterion_statistics_fast
    Per-parameter grid summaries and Fisher information.
selection_criterion_statistics
    Per-parameter integration summaries and Fisher information.

Notes
-----
The statistics are computed independently for each parameter index. Bounds come
from param_box when provided; otherwise they are centered at the reference value
with half-width delta.
"""


from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar

import gpmp.num as gnp
from gpmp.misc.dataframe import DataFrame

from .un1ddist import Unnormalized1DDistribution


def _to_float(x: Any) -> float:
    """Convert scalar-like to float."""
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(x)


def fast_univariate_stats(
    single_param_fn: Callable[[float], Any],
    lower_bound: float,
    upper_bound: float,
    n_points: int = 100,
) -> Tuple[float, float, Dict[str, float], float]:
    """
    Compute weighted statistics on a scalar function evaluated on a grid.

    The pseudo density is w(x) = exp(-single_param_fn(x)).

    Parameters
    ----------
    single_param_fn : callable
        Function of one scalar returning a scalar-like value.
    lower_bound, upper_bound : float
        Integration bounds.
    n_points : int, optional
        Number of grid points.

    Returns
    -------
    mean_val : float
    variance : float
    quantiles : dict
        Keys are "0.1", "0.25", "0.5", "0.75", "0.9".
    mode_val : float
        Grid mode (argmax of w).
    """
    xs = np.linspace(float(lower_bound), float(upper_bound), int(n_points))

    vals = np.array([_to_float(single_param_fn(float(x))) for x in xs], dtype=float)
    logw = -vals
    logw -= np.max(logw)  # stabilize exp
    w = np.exp(logw)

    Z = np.trapz(w, xs)
    if not np.isfinite(Z) or Z <= 0.0:
        raise ValueError("Normalization failed in fast_univariate_stats.")

    mean_val = float(np.trapz(xs * w, xs) / Z)
    second = float(np.trapz((xs**2) * w, xs) / Z)
    variance = float(second - mean_val**2)

    cdf = cumulative_trapezoid(w, xs, initial=0.0) / Z
    quantiles: Dict[str, float] = {}
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        quantiles[str(q)] = float(np.interp(q, cdf, xs))

    mode_val = float(xs[int(np.argmax(w))])
    return mean_val, variance, quantiles, mode_val


def make_single_param_criterion_function(
    selection_criterion: Callable[[Any], Any],
    covparam: Any,
    param_index: int,
) -> Callable[[float], Any]:
    """
    Freeze all parameters except one in a covariance parameter vector.

    Parameters
    ----------
    selection_criterion : callable
        Function f(covparam) -> scalar-like.
    covparam : array-like
        Reference covariance parameter vector.
    param_index : int
        Index of the parameter to vary.

    Returns
    -------
    callable
        Function g(x) that evaluates f(covparam with covparam[param_index]=x).
    """

    covparam_ref = gnp.asarray(covparam)

    def single_param_function(x: float) -> Any:
        cp = gnp.copy(covparam_ref)
        cp[param_index] = x
        return selection_criterion(cp)

    return single_param_function


def selection_criterion_statistics_fast(
    info: Optional[Any] = None,
    model: Optional[Any] = None,
    xi: Optional[Any] = None,
    selection_criterion: Optional[Callable[[Any], Any]] = None,
    covparam: Optional[Any] = None,
    ind: Optional[Iterable[int]] = None,
    param_box: Optional[Any] = None,
    delta: float = 5.0,
    n_points: int = 250,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Grid-based parameter statistics and Fisher information.

    Parameters
    ----------
    info : object, optional
        If provided, defaults are taken from attributes:
        selection_criterion_nograd, covparam, model, xi.
    model : object
        Must expose fisher_information(xi, covparam, epsilon=...).
    xi : array-like
        Inputs passed to fisher_information.
    selection_criterion : callable, optional
        Function f(covparam) -> scalar-like.
    covparam : array-like, optional
        Covariance parameter vector.
    ind : iterable of int, optional
        Parameter indices. Default is all.
    param_box : array-like, optional
        Shape (2, n_params). Bounds per parameter.
    delta : float, optional
        Range is [opt - delta, opt + delta] when param_box is None.
    n_points : int, optional
        Grid size per parameter.
    verbose : bool, optional
        Print per-parameter summaries.

    Returns
    -------
    dict
        Keys are "parameter_statistics" (DataFrame) and "fisher_information".
    """
    if info is not None:
        if selection_criterion is None:
            selection_criterion = info.selection_criterion_nograd
        if covparam is None:
            covparam = info.covparam
        if model is None and hasattr(info, "model"):
            model = info.model
        if xi is None and hasattr(info, "xi"):
            xi = info.xi

    if selection_criterion is None:
        raise ValueError("selection_criterion is required.")
    if covparam is None:
        raise ValueError("covparam is required.")
    if model is None:
        raise ValueError("model is required.")
    if xi is None:
        raise ValueError("xi is required.")

    covparam = gnp.asarray(covparam).reshape(-1)
    n_params = int(covparam.shape[0])
    if ind is None:
        ind_list = list(range(n_params))
    else:
        ind_list = [int(i) for i in ind]

    box = None if param_box is None else np.asarray(param_box, dtype=float)

    rows: List[List[float]] = []
    row_names: List[str] = []
    col_names = [
        "mean",
        "variance",
        "quantile_0.1",
        "quantile_0.25",
        "quantile_0.5",
        "quantile_0.75",
        "quantile_0.9",
        "mode",
    ]

    for j in ind_list:
        opt = _to_float(covparam[j])
        if box is not None:
            lo = float(box[0, j])
            hi = float(box[1, j])
        else:
            lo = opt - float(delta)
            hi = opt + float(delta)

        sp = make_single_param_criterion_function(selection_criterion, covparam, j)
        mean_val, var_val, q, mode_val = fast_univariate_stats(
            sp, lo, hi, n_points=int(n_points)
        )

        if verbose:
            print(f"param {j}: mean={mean_val:.6g} var={var_val:.6g} mode={mode_val:.6g}")

        rows.append(
            [
                float(mean_val),
                float(var_val),
                float(q["0.1"]),
                float(q["0.25"]),
                float(q["0.5"]),
                float(q["0.75"]),
                float(q["0.9"]),
                float(mode_val),
            ]
        )
        row_names.append(f"param_{j:d}")

    stats_df = DataFrame(np.asarray(rows, dtype=float), col_names, row_names)
    fisher = model.fisher_information(xi, covparam, epsilon=1e-3)

    return {"parameter_statistics": stats_df, "fisher_information": fisher}


def selection_criterion_statistics(
    info: Optional[Any] = None,
    model: Optional[Any] = None,
    xi: Optional[Any] = None,
    selection_criterion: Optional[Callable[[Any], Any]] = None,
    covparam: Optional[Any] = None,
    ind: Optional[Iterable[int]] = None,
    param_box: Optional[Any] = None,
    delta: float = 5.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Integration-based parameter statistics and Fisher information.

    Each 1D marginal uses the pseudo log-pdf log p(x) = -criterion(x).

    Parameters
    ----------
    info, model, xi, selection_criterion, covparam, ind, param_box, delta, verbose
        Same meaning as in selection_criterion_statistics_fast.

    Returns
    -------
    dict
        Keys are "parameter_statistics" (DataFrame) and "fisher_information".
    """
    if info is not None:
        if selection_criterion is None:
            selection_criterion = info.selection_criterion_nograd
        if covparam is None:
            covparam = info.covparam
        if model is None and hasattr(info, "model"):
            model = info.model
        if xi is None and hasattr(info, "xi"):
            xi = info.xi

    if selection_criterion is None:
        raise ValueError("selection_criterion is required.")
    if covparam is None:
        raise ValueError("covparam is required.")
    if model is None:
        raise ValueError("model is required.")
    if xi is None:
        raise ValueError("xi is required.")

    covparam = gnp.asarray(covparam).reshape(-1)
    n_params = int(covparam.shape[0])
    if ind is None:
        ind_list = list(range(n_params))
    else:
        ind_list = [int(i) for i in ind]

    box = None if param_box is None else np.asarray(param_box, dtype=float)

    rows: List[List[float]] = []
    row_names: List[str] = []
    col_names = [
        "mean",
        "variance",
        "quantile_0.1",
        "quantile_0.25",
        "quantile_0.5",
        "quantile_0.75",
        "quantile_0.9",
        "mode",
    ]

    for j in ind_list:
        opt = _to_float(covparam[j])
        if box is not None:
            lo = float(box[0, j])
            hi = float(box[1, j])
        else:
            lo = opt - float(delta)
            hi = opt + float(delta)

        sp = make_single_param_criterion_function(selection_criterion, covparam, j)

        def log_pdf_scalar(x: float) -> float:
            return -_to_float(sp(float(x)))

        dist = Unnormalized1DDistribution(log_pdf_scalar, bounds=(lo, hi))

        mean_val = float(dist.mean())
        var_val = float(dist.var())
        q01 = float(dist.quantile(0.1))
        q25 = float(dist.quantile(0.25))
        q50 = float(dist.quantile(0.5))
        q75 = float(dist.quantile(0.75))
        q90 = float(dist.quantile(0.9))

        res = minimize_scalar(lambda x: _to_float(sp(float(x))), bounds=(lo, hi), method="bounded")
        mode_val = float(res.x) if getattr(res, "success", False) else float(opt)

        if verbose:
            print(f"param {j}: mean={mean_val:.6g} var={var_val:.6g} mode={mode_val:.6g}")

        rows.append([mean_val, var_val, q01, q25, q50, q75, q90, mode_val])
        row_names.append(f"param_{j:d}")

    stats_df = DataFrame(np.asarray(rows, dtype=float), col_names, row_names)
    fisher = model.fisher_information(xi, covparam, epsilon=1e-3)

    return {"parameter_statistics": stats_df, "fisher_information": fisher}


__all__ = [
    "fast_univariate_stats",
    "make_single_param_criterion_function",
    "selection_criterion_statistics_fast",
    "selection_criterion_statistics",
]
