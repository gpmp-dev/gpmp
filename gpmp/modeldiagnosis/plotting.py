# gpmp/modeldiagnosis/plotting.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Plotting helpers for GP model diagnosis.

Defines
-------
plot_pit_ecdf
    Empirical CDF of PIT values.
plot_selection_criterion_crosssections
    One-dimensional cross sections of one or several selection criteria.
plot_selection_criterion_2d
    Two-dimensional profile of a selection criterion for two parameters.
plot_selection_criterion_sigma_rho
    Convenience wrapper for the common (sigma, rho) case.

Notes
-----
Matplotlib is imported inside this module. Importing gpmp.modeldiagnosis.plotting
will import matplotlib. Other diagnosis submodules do not import matplotlib.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

import gpmp.num as gnp

import matplotlib.pyplot as plt


def plot_pit_ecdf(pit: Any, fig: Optional[Any] = None) -> None:
    """
    Plot the empirical CDF of PIT values.

    Parameters
    ----------
    pit : array-like, shape (n,)
        PIT values in [0, 1].
    fig : matplotlib.figure.Figure, optional
        If provided, plot into this figure. If None, create a new figure.

    Returns
    -------
    None
    """
    pit = gnp.asarray(pit).reshape(-1)
    n = int(pit.shape[0])

    p = gnp.concatenate((gnp.array([0.0]), gnp.linspace(0.0, 1.0, n)))
    pit_sorted = gnp.concatenate((gnp.array([0.0]), gnp.sort(pit)))

    if fig is None:
        plt.figure()
    plt.step(gnp.to_np(pit_sorted), gnp.to_np(p))
    plt.plot([0.0, 1.0], [0.0, 1.0])
    plt.title("PIT ECDF")
    plt.xlabel("PIT")
    plt.ylabel("ECDF")
    plt.show()


def plot_selection_criterion_crosssections(
    *,
    info: Optional[Any] = None,
    selection_criterion: Optional[Any] = None,
    selection_criteria: Optional[Sequence[Any]] = None,
    covparam: Optional[Any] = None,
    n_points: int = 100,
    param_names: Optional[Sequence[str]] = None,
    criterion_name: str = "selection criterion",
    criterion_names: Optional[Sequence[str]] = None,
    criterion_name_full: str = "Cross sections of selection criterion",
    ind: Optional[Sequence[int]] = None,
    ind_pooled: Optional[Sequence[int]] = None,
    param_box: Optional[Any] = None,
    param_box_pooled: Optional[Any] = None,
    delta: float = 5.0,
) -> None:
    """
    Plot 1D cross sections of one or several selection criteria.

    Parameters
    ----------
    info : object, optional
        If provided, defaults are taken from:
        - info.selection_criterion_nograd
        - info.covparam
    selection_criterion : callable, optional
        Function f(covparam) -> scalar. Used when selection_criteria is None.
    selection_criteria : sequence of callables, optional
        Each callable f(covparam) -> scalar. If provided, overrides selection_criterion.
    covparam : array-like, optional
        Reference parameter vector. Required if info is None.
    n_points : int, optional
        Number of evaluation points per cross section.
    param_names : sequence of str, optional
        Names of parameters for titles/labels.
    criterion_name : str, optional
        Base label used when a single criterion is plotted.
    criterion_names : sequence of str, optional
        Labels for each criterion when multiple criteria are plotted.
    criterion_name_full : str, optional
        Figure title.
    ind : sequence of int, optional
        Indices for individual subplots. If both ind and ind_pooled are None,
        defaults to all parameters.
    ind_pooled : sequence of int, optional
        Indices for pooled plot (all curves on a single axis).
    param_box : array-like, optional
        Bounds for individual plots, shape (2, n_params). If provided:
        - lo = param_box[0, j], hi = param_box[1, j]
    param_box_pooled : array-like, optional
        Bounds for pooled plot, shape (2, n_params).
    delta : float, optional
        Default half-width around the reference value when a box is not provided.

    Returns
    -------
    None
    """
    try:
        interpreter = sys.ps1
    except AttributeError:
        interpreter = sys.flags.interactive
    if interpreter:
        plt.ion()

    if selection_criteria is None:
        if selection_criterion is None:
            if info is None:
                raise ValueError("Provide info or selection_criterion/selection_criteria.")
            selection_criterion = info.selection_criterion_nograd
        selection_criteria = (selection_criterion,)
    else:
        selection_criteria = tuple(selection_criteria)

    n_crit = len(selection_criteria)

    if criterion_names is None:
        if n_crit == 1:
            criterion_names = (criterion_name,)
        else:
            criterion_names = tuple(f"{criterion_name} #{k}" for k in range(n_crit))
    if len(criterion_names) != n_crit:
        raise ValueError("criterion_names length must match number of criteria.")

    if info is None:
        if covparam is None:
            raise ValueError("covparam must be supplied when info is None.")
        param_opt = gnp.asarray(covparam).reshape(-1)
    else:
        param_opt = gnp.asarray(covparam if covparam is not None else info.covparam).reshape(-1)

    n_params = int(param_opt.shape[0])

    if ind is None and ind_pooled is None:
        ind = list(range(n_params))

    def _grid_for_param(param_index: int, opt_val: Any, box: Optional[Any]) -> Any:
        if box is not None:
            lo = float(np.asarray(box)[0, param_index])
            hi = float(np.asarray(box)[1, param_index])
        else:
            lo = float(opt_val) - float(delta)
            hi = float(opt_val) + float(delta)
        return gnp.linspace(lo, hi, n_points)

    if ind is not None:
        ind = list(ind)
        n_ind = len(ind)
        fig, axes = plt.subplots(n_ind, 1, figsize=(8, min(9, 3 * n_ind)))
        if n_ind == 1:
            axes = [axes]

        for ax_i, param_idx in enumerate(ind):
            opt_value = param_opt[param_idx]
            p_values = _grid_for_param(param_idx, opt_value, param_box)
            crit_values = gnp.zeros((n_crit, n_points))
            for j, x_val in enumerate(p_values):
                param = gnp.copy(param_opt)
                param[param_idx] = x_val
                for k, f in enumerate(selection_criteria):
                    crit_values[k, j] = f(param)
            ax = axes[ax_i]
            for k in range(n_crit):
                ax.plot(gnp.to_np(p_values), gnp.to_np(crit_values[k]), label=criterion_names[k])
            ax.axvline(float(opt_value), color="red", linestyle="--", label="reference")
            name = (
                param_names[param_idx]
                if param_names is not None and param_idx < len(param_names)
                else f"param {param_idx}"
            )
            ax.set_title(name)
            ax.set_ylabel("criterion value")
            if ax_i == n_ind - 1:
                ax.set_xlabel("parameter value")
            if ax_i == 0:
                ax.legend()

        fig.suptitle(criterion_name_full, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    if ind_pooled is not None:
        ind_pooled = list(ind_pooled)
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, param_idx in enumerate(ind_pooled):
            opt_value = param_opt[param_idx]
            p_values = _grid_for_param(i, opt_value, param_box_pooled)
            crit_values = gnp.zeros((n_crit, n_points))
            for j, x_val in enumerate(p_values):
                param = gnp.copy(param_opt)
                param[param_idx] = x_val
                for k, f in enumerate(selection_criteria):
                    crit_values[k, j] = f(param)
            name = (
                param_names[param_idx]
                if param_names is not None and param_idx < len(param_names)
                else f"param {param_idx}"
            )
            for k in range(n_crit):
                ax.plot(
                    gnp.to_np(p_values),
                    gnp.to_np(crit_values[k]),
                    label=f"{name} - {criterion_names[k]}",
                )
            ax.axvline(float(opt_value), color="red", linestyle="--")

        ax.set_xlabel("parameter value")
        ax.set_ylabel("criterion value")
        ax.set_title(criterion_name_full)
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_selection_criterion_2d(
    model: Any,
    info: Any,
    *,
    param_indices: Tuple[int, int] = (0, 1),
    param_names: Optional[Sequence[str]] = None,
    criterion_name: str = "selection criterion",
    n: int = 130,
    factor: float = 4.0,
    shift_criterion: bool = True,
) -> None:
    """
    Plot a 2D profile of a selection criterion over two parameters.

    Parameters
    ----------
    model : object
        Model exposing covparam.
    info : object
        Must expose selection_criterion_nograd, covparam, and covparam0 (optional, for display).
    param_indices : tuple of int, optional
        Indices (i, j) in covparam to vary.
        Convention assumed by this function:
        - index 0 is log(sigma^2)
        - other indices are log(1/rho)
    param_names : sequence of str, optional
        Axis labels (two strings). If None, default labels are used.
    criterion_name : str, optional
        Title label.
    n : int, optional
        Grid size per axis (n x n evaluations).
    factor : float, optional
        Multiplicative factor defining the parameter range around the reference values
        in the natural (sigma, rho) space.
    shift_criterion : bool, optional
        If True, shift values by -min before log10 for display.

    Returns
    -------
    None
    """
    tic = time.time()

    def _progress(i: int) -> None:
        elapsed = time.time() - tic
        avg = elapsed / (i + 1)
        rem = avg * (n - (i + 1))
        pct = 100.0 * (i + 1) / n
        print(f"       Progress: {pct:.2f}% | time remaining: {rem:.1f}s", end="\r")

    def _final() -> None:
        elapsed = time.time() - tic
        print(f"       Progress: 100% complete | Total time: {elapsed:.3f}s")
        print(f"       number of evaluations: {n * n}")

    print(f"  ***  Computing {criterion_name} profile for plotting...")

    i1, i2 = param_indices
    cov0 = gnp.asarray(model.covparam).reshape(-1)

    p1_0 = math.exp(float(cov0[i1]) / 2.0) if i1 == 0 else math.exp(-float(cov0[i1]))
    p2_0 = math.exp(float(cov0[i2]) / 2.0) if i2 == 0 else math.exp(-float(cov0[i2]))

    p1 = gnp.logspace(math.log10(p1_0) - math.log10(factor), math.log10(p1_0) + math.log10(factor), n)
    p2 = gnp.logspace(math.log10(p2_0) - math.log10(factor), math.log10(p2_0) + math.log10(factor), n)

    p1_mesh, p2_mesh = np.meshgrid(p1, p2)
    log_p1 = gnp.log(p1_mesh**2) if i1 == 0 else gnp.log(1.0 / p1_mesh)
    log_p2 = gnp.log(p2_mesh**2) if i2 == 0 else gnp.log(1.0 / p2_mesh)

    f = info.selection_criterion_nograd
    values = gnp.zeros((n, n), dtype=gnp.get_dtype())

    covparam = gnp.copy(info.covparam)
    for i in range(n):
        _progress(i)
        for j in range(n):
            covparam[i1] = log_p1[i, j]
            covparam[i2] = log_p2[i, j]
            values[i, j] = f(covparam)

    values = gnp.nan_to_num(values, copy=False)
    _final()

    shift = -float(gnp.min(values.ravel())) if shift_criterion else 0.0
    z = gnp.log10(gnp.maximum(1e-2, values + shift))

    plt.figure()
    plt.contourf(gnp.log10(p1_mesh), gnp.log10(p2_mesh), z)

    cov_opt = gnp.asarray(info.covparam).reshape(-1)
    x_opt = 0.5 * gnp.log10(gnp.exp(cov_opt[i1])) if i1 == 0 else -gnp.log10(gnp.exp(float(cov_opt[i1])))
    y_opt = 0.5 * gnp.log10(gnp.exp(float(cov_opt[i2]))) if i2 == 0 else -gnp.log10(gnp.exp(float(cov_opt[i2])))
    plt.plot(x_opt, y_opt, "ro")

    cov0_disp = getattr(info, "covparam0", None)
    if cov0_disp is not None:
        cov0_disp = gnp.asarray(cov0_disp).reshape(-1)
        x0 = 0.5 * gnp.log10(gnp.exp(cov0_disp[i1])) if i1 == 0 else -gnp.log10(gnp.exp(float(cov0_disp[i1])))
        y0 = 0.5 * gnp.log10(gnp.exp(float(cov0_disp[i2]))) if i2 == 0 else -gnp.log10(gnp.exp(float(cov0_disp[i2])))
        plt.plot(x0, y0, "bo")

    if param_names is not None and len(param_names) >= 2:
        x_label, y_label = param_names[0], param_names[1]
    else:
        x_label = f"Parameter {i1} (log10)"
        y_label = f"Parameter {i2} (log10)"

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    title = "log10 of " + ("shifted " if shift_criterion else "") + str(criterion_name)
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_selection_criterion_sigma_rho(
    model: Any,
    info: Any,
    *,
    criterion_name: str = "negative log restricted likelihood",
    n: int = 130,
    factor: float = 4.0,
    shift_criterion: bool = True,
) -> None:
    """
    Plot a 2D profile for sigma (index 0) and rho (index 1).

    Parameters
    ----------
    model : object
        Model exposing covparam.
    info : object
        Must expose selection_criterion_nograd and covparam.
    criterion_name : str, optional
        Title label.
    n : int, optional
        Grid size per axis.
    factor : float, optional
        Multiplicative factor defining the range in (sigma, rho) space.
    shift_criterion : bool, optional
        If True, shift values by -min before log10 for display.

    Returns
    -------
    None
    """
    plot_selection_criterion_2d(
        model,
        info,
        param_indices=(0, 1),
        param_names=("sigma (log10)", "rho (log10)"),
        criterion_name=criterion_name,
        n=n,
        factor=factor,
        shift_criterion=shift_criterion,
    )


__all__ = [
    "plot_pit_ecdf",
    "plot_selection_criterion_crosssections",
    "plot_selection_criterion_2d",
    "plot_selection_criterion_sigma_rho",
]
