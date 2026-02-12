"""Model Diagnosis Tool for Gaussian Process Models

This script provides functions to diagnose Gaussian Process (GP)
models. The functions are organized into the following sections:

1. Parameter Statistics and Analysis:
   - Unnormalized1DDistribution: Constructs a normalized density from
     an unnormalized log-pdf.
   - test_unormalized_1d_distribution: Tests the
     Unnormalized1DDistribution class.
   - fast_univariate_stats: Computes weighted statistics using a
     grid-based evaluation.
   - make_single_param_criterion_function: Generates a function that
     varies a single parameter.
   - selection_criterion_statistics_fast: Provides fast, grid-based
     parameter statistics and Fisher information.
   - selection_criterion_statistics: Computes parameter statistics
     using integration over a pseudo-density.

2. Performance Evaluation and Model Metrics:
   - diag: Runs the overall model diagnosis and displays results.
   - perf: Computes and prints predictive performance metrics.
   - modeldiagnosis_init: Initializes a diagnostic report with
     optimization and parameter information.
   - compute_performance: Calculates GP performance metrics (LOO, test
     metrics, PIT, etc.).
   - model_diagnosis_disp: Displays detailed model diagnosis information.

3. Visualization and Plotting Tools:
   - plot_pit_ecdf: Plots the empirical cumulative distribution
     function (ECDF) for PIT values.
   - plot_selection_criterion_crosssections: Generates 1D cross-section
     plots of the selection criterion.
   - plot_selection_criterion_2d: Creates a 2D contour plot for two
     selected parameters.
   - plot_selection_criterion_sigma_rho: Specialized 2D plot for sigma
     and rho parameters.

4. Utilities and Data Description:
   - sigma_rho_from_covparam: Extracts sigma and rho from the
     covariance parameters.
   - describe_array: Builds a DataFrame with descriptive statistics
     for data arrays.
   - pretty_print_dictionnary: Formats and prints dictionaries in a
     readable manner.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import sys
import time
import math
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import brentq
import numpy as np
import gpmp.num as gnp
import gpmp as gp
from gpmp.misc.dataframe import DataFrame, ftos
from gpmp.misc.param import param_from_covparam_anisotropic, param_from_covparam_anisotropic_noisy
import matplotlib.pyplot as plt
from matplotlib import interactive

# ============================================================
# Parameter statistics and analysis
# ============================================================


class Unnormalized1DDistribution:
    def __init__(self, log_pdf, bounds):
        """
        Parameters:
        - log_pdf: function (float -> float), unnormalized log-pdf (scalar-only)
        - bounds: tuple (a, b), integration bounds for numerical integration
        """
        self.log_pdf = log_pdf
        self.bounds = bounds
        self.f_scalar = lambda x: gnp.exp(gnp.asarray(self.log_pdf(x)))

        # Normalization constant
        self.Z, _ = quad(self.f_scalar, *self.bounds)

    def f(self, x: gnp.ndarray) -> gnp.ndarray:
        """Unnormalized density"""
        return gnp.asarray([self.f_scalar(x_scalar) for x_scalar in x])

    def pdf(self, x: gnp.ndarray) -> gnp.ndarray:
        """Normalized density evaluated"""
        return self.f(x) / self.Z

    def cdf(self, x: float) -> float:
        """CDF at scalar x (float input only)"""
        integral, _ = quad(self.f_scalar, self.bounds[0], x)
        return integral / self.Z

    def mean(self) -> float:
        integrand = lambda x: x * self.f_scalar(x)
        mu, _ = quad(integrand, *self.bounds)
        return mu / self.Z

    def var(self) -> float:
        mu = self.mean()
        integrand = lambda x: x**2 * self.f_scalar(x)
        second_moment, _ = quad(integrand, *self.bounds)
        return second_moment / self.Z - mu**2

    def quantile(self, p: float, xtol: float = 1e-6) -> float:
        """Quantile for probability p"""
        a, b = self.bounds
        return brentq(lambda x: self.cdf(x) - p, a, b, xtol=xtol)


def test_unormalized_1d_distribution():
    from scipy.stats import t

    def log_pdf_scalar(x: float) -> float:
        # Non-vectorized log-pdf
        return t.logpdf(x, df=5) + 1.0

    dist = Unnormalized1DDistribution(log_pdf_scalar, bounds=(-gnp.inf, gnp.inf))

    # Scalar evaluations
    print("Mean:", dist.mean())
    print("Variance:", dist.variance())
    print("Quantile (0.9):", dist.quantile(0.9))

    # Tensor evaluation for plotting or analysis
    x = gnp.linspace(-3, 3, 200)
    pdf_vals = dist.pdf(x)


def fast_univariate_stats(single_param_fn, lower_bound, upper_bound, n_points=100):
    """
    Compute statistics on a univariate function by evaluating on a linspace.

    The pseudo-pdf is defined as:
         f(x) = exp( - single_param_fn(x) )
    so that lower criterion values yield higher weight.

    Parameters
    ----------
    single_param_fn : callable
         Function of one variable (scalar-only).
    lower_bound, upper_bound : float
         Integration bounds.
    n_points : int, optional
         Number of grid points.

    Returns
    -------
    mean_val : float
         Estimated weighted mean.
    variance : float
         Estimated weighted variance.
    quantiles : dict
         Dictionary with keys "0.1", "0.25", "0.5", "0.75", "0.9" for the quantiles.
    mode_val : float
         Grid estimate for the mode (the x-value maximizing f(x), i.e. minimizing the criterion).
    """
    xs = np.linspace(lower_bound, upper_bound, n_points)
    # Evaluate the pseudo pdf: f(x) = exp( - single_param_fn(x) )
    f_vals = np.array([np.exp(-single_param_fn(x)) for x in xs])
    # Integration step size (assumes uniform grid)
    dx = xs[1] - xs[0]
    # Normalization constant (using trapezoidal rule)
    Z = np.trapz(f_vals, xs)
    mean_val = np.trapz(xs * f_vals, xs) / Z
    variance = np.trapz((xs**2) * f_vals, xs) / Z - mean_val**2
    # Compute CDF via cumulative trapezoidal integration.
    cdf_vals = cumulative_trapezoid(f_vals, xs, initial=0)
    cdf_vals = cdf_vals / Z
    quantiles = {}
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        quantiles[str(q)] = float(np.interp(q, cdf_vals, xs))
    # Mode: x-value at maximum f(x)
    mode_val = xs[np.argmax(f_vals)]
    return mean_val, variance, quantiles, mode_val


def make_single_param_criterion_function(selection_criterion, covparam, param_index):
    """
    Create a function of one float variable that computes the selection criterion
    when only the parameter at param_index is varied, keeping all other parameters
    in covparam fixed.
    """

    def single_param_function(x):
        new_covparam = gnp.copy(covparam)
        new_covparam[param_index] = x
        return selection_criterion(new_covparam)

    return single_param_function


def selection_criterion_statistics_fast(
    info=None,
    model=None,
    xi=None,
    selection_criterion=None,
    covparam=None,
    ind=None,
    param_box=None,
    delta=5.0,
    n_points=250,
    verbose=False,
):
    """Compute statistics on the selection criterion viewed as a
    negative log pdf, for the specified dimensions of covparam, and
    compute the Fisher information matrix using
    model.fisher_information(xi, covparam, epsilon=1e-3).

    For each parameter in `ind`, the function:
      - Creates a single-parameter function (via make_single_param_criterion_function).
      - Evaluates the pseudo–pdf f(x)=exp(–selection_criterion) on a linspace.
      - Computes the weighted mean, variance, and quantiles at 0.1, 0.25, 0.5, 0.75, 0.9.
      - Determines the mode, defined as the x-value at which f(x) is maximized.

    Parameters
    ----------
    info : object, optional
        If provided—and if selection_criterion or covparam is not supplied explicitly—they
        are taken from this object (attributes selection_criterion_nograd and covparam).
    model : object, required
        The model object, assumed to implement a method fisher_information(xi, covparam, epsilon).
    xi : array-like, required
        Input data points to be passed to model.fisher_information.
    selection_criterion : callable, optional
        A function that accepts a covparam vector and returns a scalar value.
    covparam : array-like, optional
        The covariance parameters vector.
    ind : list or array-like, optional
        List of indices (dimensions) for which statistics are computed. If not provided, all dimensions are used.
    param_box : 2D array-like, optional
        Two-row array with min and max values for each parameter; if provided, these are used
        to set integration bounds. Otherwise, bounds are set to:
            [opt_value - delta, opt_value + delta],
        where opt_value is the reference value in covparam.
    delta : float, optional
        Default range ±delta around the reference parameter value if param_box is not provided.
    n_points : int, optional
        Number of points to use in the linspace for integration (default is 250).
    verbose : bool, optional
        If True, prints a compressed summary for each parameter and the Fisher matrix.

    Returns
    -------
    result_dict : dict
        Dictionary with two keys:
          "parameter_statistics" : DataFrame
              A DataFrame where each row corresponds to a parameter. Columns are:
                "mean", "variance", "quantile_0.1", "quantile_0.25",
                "quantile_0.5", "quantile_0.75", "quantile_0.9", "mode".
          "fisher_information" : array-like
              The Fisher information matrix as computed by model.fisher_information(xi, covparam, epsilon=1e-3).

    """
    # If info is provided, set defaults.
    if info is not None:
        if selection_criterion is None:
            selection_criterion = info.selection_criterion_nograd
        if covparam is None:
            covparam = info.covparam
        if model is None and hasattr(info, "model"):
            model = info.model
        if xi is None and hasattr(info, "xi"):
            xi = info.xi

    if covparam is None:
        raise ValueError("covparam must be provided either directly or via info.")
    if model is None:
        raise ValueError("model must be provided either directly or via info.")
    if xi is None:
        raise ValueError("xi must be provided either directly or via info.")

    covparam = gnp.asarray(covparam)
    n_params = covparam.shape[0]
    # If no indices specified, use all dimensions.
    if ind is None:
        ind = list(range(n_params))

    parameter_stats = {}

    for param_index in ind:
        opt_value = covparam[param_index]
        if param_box is not None:
            lower_bound = float(param_box[0][param_index])
            upper_bound = float(param_box[1][param_index])
        else:
            lower_bound = opt_value - delta
            upper_bound = opt_value + delta

        if verbose:
            print(f"\nProcessing parameter index {param_index}:")
            print(
                f"  Reference value: {opt_value}, Integration bounds: [{lower_bound}, {upper_bound}]"
            )

        # Create the single-parameter function.
        single_param_fn = make_single_param_criterion_function(
            selection_criterion, covparam, param_index
        )
        # Compute fast stats using the grid-based method.
        mean_val, var_val, quantiles, mode_val = fast_univariate_stats(
            single_param_fn, lower_bound, upper_bound, n_points=n_points
        )

        parameter_stats[param_index] = {
            "mean": float(mean_val),
            "variance": float(var_val),
            "quantile_0.1": quantiles["0.1"],
            "quantile_0.25": quantiles["0.25"],
            "quantile_0.5": quantiles["0.5"],
            "quantile_0.75": quantiles["0.75"],
            "quantile_0.9": quantiles["0.9"],
            "mode": float(mode_val),
        }

        if verbose:
            print(
                f"  Stats for param {param_index} -> Mean: {mean_val:.4f}, Var: {var_val:.4f}, "
                f"Quantiles: [0.1: {quantiles['0.1']:.4f}, 0.25: {quantiles['0.25']:.4f}, "
                f"0.5: {quantiles['0.5']:.4f}, 0.75: {quantiles['0.75']:.4f}, 0.9: {quantiles['0.9']:.4f}], "
                f"Mode: {mode_val:.4f}"
            )

    # Convert the parameter_stats dictionary into a DataFrame.
    # Create a list of row data and corresponding row names.
    rows = []
    row_names = []
    # Define column order.
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
    for param_index in sorted(parameter_stats.keys()):
        stats = parameter_stats[param_index]
        row = [stats[col] for col in col_names]
        rows.append(row)
        row_names.append(f"param_{param_index}")

    parameter_stats_df = DataFrame(np.array(rows), col_names, row_names)

    # Compute the Fisher information matrix using the provided model.
    fisher_information = model.fisher_information(xi, covparam, epsilon=1e-3)

    if verbose:
        print("\nFisher Information Matrix:")
        print(fisher_information)

    result_dict = {
        "parameter_statistics": parameter_stats_df,
        "fisher_information": fisher_information,
    }
    return result_dict


def selection_criterion_statistics(
    info=None,
    model=None,
    xi=None,
    selection_criterion=None,
    covparam=None,
    ind=None,
    param_box=None,
    delta=5.0,
    verbose=False,
):
    """Compute statistics on the selection criterion viewed as a
    negative log pdf, for the specified dimensions of covparam, and
    compute the Fisher information matrix using
    model.fisher_information(xi, covparam, epsilon=1e-3).

    For each parameter in `ind`, the function:
      - Creates a single-parameter function (via make_single_param_criterion_function).
      - Builds a pseudo-density with log_pdf = -selection_criterion (so that lower criterion values get higher weight).
      - Computes the weighted mean, variance, and quantiles at 0.1, 0.25, 0.5, 0.75, 0.9.
      - Computes the mode, defined as the value that minimizes the selection criterion via bounded minimization.

    Parameters
    ----------
    info : object, optional
        If provided—and if selection_criterion or covparam is not supplied explicitly—they
        are taken from this object (attributes selection_criterion_nograd and covparam).
    model : object, required
        The model object, assumed to implement a method fisher_information(xi, covparam, epsilon).
    xi : array-like, required
        Input data points to be passed to model.fisher_information.
    selection_criterion : callable, optional
        A function that accepts a covparam vector and returns a scalar value.
    covparam : array-like, optional
        The covariance parameters vector.
    ind : list or array-like, optional
        List of indices (dimensions) for which statistics are computed. If not provided, all dimensions are used.
    param_box : 2D array-like, optional
        Two-row array with min and max values for each parameter; if provided, these are used
        to set integration bounds. Otherwise, bounds are set to:
            [opt_value - delta, opt_value + delta],
        where opt_value is the reference value in covparam.
    delta : float, optional
        Default range ±delta around the reference parameter value if param_box is not provided.
    verbose : bool, optional
        If True, prints detailed compressed information during computation.

    Returns
    -------
    result_dict : dict
        Dictionary with two keys:
          "parameter_statistics" : DataFrame
              A DataFrame where each row corresponds to a parameter. Columns are:
                "mean", "variance", "quantile_0.1", "quantile_0.25",
                "quantile_0.5", "quantile_0.75", "quantile_0.9", "mode".
          "fisher_information" : array-like
              The Fisher information matrix as computed by model.fisher_information(xi, covparam, epsilon=1e-3).

    """
    # If info is provided, set defaults.
    if info is not None:
        if selection_criterion is None:
            selection_criterion = info.selection_criterion_nograd
        if covparam is None:
            covparam = info.covparam
        if model is None and hasattr(info, "model"):
            model = info.model
        if xi is None and hasattr(info, "xi"):
            xi = info.xi

    if covparam is None:
        raise ValueError("covparam must be provided either directly or via info.")
    if model is None:
        raise ValueError("model must be provided either directly or via info.")
    if xi is None:
        raise ValueError("xi must be provided either directly or via info.")

    covparam = gnp.asarray(covparam)
    n_params = covparam.shape[0]
    # If no indices specified, use all dimensions.
    if ind is None:
        ind = list(range(n_params))

    parameter_stats = {}

    for param_index in ind:
        opt_value = covparam[param_index]
        if param_box is not None:
            lower_bound = float(param_box[0][param_index])
            upper_bound = float(param_box[1][param_index])
        else:
            lower_bound = opt_value - delta
            upper_bound = opt_value + delta

        if verbose:
            print(f"\nProcessing parameter index {param_index}:")
            print(
                f"  Reference value: {opt_value}, Integration bounds: [{lower_bound}, {upper_bound}]"
            )

        # Create the single-parameter function for the current dimension.
        single_param_fn = make_single_param_criterion_function(
            selection_criterion, covparam, param_index
        )
        # Define a pseudo log-pdf as minus the selection criterion (lower criterion gives higher weight).
        log_pdf = lambda x: -single_param_fn(x)
        # Build the pseudo-distribution.
        dist = Unnormalized1DDistribution(log_pdf, bounds=(lower_bound, upper_bound))

        # Compute weighted statistics.
        mean_val = dist.mean()
        var_val = dist.var()
        quantile_0_1 = dist.quantile(0.1)
        quantile_0_25 = dist.quantile(0.25)
        quantile_0_5 = dist.quantile(0.5)
        quantile_0_75 = dist.quantile(0.75)
        quantile_0_9 = dist.quantile(0.9)

        # Determine the mode as the covparam value.
        mode_val = covparam[param_index]

        parameter_stats[param_index] = {
            "mean": float(mean_val),
            "variance": float(var_val),
            "quantile_0.1": float(quantile_0_1),
            "quantile_0.25": float(quantile_0_25),
            "quantile_0.5": float(quantile_0_5),
            "quantile_0.75": float(quantile_0_75),
            "quantile_0.9": float(quantile_0_9),
            "mode": float(mode_val),
        }

        if verbose:
            print(
                f"  Stats for param {param_index} -> Mean: {mean_val:.4f}, Var: {var_val:.4f}, "
                f"Quantiles: [0.1: {quantile_0_1:.4f}, 0.25: {quantile_0_25:.4f}, 0.5: {quantile_0_5:.4f}, "
                f"0.75: {quantile_0_75:.4f}, 0.9: {quantile_0_9:.4f}], Mode: {mode_val:.4f}"
            )

    # Convert the parameter_stats dictionary into a DataFrame.
    rows = []
    row_names = []
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
    for param_index in sorted(parameter_stats.keys()):
        stats = parameter_stats[param_index]
        row = [stats[col] for col in col_names]
        rows.append(row)
        row_names.append(f"param_{param_index}")

    parameter_stats_df = DataFrame(np.array(rows), col_names, row_names)

    # Compute the Fisher information matrix using the provided model.
    fisher_information = model.fisher_information(xi, covparam, epsilon=1e-3)

    if verbose:
        print("\nFisher Information Matrix:")
        print(fisher_information)

    result_dict = {
        "parameter_statistics": parameter_stats_df,
        "fisher_information": fisher_information,
    }
    return result_dict


# ============================================================
# Performance evaluation and model metrics
# ============================================================


def compute_performance(
    model, xi, zi, loo=True, loo_res=None, xtzt=None, zpmzpv=None, compute_pit=False
):
    """Compute performance metrics of the GP model.

    Parameters
    ----------
    model : instance of GP model
        The GP model used to make predictions.
    xi : ndarray of shape (n, d)
        The input data used to fit the GP model.
    zi : ndarray of shape (n,)
        The target data used to fit the GP model.
    loo : bool, optional
        Whether or not to compute the leave-one-out (LOO) metrics. Default is True.
    loo_res : tuple, optional
        The output of the `loo` method of the GP model if already computed. Default is None.
    xtzt : tuple, optional
        The test input data and corresponding targets to be used for computing test set metrics. Default is None.
    zpmzpv : tuple, optional
        The predicted mean and variance on the test set data. Default is None.
    compute_pit : bool, optional
        Whether to compute probability integral transform (PIT)

    Returns
    -------
    perf : dict

        A dictionary containing the computed performance metrics. The
        metrics that are computed depend on the input arguments.
        If `loo` is True, the following keys are present:
        - 'data_tss': total sum of squares in the data.
        - 'loo_press': predictive residual error sum of squares in the
          leave-one-out predictions.
        - 'loo_Q2': coefficient of determination for the leave-one-out predictions.
        - 'loo_log10ratio': logarithm of the predictive
          residual error sum of squares to the total sum of squares in
          the leave-one-out predictions.
        - 'loo_pit': probability integral transform (PIT) of the
          leave-one-out predictions.
        If `xtzt` and `zpmzpv` are not None, the following keys are present:
        - 'test_tss': total sum of squares in the test set predictions.
        - 'test_rss': residual error sum of squares in
          the test set predictions.
        - 'test_R2': coefficient of determination for the test set predictions.
        - 'test_log10ratio': logarithm of the residual error sum of squares
          to the total sum of squares in the test set predictions.
        - 'test_pit': PIT of the test set predictions.
    """
    xi = gnp.asarray(xi)
    zi = gnp.asarray(zi)

    if loo and loo_res == None:
        zloom, zloov, eloo = model.loo(xi, zi)
    elif loo and type(loo_res) is tuple:
        zloom, zloov, eloo = loo_res

    test_set = False
    if xtzt is not None:
        test_set = True
        xt, zt = xtzt
        xt = gnp.asarray(xt)
        zt = gnp.asarray(zt)
    if test_set and zpmzpv is None:
        zpm, zpv = model.predict(xi, zi, xt)
    elif xtzt is not None and zpmzpv is not None:
        zpm, zpv = zpmzpv
        zpm = gnp.asarray(zpm)
        zpv = gnp.asarray(zpv)

    perf = {}

    if loo:
        # total sum of squares
        perf["data_tss"] = gnp.norm(zi - gnp.mean(zi), ord=2) ** 2
        # Predictive residual Error sum of squares
        perf["loo_press"] = gnp.norm(eloo, ord=2) ** 2
        # Q2
        perf["loo_Q2"] = 1 - perf["loo_press"] / perf["data_tss"]
        # L2err
        perf["loo_log10ratio"] = gnp.log10(perf["loo_press"] / perf["data_tss"])
        # PIT
        if compute_pit:
            perf["loo_pit"] = gnp.normal.cdf(zi, loc=zloom, scale=gnp.sqrt(zloov))

    if test_set:
        perf["test_tss"] = gnp.norm(zt - gnp.mean(zt), ord=2) ** 2
        perf["test_rss"] = gnp.norm(zt - zpm, ord=2) ** 2
        perf["test_R2"] = 1 - perf["test_rss"] / perf["test_tss"]
        perf["test_log10ratio"] = gnp.log10(perf["test_rss"] / perf["test_tss"])
        if compute_pit:
            perf["test_pit"] = gnp.normal.cdf(zt, loc=zpm, scale=gnp.sqrt(zpv))

    return perf


def perf(model, xi, zi, loo=True, loo_res=None, xtzt=None, zpmzpv=None):
    perf = compute_performance(model, xi, zi, loo, loo_res, xtzt, zpmzpv)
    perf_disp = perf
    try:
        perf_disp.pop("loo_pit")
    except:
        pass
    try:
        perf_disp.pop("test_pit")
    except:
        pass
    print("[Prediction performances]")
    pretty_print_dictionnary(perf_disp)


def diag(
    model,
    info_select_parameters,
    xi,
    zi,
    *,
    model_type="linear_mean_matern_anisotropic",
    param_obj=None,
):
    """Run model diagnosis and display the results.

    Parameters
    ----------
    model : object
        GP Model object.
    info_select_parameters : object
        Information object containing the parameter selection process.
    xi : array-like
        Input data matrix.
    zi : array-like
        Output data matrix.
    model_type : str, optional
        Type of the model (default "linear_mean_matern_anisotropic").
    param_obj : Param, optional
        If provided, this Param object is used directly (no reconstruction).
    """
    md = modeldiagnosis_init(
        model,
        info_select_parameters,
        model_type=model_type,
        param_obj=param_obj,
    )
    model_diagnosis_disp(md, xi, zi, model_type=model_type)


def modeldiagnosis_init(
    model,
    info,
    *,
    model_type="linear_mean_matern_anisotropic",
    param_obj=None,
):
    """Build model diagnosis based on the provided model/info.

    Parameters
    ----------
    model : object
        Model object (must expose `meanparam` and `covparam`).
    info : object
        Parameter selection info. If it has an attribute `bounds`
        shaped (n_params, 2) in the optimizer's normalized space,
        those bounds will be applied to the returned Param.
        The expected optimizer param order is [meanparam, covparam].
    model_type : str
        Type of the model. Used only when `param_obj` is not provided.
    param_obj : Param, optional
        If provided, use this Param directly (bounds from `info.bounds`
        will still be applied if possible).

    Returns
    -------
    dict
        Model diagnosis information.
    """
    md = {
        "optim_info": info,
        "param_selection": {},
        "parameters": {},
        "param_obj": None,
        "loo": {},
        "data": {},
    }

    md["param_selection"] = {
        "cvg_reached": info.success,
        "optimal_val": info.best_value_returned,
        "n_evals": info.nfev,
        "time": info.total_time,
        "initial_val": float(info.selection_criterion(info.initial_params)),
        "final_val": info.fun,
    }

    # Helper: apply a (k,2) bounds array to the covariance portion of a Param
    def _apply_cov_bounds_to_param(param_obj, cov_bounds):
        """
        param_obj : Param
            Must contain cov entries in order (sigma2, then rhos, etc.)
        cov_bounds : array_like (k,2)
            Bounds for the covariance vector in optimizer's normalized space.
            (-inf, +inf) entries are treated as "no bound" and stored as None.
        """
        import numpy as _np

        cov_bounds = _np.asarray(cov_bounds, dtype=float)
        # Find indices in Param that belong to 'covparam' (in the same order)
        cov_inds = [j for j, p in enumerate(param_obj.paths) if p and p[0] == "covparam"]
        if len(cov_inds) != cov_bounds.shape[0]:
            # If shapes mismatch, bail out silently rather than corrupting Param
            # (could also raise, but diagnosis should be robust).
            return param_obj

        for dst_idx, (lo, hi) in zip(cov_inds, cov_bounds):
            # Treat fully unbounded as None; keep one-sided bounds if present
            if _np.isinf(lo) and _np.isinf(hi):
                param_obj.bounds[dst_idx] = None
            else:
                param_obj.bounds[dst_idx] = (float(lo), float(hi))
        return param_obj

    # If caller did not pass a Param, build one from the covariance vector.
    if param_obj is None:
        covparam = gnp.asarray(model.covparam)
        param_builders = {
            "linear_mean_matern_anisotropic": param_from_covparam_anisotropic,
            "linear_mean_matern_anisotropic_noisy": param_from_covparam_anisotropic_noisy,
        }
        builder = param_builders.get(model_type, None)
        if builder is None:
            raise ValueError(f"Unknown model type: {model_type}")
        # Build WITHOUT bounds first; we will inject bounds from info.bounds below.
        param_obj = builder(covparam, None, None, name_prefix="")

    # If `info.bounds` is present, project it onto the covariance part of the Param.
    bounds_arr = getattr(info, "bounds", None)
    if bounds_arr is not None:
        # optimizer order is [meanparam, covparam]; determine mean length
        if getattr(model, "meanparam", None) is None:
            mpl = 0
        else:
            mpl = int(gnp.asarray(model.meanparam).reshape(-1).shape[0])
        cov_len = int(gnp.asarray(model.covparam).reshape(-1).shape[0])
        bounds_arr = gnp.asarray(bounds_arr)
        if bounds_arr.ndim == 2 and bounds_arr.shape[1] == 2 and bounds_arr.shape[0] >= mpl + cov_len:
            cov_bounds = bounds_arr[mpl: mpl + cov_len]
            param_obj = _apply_cov_bounds_to_param(param_obj, cov_bounds)

    md["parameters"] = param_obj.to_simple_dict()
    md["param_obj"] = param_obj
    return md


def model_diagnosis_disp(md, xi, zi, model_type="linear_mean_matern_anisotropic"):
    """Display model diagnosis information.

    Parameters
    ----------
    md : dict
        Model diagnosis information (must contain 'param_obj').
    xi : array-like
        Input data matrix.
    zi : array-like
        Output data matrix.
    model_type : str
        Type of the model.
    """
    print("[Model diagnosis]")
    print("  * Parameter selection")
    pretty_print_dictionnary(md["param_selection"])

    print("  * Parameters")
    # Uses Param.__repr__ to display as a table-like block
    print("\n".join("    " + line for line in str(md["param_obj"]).splitlines()))

    print("  * Data")
    print("    {:>0}: {:d}".format("count", zi.shape[0]))
    print("    -----")

    # Scale factors derived from the provided/built Param
    param_values = np.array(list(md["parameters"].values()))

    # zi
    if getattr(zi, "ndim", 1) == 1:
        rownames = ["zi"]
    else:
        rownames = [f"zi_{j}" for j in range(zi.shape[1])]
    df_zi = describe_array(zi, rownames, 1 / param_values[0])

    # xi
    n, d = xi.shape
    rownames = [f"xi_{j}" for j in range(d)]
    df_xi = describe_array(xi, rownames, 1 / param_values[-d:])

    # zi + xi
    print(df_zi.concat(df_xi))


# ============================================================
# Visualization and plotting tools
# ============================================================


def plot_pit_ecdf(pit, fig=None):
    """Plot the empirical cumulative distribution function (ECDF) of a Probability
    Integral Transform (PIT) vector.

    Parameters
    ----------
    pit : gnp.ndarray, shape (n,)
        An array of PIT values.
    fig : matplotlib.figure.Figure, optional
        Matplotlib figure object to plot the PIT ECDF on. If None, a
        new figure is created. Default is None.

    Returns
    -------
    None
    """
    n = pit.shape[0]
    p = gnp.concatenate((gnp.array([0]), gnp.linspace(0, 1, n)))
    pit_sorted = gnp.concatenate((gnp.array([0.0]), gnp.sort(pit)))

    if fig is None:
        plt.figure()
    plt.step(pit_sorted, p)
    plt.plot([0, 1], [0, 1])
    plt.title("PIT (Probability Integral Transform) ECDF")
    plt.show()


def plot_selection_criterion_crosssections(
    *,
    info=None,
    selection_criterion=None,
    selection_criteria=None,
    covparam=None,
    n_points=100,
    param_names=None,
    criterion_name="selection criterion",
    criterion_names=None,
    criterion_name_full="Cross sections for negative log restricted likelihood",
    ind=None,
    ind_pooled=None,
    param_box=None,
    param_box_pooled=None,
    delta=5.0,
):
    """
    Plot cross-sections of one or several selection criteria.

    Each selected parameter is varied while the others remain fixed and the criterion is computed.
    Two options allow you to specify the variation range:
      - param_box: a 2D array with two rows (first row for min values, second row for max values)
                   for the individual plots.
      - param_box_pooled: a similar 2D array for the pooled plot.
    If neither is provided, the default range for each parameter is [-delta, +delta].
    These ranges are used exclusively for their respective plots.

    Parameters:
        info: Object that may hold optimal parameters (covparam) and a function
              selection_criterion_nograd(param). If not provided, you must supply
              selection_criterion or selection_criteria and covparam explicitly.
        selection_criterion: A single callable of the form f(param) -> scalar.
                             If info is given and selection_criteria is not provided,
                             it defaults to info.selection_criterion_nograd.
        selection_criteria: Optional list or tuple of several callables, each of the
                            form f(param) -> scalar. If provided, overrides
                            selection_criterion.
        covparam: Optional override for info.covparam (if info is provided).
                  If info is None, covparam must not be None.
        n_points: Number of points in each cross-section.
        param_names: List of parameter names.
        criterion_name: Label or base label for criterion curves (if one criterion).
        criterion_names: List of labels for each criterion (if multiple criteria).
        criterion_name_full: Figure title.
        ind: List of indices for individual subplots.
        ind_pooled: List of indices for the pooled plot.
        param_box: 2D array of shape (2, n_params) for individual plot bounds.
        param_box_pooled: 2D array of shape (2, n_params) for pooled plot bounds.
        delta: Default +/- range to scan around the reference parameter.

    If multiple criteria are provided, all will be plotted on the same axes for
    each parameter index. Legends will show corresponding names from criterion_names.

    Displays the plots.
    """
    # Decide which interpreter mode we are in, to possibly use interactive mode.
    try:
        interpreter = sys.ps1
    except AttributeError:
        interpreter = sys.flags.interactive
    if interpreter:
        plt.ion()

    if selection_criteria is None:
        if selection_criterion is None and info is None:
            raise ValueError("Provide at least one selection criterion.")
        if selection_criterion is None:
            selection_criterion = info.selection_criterion_nograd
        selection_criteria = (selection_criterion,)
    else:
        selection_criteria = tuple(selection_criteria)

    n_crit = len(selection_criteria)

    if criterion_names is None:
        if isinstance(criterion_name, (list, tuple)):
            criterion_names = list(criterion_name)
        else:
            criterion_names = [
                f"{criterion_name} #{k}" if n_crit > 1 else criterion_name
                for k in range(n_crit)
            ]
    if len(criterion_names) != n_crit:
        raise ValueError("criterion_names length must match number of criteria.")

    if info is None:
        if covparam is None:
            raise ValueError("covparam must be supplied when info is None.")
        param_opt = gnp.asarray(covparam)
    else:
        param_opt = gnp.asarray(covparam if covparam is not None else info.covparam)

    n_params = param_opt.shape[0]

    if ind is None and ind_pooled is None:
        ind = list(range(n_params))

    def get_p_values(param_idx, opt_val, box):
        if box is not None:
            lo = float(box[0][param_idx])
            hi = float(box[1][param_idx])
        else:
            lo = opt_val - delta
            hi = opt_val + delta
        return gnp.linspace(lo, hi, n_points)

    # Plot individual cross-sections if 'ind' is specified
    if ind is not None:
        n_ind = len(ind)
        fig, axes = plt.subplots(n_ind, 1, figsize=(8, min(9, 3 * n_ind)))
        if n_ind == 1:
            axes = [axes]
        for idx, param_idx in enumerate(ind):
            opt_value = param_opt[param_idx]
            p_values = get_p_values(idx, opt_value, param_box)
            crit_values = gnp.zeros((n_crit, n_points))
            for j, x_val in enumerate(p_values):
                param = gnp.copy(param_opt)
                param[param_idx] = x_val
                for k, f in enumerate(selection_criteria):
                    v = f(param)
                    crit_values[k, j] = v
            ax = axes[idx]
            for k in range(n_crit):
                ax.plot(p_values, crit_values[k], label=criterion_names[k])
            ax.axvline(
                opt_value,
                color="red",
                linestyle="--",
                label="reference" if covparam is not None else "optimal",
            )
            name = (
                param_names[param_idx]
                if param_names is not None and param_idx < len(param_names)
                else f"param {param_idx}"
            )
            ax.set_ylabel("criterion value")
            if idx == n_ind - 1:
                ax.set_xlabel("parameter value")
            if idx == 0:
                ax.legend()
            ax.set_title(name)

        fig.suptitle(criterion_name_full, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Plot pooled cross-sections if 'ind_pooled' is specified
    if ind_pooled is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, param_idx in enumerate(ind_pooled):
            opt_value = param_opt[param_idx]
            p_values = get_p_values(idx, opt_value, param_box_pooled)
            crit_values = gnp.zeros((n_crit, n_points))
            for j, x_val in enumerate(p_values):
                param = gnp.copy(param_opt)
                param[param_idx] = x_val
                for k, f in enumerate(selection_criteria):
                    v = f(param)
                    crit_values[k, j] = v

            name = (
                param_names[param_idx]
                if param_names is not None and param_idx < len(param_names)
                else f"param {param_idx}"
            )
            for k in range(n_crit):
                ax.plot(p_values, crit_values[k], label=f"{name} - {criterion_names[k]}")
            ax.axvline(opt_value, color="red", linestyle="--")
        ax.set_xlabel("parameter value")
        ax.set_ylabel("criterion value")
        ax.set_title(f"Pooled cross sections of {criterion_name_full}")
        ax.legend()
        plt.tight_layout()
        plt.show()

# def plot_selection_criterion_crosssections(
#     info=None,
#     selection_criterion=None,
#     covparam=None,
#     n_points=100,
#     param_names=None,
#     criterion_name="selection criterion",
#     criterion_name_full="Cross sections for negative log restricted likelihood",
#     ind=None,
#     ind_pooled=None,
#     param_box=None,
#     param_box_pooled=None,
#     delta=5.0,
# ):
#     """
#     Plot cross-sections of the selection criterion.

#     Each selected parameter is varied while the others remain fixed and the criterion is computed.
#     Two options allow you to specify the variation range:
#       - param_box: a 2D array with two rows (first row for min values, second row for max values)
#                    for the individual plots.
#       - param_box_pooled: a similar 2D array for the pooled plot.
#     If neither is provided, the default range for each parameter is [-delta, +delta.0].
#     These ranges are used exclusively for their respective plots.

#     Parameters:
#         info: Object that may hold optimal parameters (covparam) and a function
#               selection_criterion_nograd(param).
#               If not provided, you must supply selection_criterion and covparam explicitly.
#         selection_criterion: A callable of the form f(param) -> scalar. If info is given,
#                              it defaults to info.selection_criterion_nograd.
#         covparam: Optional override for info.covparam (if info is provided).
#                   If info is None, covparam must not be None.
#         n_points: Number of points in each cross-section.
#         param_names: List of parameter names.
#         criterion_name: Label for criterion curves.
#         criterion_name_full: Figure title.
#         ind: List of indices for individual subplots.
#         ind_pooled: List of indices for the pooled plot.
#         param_box: 2D array of shape (2, n_params) for individual plot bounds.
#         param_box_pooled: 2D array of shape (2, n_params) for pooled plot bounds.
#         delta: Default +/- range to scan around the reference parameter.

#     Displays the plots.
#     """

#     # Decide which interpreter mode we are in, to possibly use interactive mode.
#     try:
#         interpreter = sys.ps1
#     except AttributeError:
#         interpreter = sys.flags.interactive
#     if interpreter:
#         plt.ion()

#     # ---------------------
#     # Handle info vs. no info
#     # ---------------------
#     if info is None:
#         # Must have selection_criterion and covparam
#         if selection_criterion is None:
#             raise ValueError(
#                 "selection_criterion must be provided if info is not given."
#             )
#         if covparam is None:
#             raise ValueError("covparam must be provided if info is not given.")
#         param_opt = gnp.asarray(covparam)
#     else:
#         # If an info object is provided, default from it:
#         if selection_criterion is None:
#             selection_criterion = info.selection_criterion_nograd
#         if covparam is None:
#             param_opt = gnp.asarray(info.covparam)
#         else:
#             param_opt = gnp.asarray(covparam)

#     n_params = param_opt.shape[0]

#     # If neither ind nor ind_pooled are provided, use all parameters.
#     if ind is None and ind_pooled is None:
#         ind = list(range(n_params))

#     # Helper function to generate p values, using provided box or [opt_value - delta, opt_value + delta].
#     def get_p_values(param_index, opt_value, box):
#         if box is not None:
#             lower_bound = float(box[0][param_index])
#             upper_bound = float(box[1][param_index])
#         else:
#             lower_bound = opt_value - delta
#             upper_bound = opt_value + delta
#         return gnp.linspace(lower_bound, upper_bound, n_points)

#     # ----------------------------------------------------------
#     # Plot individual cross-sections if 'ind' is specified
#     # ----------------------------------------------------------
#     if ind is not None:
#         n_ind = len(ind)
#         fig, axes = plt.subplots(n_ind, 1, figsize=(8, min(9, 3 * n_ind)))
#         if n_ind == 1:
#             axes = [axes]
#         for idx, param_index in enumerate(ind):
#             opt_value = param_opt[param_index]
#             p_values = get_p_values(idx, opt_value, param_box)
#             crit_values = gnp.zeros(n_points)
#             for j, x_val in enumerate(p_values):
#                 param = gnp.copy(param_opt)
#                 param = gnp.set_elem_1d(param, param_index, x_val)
#                 crit = selection_criterion(param)
#                 crit_values = gnp.set_elem_1d(crit_values, j, crit)

#             ax = axes[idx]
#             ax.plot(p_values, crit_values, label=criterion_name)
#             ax.axvline(
#                 opt_value,
#                 color="red",
#                 linestyle="--",
#                 label=("reference" if covparam is not None else "optimal"),
#             )
#             # Label for the parameter
#             name = (
#                 param_names[param_index]
#                 if (param_names is not None and param_index < len(param_names))
#                 else f"param {param_index}"
#             )
#             ax.set_ylabel("Criterion value")
#             if idx == n_ind - 1:
#                 ax.set_xlabel("Param value")
#             if idx == 0:
#                 ax.legend()
#             ax.set_title(f"{name}")
#         fig.suptitle(criterion_name_full, fontsize=12)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()

#     # ----------------------------------------------------------
#     # Plot pooled cross-sections if 'ind_pooled' is specified
#     # ----------------------------------------------------------
#     if ind_pooled is not None:
#         fig, ax = plt.subplots(figsize=(8, 6))
#         for idx, param_index in enumerate(ind_pooled):
#             opt_value = param_opt[param_index]
#             p_values = get_p_values(0, opt_value, param_box_pooled)
#             crit_values = gnp.zeros(n_points)
#             for j, x_val in enumerate(p_values):
#                 param = gnp.copy(param_opt)
#                 param = gnp.set_elem_1d(param, param_index, x_val)
#                 crit = selection_criterion(param)
#                 crit_values = gnp.set_elem_1d(crit_values, j, crit)

#             name = (
#                 param_names[param_index]
#                 if (param_names is not None and param_index < len(param_names))
#                 else f"param {param_index}"
#             )
#             ax.plot(p_values, crit_values, label=name)
#             if covparam is None and idx == len(ind_pooled) - 1:
#                 label_ref = "optimal"
#             elif idx == len(ind_pooled) - 1:
#                 label_ref = "ref"
#             else:
#                 label_ref = None
#             ax.axvline(opt_value, color="red", linestyle="--", label=label_ref)
#         ax.set_xlabel("Parameter value")
#         ax.set_ylabel("Criterion value")
#         ax.set_title(f"Pooled cross sections of {criterion_name}")
#         ax.legend()
#         plt.tight_layout()
#         plt.show()


def plot_selection_criterion_2d(
    model,
    info,
    param_indices=(0, 1),
    param_names=None,
    criterion_name="selection criterion",
):
    """Plot selection criterion profile for any two parameters in covparam.

    Parameters
    ----------
    model : object
        Model object.
    info : object
        Information object containing the parameter selection process.
    param_indices : tuple, optional
        Indices of the two parameters to plot (default is (0,
        1)). First parameter is assumed to be the log of the variance
        parameter. Other parameters are the log of inverse
        lengthscales. #FIXME
    param_names : list, optional
        Names of the two parameters for labeling the axes (default is None).
    criterion_name : string, optional
        Name of the selection criterion to be displayed in the title

    """
    n = 130
    tic = time.time()

    def print_progress(i):
        elapsed_time = time.time() - tic
        average_time_per_iteration = elapsed_time / (i + 1)
        remaining_time = average_time_per_iteration * (n - (i + 1))
        percentage = (i + 1) / n * 100
        print(
            f"       Progress: {percentage:.2f}% | time remaining: {remaining_time:.1f}s",
            end="\r",
        )

    def print_final_time():
        elapsed_time = time.time() - tic
        print(f"       Progress: 100% complete | Total time: {elapsed_time:.3f}s")
        print(f"       number of evaluations: {n * n}")

    print(f"  ***  Computing {criterion_name} profile for plotting...")

    param_1_idx, param_2_idx = param_indices

    # Initialize param1 and param2 based on their indices (standard deviation or scale parameter)
    param_1_0 = math.exp(
        model.covparam[param_1_idx] / 2
        if param_1_idx == 0
        else -model.covparam[param_1_idx]
    )
    param_2_0 = math.exp(
        model.covparam[param_2_idx] / 2
        if param_2_idx == 0
        else -model.covparam[param_2_idx]
    )

    f = 4  # multiplying factor
    param_1 = np.logspace(
        math.log10(param_1_0) - math.log10(f), math.log10(param_1_0) + math.log(f), n
    )
    param_2 = np.logspace(
        math.log10(param_2_0) - math.log10(f), math.log10(param_2_0) + math.log(f), n
    )

    param_1_mesh, param_2_mesh = np.meshgrid(param_1, param_2)
    log_param_1 = (
        np.log(param_1_mesh**2) if param_1_idx == 0 else np.log(1 / param_1_mesh)
    )
    log_param_2 = (
        np.log(param_2_mesh**2) if param_2_idx == 0 else np.log(1 / param_2_mesh)
    )

    selection_criterion = info.selection_criterion_nograd
    selection_criterion_values = np.zeros((n, n))

    covparam = gnp.copy(info.covparam)
    for i in range(n):
        print_progress(i)
        for j in range(n):
            covparam[param_1_idx] = log_param_1[i, j]
            covparam[param_2_idx] = log_param_2[i, j]
            selection_criterion_values[i, j] = selection_criterion(covparam)

    selection_criterion_values = np.nan_to_num(selection_criterion_values, copy=False)
    print_final_time()

    shift_criterion = True
    shift = -np.min(selection_criterion_values) if shift_criterion else 0

    # Plot the selection criterion profile
    plt.figure()
    plt.contourf(
        np.log10(param_1_mesh),
        np.log10(param_2_mesh),
        np.log10(np.maximum(1e-2, selection_criterion_values + shift)),
    )
    plt.plot(
        0.5 * np.log10(np.exp(info.covparam[param_1_idx])),
        -np.log10(np.exp(info.covparam[param_2_idx])),
        "ro",
    )
    plt.plot(
        0.5 * np.log10(np.exp(info.covparam0[param_1_idx])),
        -np.log10(np.exp(info.covparam0[param_2_idx])),
        "bo",
    )

    # Define axis labels (use names if provided)
    x_label = param_names[0] if param_names else f"Parameter {param_1_idx} (log10)"
    y_label = param_names[1] if param_names else f"Parameter {param_2_idx} (log10)"

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'log10 of the {"shifted " if shift_criterion else ""}{criterion_name}')
    plt.colorbar()
    plt.show()


def plot_selection_criterion_sigma_rho(
    model, info, criterion_name="negative log restricted likelihood"
):
    """
    Specific case of selection criterion plotting for sigma (param 0) and rho (param 1).

    Parameters
    ----------
    model : object
        Model object.
    info : object
        Information object containing the parameter selection process.
    criterion_name : string, optional
        Name of the selection criterion to be displayed in the title
    """
    # Use the generalized function to plot sigma and rho with custom names
    plot_selection_criterion_2d(
        model,
        info,
        param_indices=(0, 1),
        param_names=["sigma (log10)", "rho (log10)"],
        criterion_name=criterion_name,
    )


# ============================================================
# Utilities and Data Description
# ============================================================


def sigma_rho_from_covparam(covparam):
    """Extract sigma and rho parameters from the covariance parameters.

    Parameters
    ----------
    covparam : array-like
        Covariance parameters.

    Returns
    -------
    dict
        Dictionary containing sigma and rho values.
    """
    pdict = {}
    pdict["sigma"] = gnp.exp(0.5 * covparam[0])
    for i in range(covparam.shape[0] - 1):
        k = "rho{:d}".format(i)
        v = gnp.exp(-covparam[i + 1])
        pdict[k] = v

    return pdict


def describe_array(x, rownames, sigma_factor=None):
    """Create a DataFrame containing descriptive statistics for the given data.

    Parameters
    ----------
    x : array-like
        Input data matrix.
    rownames : list
        List of row names for the DataFrame.
    sigma_factor : float, optional
        Normalizing factor to compute the 'delta/sigma' column, by default None.

    Returns
    -------
    DataFrame
        DataFrame with descriptive statistics.
    """
    x = np.array(x)
    if sigma_factor is None:
        n_descriptors = 5
        colnames = ["mean", "std", "min", "max", "delta"]
    else:
        n_descriptors = 6
        colnames = ["min", "max", "delta", "mean", "std", "delta/sigma"]
    dim = 1 if x.ndim == 1 else x.shape[1]

    data = np.empty((dim, n_descriptors))

    data[:, 0] = np.min(x, axis=0)
    data[:, 1] = np.max(x, axis=0)
    data[:, 2] = data[:, 1] - data[:, 0]
    data[:, 3] = np.mean(x, axis=0)
    data[:, 4] = np.std(x, axis=0)

    if sigma_factor is not None:
        data[:, 5] = data[:, 4] * sigma_factor

    return DataFrame(data, colnames, rownames)


def pretty_print_dictionnary(d, fp=4):
    """Print a dictionary with formatted values.

    Parameters
    ----------
    d : dict
        The dictionary to be printed.
    fp : int, optional
        Number of decimal places for floating-point values, by default 4.
    """
    max_key_length = max(15, max(len(str(k)) for k in d.keys()) + 2)

    for k, v in d.items():
        if not gnp.isscalar(v):
            v = v.item()
        if isinstance(v, float):
            s = f"{{:>{max_key_length}s}}: {{:s}}"
            print(s.format(k, ftos(v, fp)))
        else:
            print(f"{k:>{max_key_length}s}: {v}")
