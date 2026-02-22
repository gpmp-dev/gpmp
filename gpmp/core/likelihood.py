# gpmp/core/likelihood.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Negative (restricted) log-likelihood computations.

This module implements the numerical routines used by `gpmp.core.Model`
for likelihood evaluation in the zero-mean, parameterized-mean, and
linear_predictor cases.
"""
import gpmp.num as gnp
from .linalg import compute_contrast_matrix, compute_contrast_covariance


def negative_log_likelihood_zero_mean(model, covparam, xi, zi):
    """Computes the negative log-likelihood of the Gaussian process model with zero
    mean.

    This function is specific to the zero-mean case, and the negative log-likelihood
    is computed based on the provided covariance function and parameters.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `covariance`.
    covparam : gnp.array
        Parameters for the covariance function. This array contains the hyperparameters
        required by the chosen covariance function.
    xi : ndarray(ni,d)
        Locations of the data points in the input space, where ni is the number of data points
        and d is the dimensionality of the input space.
    zi : ndarray(ni, )
        Observed values corresponding to each data point in xi.

    Returns
    -------
    nll : scalar
        Negative log-likelihood of the observed data given the model and covariance parameters.
    """
    K = model.covariance(xi, xi, covparam)
    n = K.shape[0]
    try:
        Kinv_zi, C = gnp.cholesky_solve(K, zi)
    except RuntimeError:
        return gnp.safe_inf()
    norm2 = gnp.einsum("i..., i...", zi, Kinv_zi)
    ldetK = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))
    L = 0.5 * (n * gnp.log(2.0 * gnp.pi) + ldetK + norm2)
    return L.reshape(())


def negative_log_likelihood(model, meanparam, covparam, xi, zi):
    """Computes the negative log-likelihood of the Gaussian process model with a
    given mean.

    This function computes the negative log-likelihood based on
    the provided mean function, covariance function, and their
    parameters.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `mean` and `covariance`.
    meanparam : gnp.array
        Parameters for the mean function. This array contains the
        hyperparameters required by the chosen mean function.
    covparam : gnp.array
        Parameters for the covariance function. This array
        contains the hyperparameters required by the chosen
        covariance function.
    xi : ndarray(ni,d)
        Locations of the data points in the input space, where ni
        is the number of points and d is the dimensionality
        of the input space.
    zi : ndarray(ni, )
        Observed values corresponding to each data point in xi.

    Returns
    -------
    nll : scalar
        Negative log-likelihood of the observed data given the
        model, mean, and covariance parameters.
    """
    zi_prior_mean = model.mean(xi, meanparam).reshape(-1)
    centered_zi = zi - zi_prior_mean
    return negative_log_likelihood_zero_mean(model, covparam, xi, centered_zi)


def negative_log_restricted_likelihood(model, covparam, xi, zi):
    """Compute the negative log-restricted likelihood of the GP model.

    This method calculates the negative log-restricted likelihood,
    which is used for parameter estimation in the Gaussian Process
    model with a mean of type "linear predictor".

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `mean` and `covariance`.
    covparam : gnp.array
        Covariance parameters for the Gaussian Process.
    xi : array_like, shape (n, d)
        Input data points used for fitting the GP model, where n
        is the number of points and d is the dimensionality.
    zi : array_like, shape (n, )
        Output (response) values corresponding to the input data points xi.

    Returns
    -------
    L : float
        Negative log-restricted likelihood value.
    """
    K = model.covariance(xi, xi, covparam)
    P = model.mean(xi, model.meanparam)
    W = compute_contrast_matrix(P)  # (n, n-q)
    Wzi = gnp.matmul(W.T, zi)  # (n-q,)
    G = compute_contrast_covariance(W, K)  # (n-q, n-q)
    try:
        WKWinv_Wzi, C = gnp.cholesky_solve(G, Wzi)
    except RuntimeError:
        return gnp.safe_inf()
    norm2 = gnp.einsum("i..., i...", Wzi, WKWinv_Wzi)
    ldetWKW = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))
    n, q = P.shape
    L = 0.5 * ((n - q) * gnp.log(2.0 * gnp.pi) + ldetWKW + norm2)
    return L.reshape(())
