# gpmp/core/likelihood.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Negative (restricted) log-likelihoods and related quadratic forms.

This module implements the numerical routines used by `gpmp.core.Model`
for likelihood evaluation in the zero-mean, parameterized-mean, and
linear_predictor cases, as well as helper quadratic forms.
"""
import gpmp.num as gnp
from . import utils
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
        return utils.return_inf()
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
        return utils.return_inf()
    norm2 = gnp.einsum("i..., i...", Wzi, WKWinv_Wzi)
    ldetWKW = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))
    n, q = P.shape
    L = 0.5 * ((n - q) * gnp.log(2.0 * gnp.pi) + ldetWKW + norm2)
    return L.reshape(())


def norm_k_sqrd_with_zero_mean(model, xi, zi, covparam):
    """Compute the squared norm of the residual vector with zero mean.

    This method calculates the squared norm of the residual vector
    (zi - mean(xi)) using the inverse of the covariance matrix K.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `covariance`.
    xi : array_like, shape (n, d)
        Input data points used for fitting the GP model, where n
        is the number of points and d is the dimensionality.
    zi : array_like, shape (n, )
        Output (response) values corresponding to the input data points xi.
    covparam : array_like
        Covariance parameters for the Gaussian Process.

    Returns
    -------
    norm_sqrd : float
        Squared norm of the residual vector.
    """
    K = model.covariance(xi, xi, covparam)
    Kinv_zi, _ = gnp.cholesky_solve(K, zi)
    norm_sqrd = gnp.einsum("i..., i...", zi, Kinv_zi)
    return norm_sqrd


def k_inverses(model, xi, zi, covparam):
    """Compute various quantities involving the inverse of K.

    Specifically, this method calculates:
    - z^T K^-1 z
    - K^-1 1
    - K^-1 z

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `covariance`.
    xi : array_like, shape (n, d)
        Input data points used for fitting the GP model.
    zi : array_like, shape (n, 1)
        Output (response) values corresponding to the input data points xi.
    covparam : array_like
        Covariance parameters for the Gaussian process.

    Returns
    -------
    zTKinvz : float
        z^T K^-1 z
    Kinv1 : array_like, shape (n, 1)
        K^-1 1
    Kinvz : array_like, shape (n, 1)
        K^-1 z
    """
    K = model.covariance(xi, xi, covparam)
    ones_vector = gnp.ones(zi.shape)
    Kinv = gnp.cholesky_inv(K)
    Kinv_zi = gnp.einsum("...i, i...", Kinv, zi)
    Kinv_1 = gnp.einsum("...i, i...", Kinv, ones_vector)
    zTKinvz = gnp.einsum("i..., i...", zi, Kinv_zi)
    return zTKinvz, Kinv_1, Kinv_zi


def norm_k_sqrd(model, xi, zi, covparam):
    """Compute the squared norm of the residual vector after applying the contrast
    matrix W.

    This method calculates the squared norm of the residual vector
    (Wz) using the inverse of the covariance matrix (WKW), where W
    is a matrix of contrasts.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `mean` and `covariance`.
    xi : ndarray, shape (ni, d)
        Input data points used for fitting the GP model, where ni
        is the number of points and d is the dimensionality.
    zi : ndarray, shape (ni, 1) or (ni, )
        Output (response) values corresponding to the input data points xi.
    covparam : array_like
        Covariance parameters for the Gaussian Process.

    Returns
    -------
    float
        The squared norm of the residual vector after applying the
        contrast matrix W: (Wz)' (WKW)^-1 Wz.
    """
    K = model.covariance(xi, xi, covparam)
    P = model.mean(xi, model.meanparam)
    W = compute_contrast_matrix(P)  # (n, n-q)
    # Contrasts (n-q) x 1
    Wzi = gnp.matmul(W.T, zi)
    # Covariance of contrasts G = W' * (K * W)
    G = compute_contrast_covariance(W, K)
    # Solve G^(-1) * (W' zi)
    WKWinv_Wzi, _ = gnp.cholesky_solve(G, Wzi)
    # Quadratic form
    norm_sqrd = gnp.einsum("i..., i...", Wzi, WKWinv_Wzi)
    return norm_sqrd
