# gpmp/core/sample_paths.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Sampling routines for Gaussian Process models.

This module provides:
- Unconditional sampling of GP paths on a grid `xt`.
- Conditioning-by-kriging of unconditional paths given observations.
- Conditioning variant for parameterized mean functions.
"""
import gpmp.num as gnp


def sample_paths(model, xt, nb_paths, method: str = "chol", check_result: bool = True):
    """Generates ``nb_paths`` sample paths on ``xt`` from the zero-mean GP model GP(0, k),
    where k is the covariance specified by ``model.covariance``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `covariance` and `covparam`.
    xt : ndarray, shape (nt, d)
        Input data points where the sample paths are to be generated.
    nb_paths : int
        Number of sample paths to generate.
    method : {'chol','svd'}, optional (default: 'chol')
        Factorization used to draw samples from N(0, K(xt,xt)).
    check_result : bool, optional (default: True)
        If True, checks basic success conditions for the factorization.

    Returns
    -------
    ndarray, shape (nt, nb_paths)
        Array containing the generated sample paths at the input points xt.

    Notes
    -----
    - 'chol': K = C Cᵀ, draw as C @ N(0, I).
    - 'svd' : K = U diag(s) Uᵀ, draw as (U sqrt(diag(s)) Uᵀ) @ N(0, I).
    """
    xt_ = gnp.asarray(xt)
    K = model.covariance(xt_, xt_, model.covparam)

    if method == "chol":
        C = gnp.cholesky(K)
        if check_result and gnp.isnan(C).any():
            raise AssertionError(
                "Cholesky factorization failed (NaNs). Consider adding jitter or use method='svd'."
            )
    elif method == "svd":
        # For symmetric K, use hermitian=True path if backend provides it
        U, s, Vt = gnp.svd(K, full_matrices=True, hermitian=True)
        # Build symmetric square root: U * sqrt(s) * Uᵀ
        C = gnp.matmul(U * gnp.sqrt(s), Vt)
    else:
        raise ValueError("method must be 'chol' or 'svd'")

    zsim = gnp.matmul(C, gnp.randn(K.shape[0], nb_paths))
    return zsim


def conditional_sample_paths(
    model, ztsim, xi_ind, zi, xt_ind, lambda_t, convert_out: bool = True
):
    """Generates conditional sample paths on xt from unconditional sample paths ``ztsim``,
    using the matrix of kriging weights ``lambda_t`` (as provided by
    ``Model.kriging_predictor`` or ``Model.predict``).

    Conditioning is done with respect to ``ni`` observations, located at indices
    ``xi_ind`` in ``ztsim``, with observed values ``zi``. ``xt_ind`` specifies
    indices in ``ztsim`` corresponding to conditional simulation points.

    This routine assumes the mean function is of type 'zero' or 'linear_predictor'.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model (mean/cov not directly used here).
    ztsim : ndarray, shape (n, nb_paths)
        Unconditional sample paths.
    xi_ind : ndarray, shape (ni,), dtype=int
        Indices of observed data points in ztsim.
    zi : ndarray, shape (ni, 1) or (ni,)
        Observed values corresponding to the input data points xi.
    xt_ind : ndarray, shape (nt,), dtype=int
        Indices of prediction data points in ztsim.
    lambda_t : ndarray, shape (ni, nt)
        Kriging weights.
    convert_out : bool, optional (default: True)
        Whether to return numpy arrays or keep backend types.

    Returns
    -------
    ztsimc : ndarray, shape (nt, nb_paths)
        Conditional sample paths at the prediction data points xt.

    Notes
    -----
    Implements "conditioning by kriging"; see Chiles & Delfiner (1999).
    """
    zi_ = gnp.asarray(zi).reshape(-1, 1)
    ztsim_ = gnp.asarray(ztsim)
    xi_ind = gnp.asarray(xi_ind, dtype=int).reshape(-1)

    # Innovation at observed indices
    delta = zi_ - ztsim_[xi_ind, :]  # (ni, nb_paths)

    # Conditioned paths at prediction indices
    ztsimc = ztsim_[xt_ind, :] + gnp.einsum(
        "ij,ik->jk", lambda_t, delta
    )  # (nt, nb_paths)

    if convert_out:
        ztsimc = gnp.to_np(ztsimc)
    return ztsimc


def conditional_sample_paths_parameterized_mean(
    model, ztsim, xi, xi_ind, zi, xt, xt_ind, lambda_t, convert_out: bool = True
):
    """Generates conditional sample paths with a parameterized mean function.

    This method accommodates parameterized means, adjusting the unconditional
    sample paths ``ztsim`` with respect to observed values ``zi`` at ``xi`` and
    prediction points ``xt``, using kriging weights ``lambda_t``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `mean` and `meanparam`.
    ztsim : ndarray
        Unconditional sample paths of shape (n, nb_paths).
    xi : ndarray
        Input data points for observed values.
    xi_ind : ndarray, dtype=int
        Indices of observed data points in ztsim.
    zi : ndarray
        Observed values at the input data points xi.
    xt : ndarray
        Prediction data points.
    xt_ind : ndarray, dtype=int
        Indices of prediction data points in ztsim.
    lambda_t : ndarray
        Kriging weights matrix (ni, nt).
    convert_out : bool, optional (default: True)
        Whether to return numpy arrays or keep backend types.

    Returns
    -------
    ztsimc : ndarray, shape (nt, nb_paths)
        Conditional sample paths at prediction points xt, adjusted for a parameterized mean.
    """
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi)
    xt_ = gnp.asarray(xt)
    ztsim_ = gnp.asarray(ztsim)

    xi_ind = gnp.asarray(xi_ind).reshape(-1)
    xt_ind = gnp.asarray(xt_ind).reshape(-1)

    # Center observations by parameterized prior mean
    zi_prior_mean_ = model.mean(xi_, model.meanparam).reshape(-1)
    zi_centered_ = zi_ - zi_prior_mean_

    # Prior mean at targets
    zt_prior_mean_ = model.mean(xt_, model.meanparam).reshape(-1, 1)

    # Innovation against unconditional paths at observed indices
    delta = zi_centered_.reshape(-1, 1) - ztsim_[xi_ind, :]

    # Condition, then add back prior mean at targets
    ztsimc = (
        ztsim_[xt_ind, :] + gnp.einsum("ij,ik->jk", lambda_t, delta) + zt_prior_mean_
    )

    if convert_out:
        ztsimc = gnp.to_np(ztsimc)
    return ztsimc
