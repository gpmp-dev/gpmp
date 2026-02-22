# gpmp/core/loo.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Leave-One-Out (LOO) procedures (virtual cross-validation).

This module provides the LOO prediction/error routines used by
`gpmp.core.Model.loo`, for the three mean-handling modes:
'zero', 'parameterized', and 'linear_predictor'.

The implementations follow the "virtual cross-validation" formulas,
avoiding an explicit loop over points.
"""
import gpmp.num as gnp
from .linalg import diag_Kinv_from_chol


def loo(model, xi, zi):
    """Compute the leave-one-out (LOO) prediction error.

    This function computes the LOO prediction error using the
    "virtual cross-validation" formula, which allows for
    computation of LOO predictions without looping on the data points.

    Parameters
    ----------
    model : gpmp.core.Model
        The GP model instance (provides mean/covariance and parameters).
    xi : array_like, shape (n, d)
        Input data points used for fitting the GP model, where n
        is the number of points and d is the dimensionality.
    zi : array_like, shape (n, ) or (n, 1)
        Output (response) values corresponding to the input data points xi.

    Returns
    -------
    zloo : array_like, shape (n, )
        Leave-one-out predictions for each data point in xi.
    sigma2loo : array_like, shape (n, )
        Variance of the leave-one-out predictions.
    eloo : array_like, shape (n, )
        Leave-one-out prediction errors for each data point in xi.
    """
    if model.meantype == "zero":
        return _loo_with_zero_mean(model, model.covparam, xi, zi)
    elif model.meantype == "parameterized":
        return _loo_with_parameterized_mean(
            model, model.meanparam, model.covparam, xi, zi
        )
    elif model.meantype == "linear_predictor":
        # CPD-safe default for universal kriging
        return _loo_with_linear_predictor_mean_cpd(
            model, model.meanparam, model.covparam, xi, zi
        )
    else:
        raise ValueError(f"Unknown mean type: {model.meantype}")


# ------------------------------------------------------------------------------
# Zero-mean case
# ------------------------------------------------------------------------------
def _loo_with_zero_mean(model, covparam, xi, zi):
    """Compute LOO prediction error for zero mean.

    LOO predictions based on the "virtual cross-validation" formula.
    """
    K = model.covariance(xi, xi, covparam)
    # K^{-1} z and the Cholesky C
    Kinv_zi, C = gnp.cholesky_solve(K, zi)  # returns (K^{-1} z, C)

    # diag(K^{-1}) via triangular inverse
    Kinvdiag = diag_Kinv_from_chol(C)

    # e_loo,i  = 1 / Kinv_i,i ( Kinv  z )_i
    eloo = Kinv_zi.reshape(-1) / Kinvdiag
    # sigma2_loo,i = 1 / Kinv_i,i
    sigma2loo = 1.0 / Kinvdiag
    # zloo_i = z_i - e_loo,i
    zloo = zi - eloo
    return zloo, sigma2loo, eloo


# ------------------------------------------------------------------------------
# Parameterized mean case
# ------------------------------------------------------------------------------
def _loo_with_parameterized_mean(model, meanparam, covparam, xi, zi):
    """Compute LOO prediction error for parameterized mean."""
    zi_prior_mean = model.mean(xi, meanparam).reshape(-1)
    centered_zi = zi - zi_prior_mean
    zloo_centered, sigma2loo, eloo_centered = _loo_with_zero_mean(
        model, covparam, xi, centered_zi
    )
    zloo = zloo_centered + zi_prior_mean
    return zloo, sigma2loo, eloo_centered


# ------------------------------------------------------------------------------
# Linear predictor mean (universal kriging) — CPD-safe version
# ------------------------------------------------------------------------------
def _loo_with_linear_predictor_mean_cpd(model, meanparam, covparam, xi, zi):
    """Compute LOO prediction error for linear_predictor mean. CPD-safe version.

    This implementation operates in contrast space using W that spans
    Null(Pᵀ) from a complete QR factorization P = Q R, ensuring numerical
    robustness with conditionally positive definite kernels.
    """
    K = model.covariance(xi, xi, covparam)
    P = model.mean(xi, meanparam)

    Q, _R = gnp.qr(P, mode="complete")
    W = Q[:, P.shape[1] :]  # (n, n-q)
    G = gnp.matmul(W.T, gnp.matmul(K, W))  # (n-q, n-q), SPD

    # Solve S = G^{-1} W^T  (n-q x n)
    S, _ = gnp.cholesky_solve(G, W.T)

    # Qinv z = W S z
    Qinvzi = gnp.matmul(W, gnp.matmul(S, zi))

    # diag(Qinv) = row-wise inner products w_i^T (G^{-1} w_i)
    # which equals sum_r W[i,r] * S[r,i]
    Qinvdiag = gnp.sum(W * S.T, axis=1)

    eloo = Qinvzi / Qinvdiag
    sigma2loo = 1.0 / Qinvdiag
    zloo = zi - eloo
    return zloo, sigma2loo, eloo


# ------------------------------------------------------------------------------
# (Optional) SPD-only universal kriging variant — not used by default
# ------------------------------------------------------------------------------
def _loo_with_linear_predictor_mean_spd(model, meanparam, covparam, xi, zi):
    """Compute LOO prediction error for linear_predictor mean (SPD-only formula).

    Provided for completeness; prefer the CPD-safe version above.
    """
    K = model.covariance(xi, xi, covparam)
    P = model.mean(xi, meanparam)

    # Qinv = K^-1 - K^-1P (Pt K^-1 P)^-1 Pt K^-1
    Kinv = gnp.inv(K)
    KinvP = gnp.matmul(Kinv, P)
    PtKinvP = gnp.einsum("ki, kj->ij", P, KinvP)
    R = gnp.solve(PtKinvP, KinvP.T)
    Qinv = Kinv - gnp.matmul(KinvP, R)

    # e_loo,i  = 1 / Q_i,i ( Qinv  z )_i
    Qinvzi = gnp.matmul(Qinv, zi)  # shape (n, )
    Qinvdiag = gnp.diag(Qinv)  # shape (n, )
    eloo = Qinvzi / Qinvdiag  # shape (n, )

    # sigma2_loo,i = 1 / Qinv_i,i
    sigma2loo = 1.0 / Qinvdiag  # shape (n, )

    # z_loo
    zloo = zi - eloo  # shape (n, )

    return zloo, sigma2loo, eloo
