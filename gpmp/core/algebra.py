# gpmp/core/algebra.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Algebraic forms used by GP parameter initialization and diagnostics.
"""
import gpmp.num as gnp
from .linalg import compute_contrast_matrix, compute_contrast_covariance


def norm_k_sqrd_with_zero_mean(model, xi, zi, covparam):
    """Compute the squared norm z^T K^-1 z for zero-mean models."""
    K = model.covariance(xi, xi, covparam)
    Kinv_zi, _ = gnp.cholesky_solve(K, zi)
    norm_sqrd = gnp.einsum("i..., i...", zi, Kinv_zi)
    return norm_sqrd


def k_inverses(model, xi, zi, covparam):
    """Compute z^T K^-1 z, K^-1 1 and K^-1 z."""
    K = model.covariance(xi, xi, covparam)
    ones_vector = gnp.ones(zi.shape)
    Kinv = gnp.cholesky_inv(K)
    Kinv_zi = gnp.einsum("...i, i...", Kinv, zi)
    Kinv_1 = gnp.einsum("...i, i...", Kinv, ones_vector)
    zTKinvz = gnp.einsum("i..., i...", zi, Kinv_zi)
    return zTKinvz, Kinv_1, Kinv_zi


def norm_k_sqrd(model, xi, zi, covparam):
    """Compute (Wz)^T (WKW)^-1 (Wz) for linear_predictor models."""
    K = model.covariance(xi, xi, covparam)
    P = model.mean(xi, model.meanparam)
    W = compute_contrast_matrix(P)  # (n, n-q)
    Wzi = gnp.matmul(W.T, zi)
    G = compute_contrast_covariance(W, K)
    WKWinv_Wzi, _ = gnp.cholesky_solve(G, Wzi)
    norm_sqrd = gnp.einsum("i..., i...", Wzi, WKWinv_Wzi)
    return norm_sqrd
