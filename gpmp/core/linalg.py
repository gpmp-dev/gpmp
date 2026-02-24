# gpmp/core/linalg.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Linear-algebra utilities shared across gpmp.core modules.

This file isolates small, backend-agnostic helpers (built on top of
`gpmp.num as gnp`) so they can be reused by kriging, validation, and
likelihood code without import cycles.
"""
import gpmp.num as gnp


def diag_Kinv_from_chol(C, lower: bool = True):
    """Return diag(K^{-1}) from a Cholesky factor C.

    Parameters
    ----------
    C : array_like, shape (n, n)
        Cholesky factor of K. If `lower` is True, K = C Cᵀ; otherwise K = Cᵀ C.
    lower : bool, optional
        Whether C is lower-triangular (default True).

    Returns
    -------
    diag_Kinv : array_like, shape (n,)
        Diagonal of K^{-1}.

    Notes
    -----
    If K = C Cᵀ with C lower-triangular, then K^{-1} = C^{-T} C^{-1}.
    Let T = C^{-1}. Then diag(K^{-1}) equals the column-wise sum of squares of T.
    For the upper-triangular convention, it is the row-wise sum of squares.
    """
    n = C.shape[0]
    I = gnp.eye(n)
    # T = C^{-1}
    T = gnp.solve_triangular(C, I, lower=lower)
    # diag(K^{-1}) = sum of squares of columns (lower) or rows (upper)
    if lower:
        return gnp.sum(T * T, axis=0)
    else:
        return gnp.sum(T * T, axis=1)


def compute_contrast_matrix(P):
    """Compute a matrix of contrasts W from a design matrix P.

    Parameters
    ----------
    P : array_like, shape (n, q)
        Design matrix of the linear predictor.

    Returns
    -------
    W : array_like, shape (n, n-q)
        Columns of W span Null(Pᵀ). Built from a complete QR of P.

    Notes
    -----
    A complete QR factorization P = Q R with Q ∈ ℝ^{n×n} yields
    Q = [Q₁ | Q₂], where Q₁ spans Col(P) and Q₂ spans Null(Pᵀ).
    We return W = Q₂.
    """
    n, q = P.shape
    Q, R = gnp.qr(P, mode="complete")
    return Q[:, q:n]


def compute_contrast_covariance(W, K):
    """Compute covariance matrix of contrasts G = Wᵀ (K W).

    Parameters
    ----------
    W : array_like, shape (n, n-q)
        Contrast matrix (columns in Null(Pᵀ)).
    K : array_like, shape (n, n)
        Covariance matrix at observation points.

    Returns
    -------
    G : array_like, shape (n-q, n-q)
        Covariance of the contrasts Wᵀ z when z ~ N(0, K).
    """
    return gnp.matmul(W.T, gnp.matmul(K, W))


# Optional helper (not strictly required by current callers) -------------------
def qr_nullspace(P):
    """Return (Q1, W, Rq) where columns of W span Null(Pᵀ).

    Parameters
    ----------
    P : array_like, shape (n, q)

    Returns
    -------
    Q1 : array_like, shape (n, q)
        Orthonormal basis of Col(P).
    W  : array_like, shape (n, n-q)
        Orthonormal basis of Null(Pᵀ).
    Rq : array_like, shape (q, q)
        Upper-triangular factor for the leading block.
    """
    Q, R = gnp.qr(P, mode="complete")
    q = P.shape[1]
    return Q[:, :q], Q[:, q:], R[:q, :q]


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
