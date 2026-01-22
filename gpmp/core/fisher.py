# gpmp/core/fisher.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Fisher information matrices for covariance parameters.

This module provides:
- Finite-difference Fisher information in SPD form.
- CPD-safe Fisher information (contrast space when using a linear predictor mean).
- A Hessian-based variant using second-order differentiation (if backend supports it).
"""
import gpmp.num as gnp


def fisher_information(model, xi, covparam=None, epsilon: float = 1e-3):
    """Compute the Fisher information matrix using
    finite-difference numerical differentiation (with respect to
    the covariance parameters).

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `covariance`.
    xi : ndarray (n, d)
        Observation points where n is the number of points and d is the dimensionality.
    covparam : ndarray, optional
        Covariance parameters at which to compute Fisher information.
        If None, uses model.covparam.
    epsilon : float, optional
        Step size for finite-difference approximation. Default is 1e-3.

    Returns
    -------
    fisher_info : ndarray (p, p)
        Fisher Information matrix, where p is the number of covariance parameters.

    Notes
    -----
    1) The Fisher information for a zero-mean Gaussian vector with covariance K(θ) is:
         I_{ij}(θ) = 0.5 * Tr(K^{-1}(θ) * ∂K(θ)/∂θ_i * K^{-1}(θ) * ∂K(θ)/∂θ_j).
    2) dK/dθ_i is computed via central finite differences:
         dK_i = [K(θ + e_i) - K(θ - e_i)] / (2 * epsilon).
    3) This method can be computationally expensive for high-dimensional parameter spaces,
       but is robust when autodiff faces numerical instabilities.
    """
    theta = model.covparam if covparam is None else gnp.asarray(covparam)
    p = theta.shape[0]
    I = gnp.empty((p, p))

    # K and its inverse
    K = model.covariance(xi, xi, theta)
    try:
        K_inv = gnp.inv(K)
    except Exception:
        raise RuntimeError("Covariance matrix not invertible; adjust hyperparameters or add jitter.")

    # Finite differences for dK_i (central)
    dK = []
    for i in range(p):

        def f(tmp_val):
            t = gnp.copy(theta)
            t = gnp.set_elem_1d(t, i, tmp_val)
            return model.covariance(xi, xi, t)

        dK_i = gnp.derivative_finite_diff(f, theta[i], epsilon)
        dK.append(dK_i)

    # Fisher entries
    for i in range(p):
        for j in range(i, p):
            term = 0.5 * gnp.trace(K_inv @ dK[i] @ K_inv @ dK[j])
            I = gnp.set_elem_2d(I, i, j, term)
            I = gnp.set_elem_2d(I, j, i, term)
    return I


def fisher_information_cpd(model, xi, covparam=None, epsilon: float = 1e-3):
    """
    Fisher information for covariance parameters with CPD kernels.

    If the mean is of type "linear_predictor", the information is computed
    in contrast space with G = Wᵀ K W, where W spans Null(Pᵀ). Otherwise,
    the standard SPD formula with K is used.

    I_ij = 0.5 * Tr( M^{-1} ∂M/∂θ_i M^{-1} ∂M/∂θ_j ),
    with M = G (CPD case) or M = K (SPD case).

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `mean` (when linear_predictor) and `covariance`.
    xi : (n, d) array
        Design points.
    covparam : (p,) array, optional
        Covariance parameters (default: model.covparam).
    epsilon : float, optional
        Step for central finite differences (default: 1e-3).

    Returns
    -------
    I : (p, p) array
        Fisher information matrix.
    """
    theta = model.covparam if covparam is None else gnp.asarray(covparam)
    p = theta.shape[0]

    # Build base covariance
    K = model.covariance(xi, xi, theta)

    # If linear predictor, operate in contrast space
    if model.meantype == "linear_predictor":
        P = model.mean(xi, model.meanparam)
        Q, _ = gnp.qr(P, mode="complete")
        W = Q[:, P.shape[1] :]  # (n, n-q)
        G = gnp.matmul(W.T, gnp.matmul(K, W))  # SPD in contrast space
        # Pre-factorize for solves
        Cg = gnp.cholesky(G)

        # Finite differences for dK_i and then project: dG_i = Wᵀ dK_i W
        dG = []
        for i in range(p):

            def f(tmp):
                t = gnp.copy(theta)
                t = gnp.set_elem_1d(t, i, tmp)
                return model.covariance(xi, xi, t)

            dK_i = gnp.derivative_finite_diff(f, theta[i], epsilon)
            dG_i = gnp.matmul(W.T, gnp.matmul(dK_i, W))
            dG.append(dG_i)

        # Helper: solve G^{-1} A via Cholesky
        def Gsolve(A):
            X, _ = gnp.cholesky_solve(G, A)
            return X

        I = gnp.empty((p, p))
        for i in range(p):
            Gi = Gsolve(dG[i])
            for j in range(i, p):
                term = 0.5 * gnp.trace(Gi @ Gsolve(dG[j]))
                I = gnp.set_elem_2d(I, i, j, term)
                I = gnp.set_elem_2d(I, j, i, term)
        return I

    # Otherwise, fallback to SPD Fisher on K
    return fisher_information(model, xi, covparam=theta, epsilon=epsilon)


def fisher_information_torch(model, xi, covparam):
    """Compute Fisher Information matrix using second-order differentiation.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model providing `covariance`.
    xi : ndarray (n, d)
        Observation points.
    covparam : ndarray (p,)
        Covariance parameters at which to compute Fisher information.

    Returns
    -------
    fisher_info : ndarray (p, p)
        Fisher Information matrix estimated as 0.5 * Hessian of log|K(θ)|.

    Notes
    -----
    This routine requires a backend that provides:
      - `gnp.cholesky`
      - `gnp.SecondOrderDifferentiableFunction` with methods:
          * evaluate(theta)
          * gradient()
          * hessian()
    """
    xi_tensor = gnp.asarray(xi)

    def log_det_cov(params):
        K = model.covariance(xi_tensor, xi_tensor, params)
        L = gnp.cholesky(K)
        return 2.0 * gnp.sum(gnp.log(gnp.diag(L)))

    sodf = gnp.SecondOrderDifferentiableFunction(log_det_cov)
    sodf.evaluate(covparam)
    sodf.gradient()
    H = sodf.hessian()
    return 0.5 * H
