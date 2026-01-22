# gpmp/core/kriging.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Kriging predictors and posterior variance computations.

This module contains the numerical routines used by `gpmp.core.Model`
to compute kriging weights and posterior variances for the three mean
handling modes: 'zero', 'parameterized', and 'linear_predictor'.

Functions
---------
kriging_predictor_with_zero_mean(model, xi, xt, return_type=0)
    Compute the kriging predictor assuming a zero mean function.

kriging_predictor(model, xi, xt, return_type=0)
    Compute the kriging predictor considering a non-zero mean function
    ('linear_predictor' / universal kriging). Falls back to a CPD-safe
    nullspace route if the block system is ill-conditioned.

select_predictor(model, xi, zi, xt)
    Helper that selects the appropriate predictor depending on
    `model.meantype` and returns centered data, prior mean at xt,
    kriging weights, and posterior variance.
"""
import gpmp.num as gnp


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def kriging_predictor_with_zero_mean(model, xi, xt, return_type=0):
    """Compute the kriging predictor with zero mean.

    Parameters
    ----------
    model : gpmp.core.Model
        The GP model instance (provides covariance and parameters).
    xi : array_like, shape (n, d)
        Observation points.
    xt : array_like, shape (m, d)
        Prediction points.
    return_type : int, optional
        Indicator for posterior variance:
          -1: return None,
           0: return variance (default),
           1: return full covariance.

    Returns
    -------
    lambda_t : array_like, shape (n, m)
        Kriging weights.
    zt_posterior_variance : array_like
        Posterior variance (or covariance if return_type==1).
    """
    Kii = model.covariance(xi, xi, model.covparam)
    Kit = model.covariance(xi, xt, model.covparam)

    lambda_t, _ = gnp.cholesky_solve(Kii, Kit)

    zt_posterior_variance = _compute_posterior_variance(
        model, xt, lambda_t, Kit, return_type
    )
    return lambda_t, zt_posterior_variance


def kriging_predictor(model, xi, xt, return_type=0):
    """Compute the kriging predictor with a non-zero mean (universal kriging).

    Parameters
    ----------
    model : gpmp.core.Model
        The GP model instance (provides covariance/mean and parameters).
    xi : array_like, shape (n, d)
        Observation points.
    xt : array_like, shape (m, d)
        Prediction points.
    return_type : int, optional
        Indicator for posterior variance:
          -1: return None,
           0: return variance (default),
           1: return full covariance.

    Returns
    -------
    lambda_t : array_like, shape (n, m)
        Kriging weights.
    zt_posterior_variance : array_like
        Posterior variance (or covariance if return_type==1).
    """
    # LHS
    Kii = model.covariance(xi, xi, model.covparam)
    Pi = model.mean(xi, model.meanparam)
    (ni, q) = Pi.shape
    LHS = gnp.vstack((gnp.hstack((Kii, Pi)), gnp.hstack((Pi.T, gnp.zeros((q, q))))))

    # RHS
    Kit = model.covariance(xi, xt, model.covparam)
    Pt = model.mean(xt, model.meanparam)
    RHS = gnp.vstack((Kit, Pt.T))

    # Solve block system; if it fails, use CPD-safe nullspace route
    try:
        lambdamu_t = gnp.solve(
            LHS, RHS, overwrite_a=True, overwrite_b=False, assume_a="sym"
        )
        lambda_t = lambdamu_t[0:ni, :]
        zt_posterior_variance = _compute_posterior_variance(
            model, xt, lambdamu_t, RHS, return_type
        )
        return lambda_t, zt_posterior_variance
    except Exception:
        return _kriging_predictor_nullspace(model, xi, xt, return_type)


def select_predictor(model, xi, zi, xt):
    """
    Select the appropriate kriging predictor based on model.meantype.

    Returns
    -------
    zi_centered : array_like, shape (n,)
        Centered observed values.
    zt_prior_mean : array_like or scalar
        Prior mean adjustment for predictions at xt.
    lambda_t : array_like, shape (n, m)
        Kriging weights.
    zt_posterior_variance : array_like
        Posterior variance (vector or matrix depending on return_type=0/1).
    """
    # Default: no mean adjustment.
    zt_prior_mean = 0.0
    zi_centered = zi

    if model.meantype == "zero":
        lambda_t, zt_posterior_variance = kriging_predictor_with_zero_mean(
            model, xi, xt, return_type=0
        )
    elif model.meantype == "linear_predictor":
        lambda_t, zt_posterior_variance = kriging_predictor(
            model, xi, xt, return_type=0
        )
    elif model.meantype == "parameterized":
        if model.meanparam is None:
            raise ValueError(
                "For meantype 'parameterized', meanparam should not be None."
            )
        # Use zero-mean predictor but center the data w.r.t the parameterized mean.
        lambda_t, zt_posterior_variance = kriging_predictor_with_zero_mean(
            model, xi, xt, return_type=0
        )
        zi_prior_mean = model.mean(xi, model.meanparam).reshape(-1)
        zi_centered = zi - zi_prior_mean
        zt_prior_mean = model.mean(xt, model.meanparam).reshape(-1)
    else:
        raise ValueError(
            f"Invalid meantype {model.meantype}. "
            "Supported types are 'zero', 'parameterized', and 'linear_predictor'."
        )

    return zi_centered, zt_prior_mean, lambda_t, zt_posterior_variance


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _compute_posterior_variance(model, xt, lambdamu_t, RHS, return_type=0):
    """Compute posterior variance based on return type.

    Parameters
    ----------
    model : gpmp.core.Model
    xt : (m, d)
    lambdamu_t : array_like
        For zero-mean case, this is lambda_t (n x m).
        For universal kriging, this is [lambda_t; mu_t] (n+q x m).
    RHS : array_like
        For zero-mean, RHS = K(Xi, Xt).
        For universal,  RHS = [K(Xi, Xt); P(Xt)^T].
    return_type : int
        -1: None, 0: marginal variances (default), 1: full covariance.

    Returns
    -------
    posterior variance or covariance matrix.
    """
    if return_type == -1:
        return None
    elif return_type == 0:
        zt_prior_variance = model.covariance(xt, None, model.covparam, pairwise=True)
        return zt_prior_variance - gnp.einsum("i..., i...", lambdamu_t, RHS)
    elif return_type == 1:
        zt_prior_variance = model.covariance(xt, None, model.covparam, pairwise=False)
        return zt_prior_variance - gnp.matmul(lambdamu_t.T, RHS)
    else:
        raise ValueError("return_type must be in {-1, 0, 1}")


def _kriging_predictor_nullspace(model, xi, xt, return_type=0):
    """CPD-safe universal kriging using contrasts (nullspace of Pᵀ).

    This is a fallback (and often a robust default) when the block system
    in `kriging_predictor` is ill-conditioned due to conditional positive
    definiteness (CPD) of the kernel with a linear predictor mean.

    Parameters
    ----------
    model : gpmp.core.Model
    xi : (n, d)
    xt : (m, d)
    return_type : int
        -1: None, 0: variances (default), 1: full covariance.

    Returns
    -------
    lambda_t : (n, m)
    zt_posterior_variance : (m,) or (m, m)
    """
    K = model.covariance(xi, xi, model.covparam)
    P = model.mean(xi, model.meanparam)
    n, q = P.shape
    Kit = model.covariance(xi, xt, model.covparam)
    Pt = model.mean(xt, model.meanparam)  # (m, q)

    Q, R = gnp.qr(P, mode="complete")
    Q1, W = Q[:, :q], Q[:, q:]  # W spans Null(Pᵀ)
    Rq = R[:q, :q]

    KW = gnp.matmul(K, W)
    G = gnp.matmul(W.T, KW)  # SPD in contrast space

    # Solve alpha = G^{-1} Wᵀ K(Xi,Xt)
    alpha, _ = gnp.cholesky_solve(G, gnp.matmul(W.T, Kit))
    # Solve beta  = Rq^{-T} P(Xt)ᵀ  (enforces unbiasedness constraints)
    beta = gnp.solve(Rq.T, Pt.T, assume_a="sym")

    lambda_t = gnp.matmul(W, alpha) + gnp.matmul(Q1, beta)

    if return_type == -1:
        zt_posterior_variance = None
    elif return_type == 0:
        v0 = model.covariance(xt, xt, model.covparam, pairwise=True)
        RHS = gnp.vstack((Kit, Pt.T))
        LM = gnp.vstack((lambda_t, beta))
        zt_posterior_variance = v0 - gnp.einsum("i..., i...", LM, RHS)
    elif return_type == 1:
        V0 = model.covariance(xt, xt, model.covparam, pairwise=False)
        RHS = gnp.vstack((Kit, Pt.T))
        LM = gnp.vstack((lambda_t, beta))
        zt_posterior_variance = V0 - gnp.matmul(LM.T, RHS)
    else:
        raise ValueError("return_type must be in {-1,0,1}")

    return lambda_t, zt_posterior_variance
