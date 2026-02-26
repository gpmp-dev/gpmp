# gpmp/kernel/priors.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Priors used in GP kernel parameter estimation.

Functions
---------
log_prior_jeffreys_variance
    Jeffreys-style log-prior on variance.
log_prior_power_law
    Power-law log-prior with soft cutoffs on covariance parameters.
log_prior_reference
    Jeffreys-rule/reference-style prior from Fisher information.
neg_log_restricted_posterior_with_jeffreys_prior
    Negative restricted posterior with variance Jeffreys-style prior.
neg_log_restricted_posterior_power_laws_prior
    Negative restricted posterior with power-law prior.
log_prior_gaussian_logsigma2
    Gaussian log-prior on ``log(sigma^2)``.
neg_log_restricted_posterior_gaussian_logsigma2_prior
    Negative restricted posterior with Gaussian prior on ``log(sigma^2)``.
neglog_f_logrho
    Barrier + linear-tail penalty on ``logrho``.
log_prior_logrho_barrier_linear
    Log-prior induced by the ``logrho`` barrier-linear penalty.
neg_log_restricted_posterior_with_logrho_prior
    Negative restricted posterior with ``logrho`` prior.
neg_log_restricted_posterior_gaussian_logsigma2_and_logrho_prior
    Negative restricted posterior with priors on ``log(sigma^2)`` and ``logrho``.
"""

import gpmp.num as gnp


# ------------------------- basic priors -------------------------
def log_prior_jeffreys_variance(covparam, lambda_var=1.0):
    """
    Compute Jeffreys-type log-prior on variance.

    Parameters
    ----------
    covparam : array_like
        Covariance parameter vector. ``covparam[0]`` is ``log(sigma^2)``.
    lambda_var : float, default=1.0
        Scaling coefficient in ``log p = -lambda_var * log(sigma^2)``.

    Returns
    -------
    scalar
        Log-prior value.

    Notes
    -----
    This is a Jeffreys-style prior component on the variance term only, not the
    full multivariate Jeffreys prior for all covariance parameters.

    References
    ----------
    Jeffreys, H. (1946). An invariant form for the prior probability in
    estimation problems. Proceedings of the Royal Society A, 186(1007), 453-461.
    """
    return -lambda_var * covparam[0]


def log_prior_power_law(
    covparam,
    lambda_var=1.0,
    cut_logvariance_high=9.21,  # ~ log(1e4)
    lambda_lengthscales=0.0,
    cut_loginvrho_low=-9.21,
    cut_loginvrho_high=9.21,
    penalty_factor=100,
):
    """
    Compute power-law log-prior with soft cutoffs on covariance parameters.

    Parameters
    ----------
    covparam : array_like
        Covariance parameter vector ``[log(sigma^2), loginvrho_1, ..., loginvrho_d]``.
    lambda_var : float, default=1.0
        Power-law exponent coefficient for ``log(sigma^2)``.
    cut_logvariance_high : float, default=9.21
        Upper soft cutoff for ``log(sigma^2)``.
    lambda_lengthscales : float, default=0.0
        Power-law exponent coefficient for inverse length-scales.
    cut_loginvrho_low : float, default=-9.21
        Lower soft cutoff for ``loginvrho`` components.
    cut_loginvrho_high : float, default=9.21
        Upper soft cutoff for ``loginvrho`` components.
    penalty_factor : float, default=100
        Linear penalty slope outside cutoff region.

    Returns
    -------
    scalar
        Log-prior value.

    Notes
    -----
    This is a pragmatic regularization prior (power-law + soft cutoffs), not a
    canonical objective prior in the Jeffreys/reference sense.

    References
    ----------
    Berger, J. O., De Oliveira, V., and Sanso, B. (2001). Objective Bayesian
    analysis of spatially correlated data. JASA, 96(456), 1361-1374.
    Paulo, R. (2005). Default priors for Gaussian processes. Annals of
    Statistics, 33(2), 556-582.
    """
    log_sigma2 = covparam[0]
    p = covparam[1:]
    log_prior_sigma2 = -lambda_var * log_sigma2
    extra_sigma2 = penalty_factor * gnp.maximum(log_sigma2 - cut_logvariance_high, 0)
    extra_low = penalty_factor * gnp.maximum(cut_loginvrho_low - p, 0)
    extra_high = penalty_factor * gnp.maximum(p - cut_loginvrho_high, 0)
    log_prior_lengths = (
        -lambda_lengthscales * gnp.sum(p) - gnp.sum(extra_low) - gnp.sum(extra_high)
    )
    return log_prior_sigma2 + extra_sigma2 + log_prior_lengths


def log_prior_reference(model, covparam, xi):
    """
    Compute Jeffreys-rule log-prior from Fisher information.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance exposing ``fisher_information``.
    covparam : array_like
        Covariance parameter vector.
    xi : array_like
        Design/input locations used to compute Fisher information.

    Returns
    -------
    scalar
        ``0.5 * log(det(FisherInfo(theta)))``.

    Notes
    -----
    This is the Jeffreys-rule form for the parameter block. In multiparameter
    settings, strict reference priors may differ from Jeffreys-rule priors and
    depend on parameter ordering.

    References
    ----------
    Jeffreys, H. (1946). An invariant form for the prior probability in
    estimation problems. Proceedings of the Royal Society A, 186(1007), 453-461.
    Bernardo, J. M. (1979). Reference Posterior Distributions for Bayesian
    Inference. JRSS-B, 41(2), 113-128.
    Berger, J. O., Bernardo, J. M., and Sun, D. (2009). The formal definition
    of reference priors. Annals of Statistics, 37(2), 905-938.
    """
    fisher_info = model.fisher_information(xi, covparam)
    return 0.5 * gnp.logdet(fisher_info)


# ---------------------- structured penalties ----------------------
def log_prior_gaussian_logsigma2(covparam, log_sigma2_0, gamma=4.0):
    """
    Compute Gaussian log-prior on ``log(sigma^2)``.

    Parameters
    ----------
    covparam : array_like
        Covariance parameter vector. ``covparam[0]`` is ``log(sigma^2)``.
    log_sigma2_0 : scalar
        Prior mean for ``log(sigma^2)``.
    gamma : float, default=4.0
        Scale factor such that one standard deviation in log-space is
        ``log(gamma)``.

    Returns
    -------
    scalar
        Gaussian log-prior value (up to an additive normalization constant).

    Notes
    -----
    A ``+/-1`` standard deviation variation corresponds to multiplying
    ``sigma^2`` by ``gamma`` or ``1/gamma``.
    This is a weakly informative regularization prior.
    """
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1.")
    log_sigma2 = covparam[0]
    std = gnp.log(gamma)
    z = (log_sigma2 - log_sigma2_0) / std
    return -0.5 * z * z


def neglog_f_logrho(logrho, logrho_min, logrho_0, alpha=1.0):
    """
    Compute elementwise barrier + linear-tail penalty on ``logrho``.

    Parameters
    ----------
    logrho : array_like
        Log-lengthscale vector.
    logrho_min : array_like
        Componentwise hard lower bound.
    logrho_0 : array_like
        Componentwise reference value (penalty minimum).
    alpha : float, default=1.0
        Linear right-tail slope parameter of the penalty.

    Returns
    -------
    array_like
        Componentwise non-negative penalty. Returns ``+inf`` where
        ``logrho <= logrho_min``.

    Notes
    -----
    This is a structural regularization penalty (barrier + tail control), not a
    Jeffreys/reference prior.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if bool(gnp.to_np(gnp.any(logrho_0 <= logrho_min))):
        raise ValueError("logrho_0 must be > logrho_min (componentwise).")

    beta = alpha
    alpha_eff = beta * (logrho_0 - logrho_min)
    logrho_shifted = logrho - logrho_min
    mask = logrho_shifted > 0.0
    shifted_safe = gnp.where(mask, logrho_shifted, 1.0)
    nlf_valid = -alpha_eff * gnp.log(shifted_safe) + beta * shifted_safe
    return gnp.where(mask, nlf_valid, gnp.safe_inf())


def log_prior_logrho_barrier_linear(covparam, logrho_min, logrho_0, alpha=1.0):
    """
    Compute log-prior on ``rho`` from barrier+linear penalty on ``logrho``.

    Parameters
    ----------
    covparam : array_like
        Covariance parameter vector with inverse length-scales in log-domain:
        ``loginvrho = covparam[1:]``.
    logrho_min : array_like
        Lower bound for ``logrho`` components.
    logrho_0 : array_like
        Reference values for ``logrho`` components.
    alpha : float, default=1.0
        Linear right-tail slope parameter.

    Returns
    -------
    scalar
        Log-prior value.

    Notes
    -----
    Induces a prior on lengthscales through ``logrho = -loginvrho`` with hard
    lower support and linear tail regularization.
    """
    logrho = -covparam[1:]
    nlf = neglog_f_logrho(logrho, logrho_min, logrho_0, alpha=alpha)
    return -gnp.sum(nlf)


# ------------------- posterior objective wrappers -------------------
def neg_log_restricted_posterior_with_jeffreys_prior(
    model, covparam, xi, zi, lambda_var=1.0
):
    """
    Compute negative restricted posterior with Jeffreys prior on variance.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    covparam : array_like
        Covariance parameter vector.
    xi, zi : array_like
        Observation inputs and targets.
    lambda_var : float, default=1.0
        Jeffreys prior scaling coefficient.

    Returns
    -------
    scalar
        ``negative_log_restricted_likelihood - log_prior_jeffreys_variance``.

    Notes
    -----
    This combines REML with a variance-only Jeffreys-style penalty.
    """
    nlrl = model.negative_log_restricted_likelihood(covparam, xi, zi)
    return nlrl - log_prior_jeffreys_variance(covparam, lambda_var)


def neg_log_restricted_posterior_power_laws_prior(model, covparam, xi, zi):
    """
    Compute negative restricted posterior with power-laws prior.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    covparam : array_like
        Covariance parameter vector.
    xi, zi : array_like
        Observation inputs and targets.

    Returns
    -------
    scalar
        ``negative_log_restricted_likelihood - log_prior_power_law``.

    Notes
    -----
    This objective corresponds to MAP with a regularization prior, not an
    objective reference prior.
    """
    nlrl = model.negative_log_restricted_likelihood(covparam, xi, zi)
    return nlrl - log_prior_power_law(covparam)


def neg_log_restricted_posterior_gaussian_logsigma2_prior(
    model,
    covparam,
    xi,
    zi,
    log_sigma2_0,
    gamma=4.0,
):
    """
    Compute negative restricted posterior with Gaussian prior on ``log(sigma^2)``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    covparam : array_like
        Covariance parameter vector.
    xi, zi : array_like
        Observation inputs and targets.
    log_sigma2_0 : scalar
        Prior mean for ``log(sigma^2)``.
    gamma : float, default=4.0
        Prior scale parameter.

    Returns
    -------
    scalar
        ``negative_log_restricted_likelihood - log_prior_gaussian_logsigma2``.

    Notes
    -----
    This objective corresponds to MAP with a weakly informative prior on
    ``log(sigma^2)``.
    """
    nlrl = model.negative_log_restricted_likelihood(covparam, xi, zi)
    return nlrl - log_prior_gaussian_logsigma2(covparam, log_sigma2_0, gamma=gamma)


def neg_log_restricted_posterior_with_logrho_prior(
    model,
    covparam,
    xi,
    zi,
    logrho_min,
    logrho_0,
    alpha=1.0,
):
    """
    Compute negative restricted posterior with prior on ``logrho``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    covparam : array_like
        Covariance parameter vector.
    xi, zi : array_like
        Observation inputs and targets.
    logrho_min : array_like
        Lower bounds for ``logrho``.
    logrho_0 : array_like
        Reference values for ``logrho``.
    alpha : float, default=1.0
        Linear right-tail slope parameter.

    Returns
    -------
    scalar
        ``negative_log_restricted_likelihood - log_prior_logrho_barrier_linear``.

    Notes
    -----
    This objective corresponds to MAP with a constrained regularization prior on
    ``logrho``.
    """
    nlrl = model.negative_log_restricted_likelihood(covparam, xi, zi)
    return nlrl - log_prior_logrho_barrier_linear(
        covparam,
        logrho_min=logrho_min,
        logrho_0=logrho_0,
        alpha=alpha,
    )


def neg_log_restricted_posterior_gaussian_logsigma2_and_logrho_prior(
    model,
    covparam,
    xi,
    zi,
    log_sigma2_0,
    gamma=4.0,
    logrho_min=None,
    logrho_0=None,
    alpha=1.0,
):
    """
    Compute negative restricted posterior with priors on ``log(sigma^2)`` and ``logrho``.

    The objective is the REML criterion regularized by two additive prior terms:

    .. math::

        J(\\theta) =
        -\\log p(z \\mid x, \\theta)_{\\mathrm{REML}}
        - \\log p_{\\sigma^2}(\\theta)
        - \\log p_{\\rho}(\\theta),

    where ``theta = covparam``, ``log(sigma^2) = covparam[0]`` and
    ``logrho = -covparam[1:]``.

    The variance prior term ``log p_{\\sigma^2}`` is Gaussian in
    ``log(sigma^2)`` with center ``log_sigma2_0`` and scale controlled by
    ``gamma`` (one standard deviation corresponds to a multiplicative factor
    ``gamma`` on ``sigma^2``).

    The lengthscale term ``log p_{\\rho}`` is a barrier + linear-tail prior in
    ``logrho`` with componentwise hard support ``logrho > logrho_min`` and
    minimum at ``logrho_0``. The public parameter ``alpha`` controls the
    linear right-tail slope; the barrier strength is adjusted internally per
    component so the minimum remains at ``logrho_0``.

    Parameters
    ----------
    model : gpmp.core.Model
        GP model instance.
    covparam : array_like
        Covariance parameter vector.
    xi, zi : array_like
        Observation inputs and targets.
    log_sigma2_0 : scalar
        Gaussian prior center for ``log(sigma^2)``.
    gamma : float, default=4.0
        Gaussian prior scale parameter on ``log(sigma^2)``.
    logrho_min : array_like, optional
        Lower bounds for ``logrho`` components.
    logrho_0 : array_like, optional
        Reference values for ``logrho`` components.
    alpha : float, default=1.0
        Linear right-tail slope for the ``logrho`` prior.

    Returns
    -------
    scalar
        ``negative_log_restricted_likelihood`` minus both prior log-densities.

    """
    if logrho_min is None or logrho_0 is None:
        raise ValueError("logrho_min and logrho_0 must be provided.")

    nlrl = model.negative_log_restricted_likelihood(covparam, xi, zi)
    return (
        nlrl
        - log_prior_gaussian_logsigma2(covparam, log_sigma2_0, gamma=gamma)
        - log_prior_logrho_barrier_linear(
            covparam,
            logrho_min=logrho_min,
            logrho_0=logrho_0,
            alpha=alpha,
        )
    )
