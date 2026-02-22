# gpmp/kernel/priors.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Priors used in GP kernel parameter estimation.
"""
import gpmp.num as gnp

def log_prior_jeffrey_variance(covparam, lambda_var=1.0):
    """Jeffreys prior on variance: log p = -lambda_var * log(sigma^2)."""
    return -lambda_var * covparam[0]

def log_prior_power_law(
    covparam,
    lambda_var=1.0,
    cut_logvariance_high=9.21,   # ~ log(1e4)
    lambda_lengthscales=0.0,
    cut_loginvrho_low=-9.21,
    cut_loginvrho_high=9.21,
    penalty_factor=100,
):
    """Power-law priors with soft penalties on too-large/small inverse length-scales."""
    log_sigma2 = covparam[0]
    p = covparam[1:]
    log_prior_sigma2 = -lambda_var * log_sigma2
    extra_sigma2 = penalty_factor * gnp.maximum(log_sigma2 - cut_logvariance_high, 0)
    extra_low = penalty_factor * gnp.maximum(cut_loginvrho_low - p, 0)
    extra_high = penalty_factor * gnp.maximum(p - cut_loginvrho_high, 0)
    log_prior_lengths = -lambda_lengthscales * gnp.sum(p) - gnp.sum(extra_low) - gnp.sum(extra_high)
    return log_prior_sigma2 + extra_sigma2 + log_prior_lengths

def log_prior_reference(model, covparam, xi):
    """Reference prior: 0.5 * log det(FisherInfo(theta))."""
    fisher_info = model.fisher_information(xi, covparam)
    return 0.5 * gnp.logdet(fisher_info)

def neg_log_restricted_posterior_with_jeffreys_prior(model, covparam, xi, zi, lambda_var=1.0):
    """-log posterior = nlrl - log_prior(Jeffreys on sigma^2)."""
    nlrl = model.negative_log_restricted_likelihood(covparam, xi, zi)
    return nlrl - log_prior_jeffrey_variance(covparam, lambda_var)

def neg_log_restricted_posterior_with_power_law_prior(model, covparam, xi, zi):
    """-log posterior = nlrl - log_prior(power-law on sigma^2 and inv length-scales)."""
    nlrl = model.negative_log_restricted_likelihood(covparam, xi, zi)
    return nlrl - log_prior_power_law(covparam)
