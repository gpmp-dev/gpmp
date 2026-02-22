# gpmp/kernel/init.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Initialization heuristics for GP covariance parameters.
"""
from math import log
import gpmp.num as gnp
from .utils import prepare_data

def anisotropic_parameters_initial_guess_zero_mean(model, xi=None, zi=None, dataloader=None):
    """Anisotropic initialization with zero mean."""
    xi_, zi_, _n, d, source = prepare_data(xi, zi, dataloader)
    delta = (gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)) if source == "arrays" \
            else (dataloader.dataset_x_max - dataloader.dataset_x_min)
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta
    covparam = gnp.concatenate((gnp.array([gnp.log(1.0)]), -gnp.log(rho)))
    sigma2_GLS_fn = lambda x, z: 1.0 / x.shape[0] * model.norm_k_sqrd_with_zero_mean(x, z, covparam)
    sigma2_GLS = sigma2_GLS_fn(xi_, zi_) if source == "arrays" else dataloader.reduce_mean(sigma2_GLS_fn)
    return gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))

def anisotropic_parameters_initial_guess_constant_mean(model, xi=None, zi=None, dataloader=None):
    """Anisotropic initialization with parameterized constant mean."""
    xi_, zi_, n, d, source = prepare_data(xi, zi, dataloader)
    delta = (gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)) if source == "arrays" \
            else (dataloader.dataset_x_max - dataloader.dataset_x_min)
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta
    covparam = gnp.concatenate((gnp.array([gnp.log(1.0)]), -gnp.log(rho)))
    if source == "arrays":
        zTKinvz, Kinv1, Kinvz = model.k_inverses(xi_, zi_, covparam)
        mean_GLS = gnp.sum(Kinvz) / gnp.sum(Kinv1)
        sigma2_GLS = (1.0 / n) * zTKinvz
    else:
        def per_batch_gls(x, z):
            zTKinvz, Kinv1, Kinvz = model.k_inverses(x, z, covparam)
            return gnp.stack([gnp.sum(Kinvz) / gnp.sum(Kinv1), zTKinvz / x.shape[0]], axis=-1)
        mean_and_sigma2 = dataloader.reduce_mean(per_batch_gls)
        mean_GLS, sigma2_GLS = mean_and_sigma2[0], mean_and_sigma2[1]
    return mean_GLS.reshape(1), gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))

def anisotropic_parameters_initial_guess(model, xi=None, zi=None, dataloader=None):
    """Anisotropic initialization for general mean handling."""
    xi_, zi_, n, d, source = prepare_data(xi, zi, dataloader)
    delta = (gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)) if source == "arrays" \
            else (dataloader.dataset_x_max() - dataloader.dataset_x_min())
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta
    covparam = gnp.concatenate((gnp.array([log(1.0)]), -gnp.log(rho)))
    if source == "arrays":
        sigma2_GLS = (1.0 / n) * model.norm_k_sqrd(xi_, zi_, covparam)
    else:
        def per_batch_sigma2(x, z): return model.norm_k_sqrd(x, z, covparam) / x.shape[0]
        sigma2_GLS = dataloader.reduce_mean(per_batch_sigma2)
    return gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))
