# gpmp/kernel/matern.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
from math import sqrt
import gpmp.num as gnp

def matern32_kernel(h):
    """Matérn 3/2 kernel.

    .. math::
        K(h) = (1 + 2\\sqrt{3/2}\\,h) \\exp(-2\\sqrt{3/2}\\,h)

    Parameters
    ----------
    h : gnp.array, shape (n,)
        Distances between points.

    Returns
    -------
    gnp.array, shape (n,)
        Kernel values.
    """
    nu = 3.0 / 2.0
    c = 2.0 * sqrt(nu)
    t = c * h
    return (1.0 + t) * gnp.exp(-t)


def maternp_kernel(p: int, h):
    """Matérn kernel with half-integer regularity :math:`\\nu = p + 1/2`.

    Using the half-integer simplification (Watson 1922; Abramowitz & Stegun):

    .. math::
        K(h) = \\exp(-2\\sqrt{\\nu}\\,h)\\,
               \\frac{\\Gamma(p+1)}{\\Gamma(2p+1)}
               \\sum_{i=0}^{p} \\frac{(p+i)!}{i!(p-i)!}\\,(4\\sqrt{\\nu}h)^{\\,p-i}

    Parameters
    ----------
    p : int
        Nonnegative integer with :math:`\\nu = p+1/2`.
    h : gnp.array
        Distances.

    Returns
    -------
    gnp.array
        Kernel values.
    """
    gln = gnp.compute_gammaln(p)  # expects integer table access
    h = gnp.inftobigf(h)
    c = 2.0 * sqrt(p + 0.5)
    twoch = 2.0 * c * h
    polynomial = gnp.ones(h.shape)
    for i in range(p):
        exp_log_combination = gnp.exp(
            gln[p + 1] - gln[2 * p + 1] + gln[p + i + 1] - gln[i + 1] - gln[p - i + 1]
        )
        polynomial += exp_log_combination * (twoch ** (p - i))
    return gnp.exp(-c * h) * polynomial


def maternp_covariance_ii_or_tt(x, p, param, pairwise=False):
    """Covariance between observations or predictands at x.

    .. math::
        K_{ij} = \\sigma^2 K(h_{ij}) + \\epsilon \\delta_{ij}

    Parameters
    ----------
    x : gnp.array, shape (n, d)
    p : int
        Half-integer regularity :math:`\\nu = p + 1/2`.
    param : gnp.array, shape (1 + d,)
        [log(sigma2), log(1/rho_j)].
    pairwise : bool
        If True, return diag vector; else full covariance.

    Returns
    -------
    gnp.array
        (n,n) matrix or (n,) vector if pairwise.
    """
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1:]
    nugget = 10.0 * sigma2 * gnp.finfo(gnp.float64).eps
    if pairwise:
        return sigma2 * gnp.ones((x.shape[0],))
    K = gnp.scaled_distance(loginvrho, x, x)
    return sigma2 * maternp_kernel(p, K) + nugget * gnp.eye(K.shape[0])


def maternp_covariance_it(x, y, p, param, pairwise=False):
    """Cross-covariance between observations x and prediction points y.

    Parameters
    ----------
    x : gnp.array, shape (nx, d)
    y : gnp.array, shape (ny, d)
    p : int
    param : gnp.array, shape (1 + d,)
        [log(sigma2), log(1/rho_j)].
    pairwise : bool
        If True, return elementwise k(x_i,y_i); else (nx,ny).

    Returns
    -------
    gnp.array
        (nx,ny) matrix or (n,) vector if pairwise.
    """
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1:]
    if pairwise:
        D = gnp.scaled_distance_elementwise(loginvrho, x, y)
    else:
        D = gnp.scaled_distance(loginvrho, x, y)
    return sigma2 * maternp_kernel(p, D)


def maternp_covariance(x, y, p, param, pairwise=False):
    """Matérn covariance (:math:`\\nu = p+1/2`). Wrapper.

    Parameters
    ----------
    x : gnp.array, shape (nx, d)
    y : gnp.array or None
    p : int
    param : gnp.array, shape (1 + d,)
    pairwise : bool

    Returns
    -------
    gnp.array
    """
    if y is x or y is None:
        return maternp_covariance_ii_or_tt(x, p, param, pairwise)
    return maternp_covariance_it(x, y, p, param, pairwise)
