## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023-2024, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
import gpmp.num as gnp




def crps_gaussian(mu, sigma, z):
    """
    Compute the CRPS for the Gaussian case.

    Parameters
    ----------
    mu : ndarray, shape (n,)
        Mean values of the Gaussian distribution.
    sigma : ndarray, shape (n,)
        Standard deviations of the Gaussian distribution.
    z : ndarray, shape (n,)
        Observations.

    Returns
    -------
    crps : ndarray, shape (n,)
        The computed CRPS values for each element in mu, sigma, and z.
    """
    mu = gnp.asarray(mu)
    sigma = gnp.asarray(sigma)
    z = gnp.asarray(z)
    
    t = (z - mu) / sigma
    term1 = t * (2 * gnp.normal.cdf(t) - 1)
    term2 = 2 * gnp.normal.pdf(t)
    term3 = 1 / gnp.sqrt(gnp.pi)
    crps = sigma * (term1 + term2 - term3)
    return crps


def h1(t):
    return t * gnp.normal.cdf(t) + gnp.normal.pdf(t)


def ei1_up(mu, sigma, z):
    """
    Compute the expected improvement EI1_up(P, z) for a Gaussian distribution.

    Parameters:
    -----------
    mu: float or np.ndarray
        The mean(s) of the Gaussian distribution(s).
    sigma: float or np.ndarray
        The standard deviation(s) of the Gaussian distribution(s).
    z: float or np.ndarray
        The input value(s).

    Returns:
    --------
    result: float or np.ndarray
        The EI1_up value(s) for the given Gaussian distribution(s).
    """
    t = (mu - z) / sigma
    return sigma * h1(t)


def ei2_up(mu, sigma, z):
    """
    Compute the expected improvement EI2_up(P, z) for a Gaussian distribution.

    Parameters:
    -----------
    mu: float or np.ndarray
        The mean(s) of the Gaussian distribution(s).
    sigma: float or np.ndarray
        The standard deviation(s) of the Gaussian distribution(s).
    z: float or np.ndarray
        The input value(s).

    Returns:
    --------
    result: float or np.ndarray
        The EI2_up(P, z) value(s) for the given Gaussian distribution(s).
    """
    t = (mu - z) / sigma
    if gnp.isscalar(t):
        t = gnp.array([t])
    delta_2_t = gnp.hstack((t.reshape(-1, 1), gnp.zeros((t.shape[0], 1))))

    D_2 = gnp.array([[-1, 0], [-1, 1]])

    term1 = 2.0 * t * gnp.multivariate_normal.cdf(
        delta_2_t, mean=gnp.zeros(2), cov=gnp.matmul(D_2, D_2.T)
    )
    term2 = 2.0 * gnp.normal.pdf(t) * gnp.normal.cdf(-t)
    term3 = 1.0 / gnp.sqrt(gnp.pi) * gnp.normal.cdf(t, loc=0.0, scale=gnp.sqrt(0.5))

    return sigma * (term1 + term2 + term3)


def h1(t):
    return t * gnp.normal.cdf(t) + gnp.normal.pdf(t)


def tcrps_gaussian(mu, sigma, z, a=-gnp.inf, b=gnp.inf):
    """
    Compute S_{a, b}^\tCRPS (P, z) for Gaussian distributions.

    Parameters:
    -----------
    mu: float or np.ndarray
        The mean(s) of the Gaussian distribution(s).
    sigma: float or np.ndarray
        The standard deviation(s) of the Gaussian distribution(s).
    z: float or np.ndarray
        The observed values.
    a: float or np.ndarray, optional
        The lower bound of the interval, default = -inf.
    b: float or np.ndarray, optional
        The upper bound of the interval, default = +inf.

    Returns:
    --------
    result: float or np.ndarray
        The S_{a, b}^\tCRPS (P, z) value(s) for the given Gaussian distribution(s).
    """
    mu = gnp.asarray(mu)
    sigma = gnp.asarray(sigma)
    z = gnp.asarray(z)
    a = gnp.asarray(a)
    b = gnp.asarray(b)
    if gnp.isfinite(a) and gnp.isfinite(b):
        term1 = gnp.maximum(gnp.minimum(b, z) - a, 0.0)
        term2 = ei2_up(mu, sigma, b) - ei2_up(mu, sigma, a)
        term3 = -2 * gnp.where(
            z <= b, ei1_up(mu, sigma, b) - ei1_up(mu, sigma, gnp.maximum(a, z)), 0
        )
        return term1 + term2 + term3
    elif ~gnp.isfinite(a) and gnp.isfinite(b):
        term1 = gnp.minimum(b, z)
        term2 = ei2_up(mu, sigma, b) - (mu + sigma / gnp.sqrt(gnp.pi))
        term3 = -2 * gnp.where(z <= b, ei1_up(mu, sigma, b) - ei1_up(mu, sigma, z), 0)
        return term1 + term2 + term3
    elif gnp.isfinite(a) and ~gnp.isfinite(b):
        return tcrps_gaussian(-mu, sigma, -z, a=-gnp.inf, b=-a)
    else:
        return crps_gaussian(mu, sigma, z)
