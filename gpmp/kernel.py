## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.special import gammaln
from math import exp, sqrt

## -- distance


def scale(x, invrho):
    """Scale input points

    Parameters
    ----------
    x : ndarray(n, d)
        Observation points
    invrho : ndarray(d), or scalar
        Inverse of scaling factors

    Returns
    -------
    xs : ndarray(n, d)
        [ x_{1,1} * invrho_1 ... x_{1,d} * invrho_d 
          ...
          x_{n,1} * invrho_1 ... x_{n,d} * invrho_d ]

    Note : If invrho is a scalar the scaling is isotropic
    """
    return invrho * x


@jax.jit
def distance(x, y, alpha=1e-8):
    """Compute a distance matrix

    Parameters
    ----------
    x : numpy.array(n,dim)
        _description_
    y : numpy.array(m,dim)
        If y is None, it is assumed y is x, by default None
    alpha : float, optional
        a small number to prevent auto-differentation problems
        with the derivative of sqrt at zero, by default 1e-8

    Returns
    -------
    numpy.array(n,m)
        distance matrix such that
    .. math:: d_{i,j} = (alpha + sum_{k=1}^dim (x_{i,k} - y_{i,k})^2)^(1/2)
    
    Notes
    -----
    in practice however, it seems that it makes no performance
    improvement; FIXME: investigate memory and CPU usage

    """
    if y is None:
        y = x

    y2 = jnp.sum(y**2, axis=1)

    # Debug: check if x is y
    # print("&x = {}, &y = {}".format(hex(id(x)), hex(id(y))))

    if x is y:
        d = jnp.sqrt(alpha + jnp.reshape(y2, [-1, 1]) + y2 -
                     2 * jnp.inner(x, y))
    else:
        x2 = jnp.reshape(jnp.sum(x**2, axis=1), [-1, 1])
        d = jnp.sqrt(alpha + x2 + y2 - 2 * jnp.inner(x, y))

    return d


@jax.jit
def distance_pairwise(x, y, alpha=1e-8):
    '''Compute a distance vector between the pairs (xi, yi)

    Inputs
      * x: numpy array n x dim
      * y: numpy array n x dim or None
      * alpha: a small number to prevent auto-differentation problems
        with the derivative of sqrt at zero

    If y is None, it is assumed y is x

    Output
      * distance vector of size n x 1 such that
        d_i = (alpha + sum_{k=1}^dim (x_{i,k} - y_{i,k})^2)^(1/2)
        or
        d_i = 0 if y is x or None

    '''
    if x is y or y is None:
        d = jnp.zeros((x.shape[0], ))
    else:
        d = jnp.sqrt(alpha + jnp.sum((x - y)**2, axis=1))

    return d


## -- kernels


def exponential_kernel(h):
    """exponential kernel

    Parameters
    ----------
    h : numpy.array
        _description_

    Returns
    -------
    numpy.array
        _description_
    """
    return jnp.exp(-h)


def matern32_kernel(h):
    """Matérn 3/2 kernel

    Parameters
    ----------
    h : numpy.array
        _description_

    Returns
    -------
    numpy.array
        _description_
    """
    nu = 3 / 2
    c = 2 * sqrt(nu)
    t = c * h

    return (1 + t) * jnp.exp(-t)


@partial(jax.jit, static_argnums=0)
def maternp_kernel(p, h):
    """Matérn kernel with half-integer regularity nu = p + 1/2

    See Stein, M. E., 1999, pp. 50, and Abramowitz and Stegun 1965,
    pp. 374-379, 443-444, Rasmussen and Williams 2006, pp. 85

    Parameters
    ----------
    p : int
        order
    h : ndarray(n)
        distance

    Returns
    -------
    k : ndarray(n)
        Values of the Matérn kernel at h

    """
    c = 2 * jnp.sqrt(p + 1 / 2)
    polynomial = 0
    for i in range(p + 1):
        polynomial = polynomial + (2 * c * h) ** (p - i) \
            * exp(gammaln(p + 1) - gammaln(2 * p + 1) \
                  + gammaln(p + i + 1) - gammaln(i + 1) - gammaln(p - i + 1))
    return jnp.exp(-c * h) * polynomial


def maternp_covariance_ii_or_tt(x, p, param, pairwise=False):
    """Covariance between observations or predictands at x

       Parameters
       ----------
       x : ndarray(nx, d)
           observation points
       p : int
           half-integer regularity nu = p + 1/2
       param : ndarray(1 + d)
           sigma2 and range parameters
       pairwise : boolean
           whether to return a covariance matrix k(x_i, x_j),
           for i and j = 1 ... nx, if pairwise is False, or a covariance
           vector k(x_i, x_i) if pairwise is True
    """
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1:])
    nugget = 10 * jnp.finfo(jnp.float64).eps

    if pairwise:
        K = sigma2 * jnp.ones((x.shape[0], ))  # nx x 0
    else:
        xs = scale(x, invrho)
        K = distance(xs, xs)  # nx x nx
        K = sigma2 * maternp_kernel(p, K) + nugget * jnp.eye(K.shape[0])

    return K

def maternp_covariance_it(x, y, p, param, pairwise=False):
    """Covariance between observations and prediction points

       Parameters
       ----------
       x : ndarray(nx, d)
           observation points
       y : ndarray(ny, d)
           observation points
       p : int
           half-integer regularity nu = p + 1/2
       param : ndarray(1 + d)
           log(sigma2) and log(1/range) parameters
       pairwise : boolean
           whether to return a covariance matrix k(x_i, y_j),
           for i in 1 ... nx and j in 1 ... ny, if pairwise is False,
           or a covariance vector k(x_i, y_i) if pairwise is True
    """
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1:])

    xs = scale(x, invrho)
    ys = scale(y, invrho)
    if pairwise:
        K = distance_pairwise(xs, ys)  # nx x 0
    else:
        K = distance(xs, ys)  # nx x ny

    K = sigma2 * maternp_kernel(p, K)

    return K
    

def maternp_covariance(x, y, p, param, pairwise=False):
    """Matérn covariance function with half-integer regularity nu = p + 1/2

    Parameters
    ----------
       x : ndarray(nx, d)
           Observation points
       y : ndarray(ny, d) or None
           Observation points. If None, it is assumed that y is x
       p : int
           Half-integer regularity nu = p + 1/2
       param : ndarray(1 + d)
           Covariance parameters
           [log(sigma2) log(1/rho_1) log(1/rho_2) ...]
       pairwise : boolean
           Whether to return a covariance matrix k(x_i, y_j),
           for i in 1 ... nx and j in 1 ... ny, if pairwise is False,
           or a covariance vector k(x_i, y_i) if pairwise is True

    Returns
    -------
    Covariance matrix (nx , ny) or covariance vector if pairwise is True
    
    Notes
    -----
    An isotropic covariance is obtained if param = [log(sigma2) log(1/rho)]
    (only one length scale parameter)
    """
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1:])
    nugget = 10 * jnp.finfo(jnp.float64).eps

    if y is x or y is None:
        return maternp_covariance_ii_or_tt(x, p, param, pairwise)
    else:
        return maternp_covariance_it(x, y, p, param, pairwise)


## -- parameters


def anisotropic_parameters_initial_guess_with_zero_mean(model, xi, zi):
    """anisotropic initialization strategy with zero mean

    Parameters
    ----------
    model : _type_
        _description_
    xi : _type_
        _description_
    zi : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    
    References
    ----------
    .. [1] Basak, S., Petit, S., Bect, J., & Vazquez, E. (2021).
       Numerical issues in maximum likelihood parameter estimation for
       Gaussian process interpolation. arXiv:2101.09747.
    """
    rho = jnp.std(xi, axis=0)
    covparam = jnp.concatenate((jnp.array([jnp.log(1.0)]), -jnp.log(rho)))
    n = xi.shape[0]
    sigma2_GLS = 1 / n * model.norm_k_sqrd_with_zero_mean(xi, zi.reshape((-1,)), covparam)

    return jnp.concatenate((jnp.array([jnp.log(sigma2_GLS)]), -jnp.log(rho)))


def anisotropic_parameters_initial_guess(model, xi, zi):
    """anisotropic initialization strategy

    Parameters
    ----------
    model : _type_
        _description_
    xi : _type_
        _description_
    zi : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    References
    ----------
    [1] Basak, S., Petit, S., Bect, J., & Vazquez, E. (2021).
       Numerical issues in maximum likelihood parameter estimation for
       Gaussian process interpolation. arXiv:2101.09747.
    """
    rho = jnp.std(xi, axis=0)
    covparam = jnp.concatenate((jnp.array([jnp.log(1.0)]), -jnp.log(rho)))
    n = xi.shape[0]
    sigma2_GLS = 1 / n * model.norm_k_sqrd(xi, zi.reshape((-1,)), covparam)

    return jnp.concatenate((jnp.array([jnp.log(sigma2_GLS)]), -jnp.log(rho)))


def autoselect_parameters(p0, criterion, gradient):
    """Automatic parameters selection

    Parameters
    ----------
    p0 : _type_
        _description_
    criterion : _type_
        _description_
    gradient : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # scipy.optimize.minimize cannot use jax arrays
    if isinstance(p0, jax.numpy.ndarray):
        p0 = jnp.asarray(p0)
    gradient_asnumpy = lambda p: np.array(jnp.asarray(gradient(p)))

    r = minimize(criterion,
                 p0,
                 args=(),
                 method='L-BFGS-B',
                 jac=gradient_asnumpy,
                 bounds=None,
                 tol=None,
                 callback=None,
                 options={
                     'disp': True,
                     'maxcor': 20,
                     'ftol': 1e-06,
                     'gtol': 1e-05,
                     'eps': 1e-08,
                     'maxfun': 15000,
                     'maxiter': 15000,
                     'iprint': -1,
                     'maxls': 40,
                     'finite_diff_rel_step': None
                 })

    best = r.x

    return best


def print_sigma_rho(covparam):
    print("sigma      : {}".format(jnp.exp(0.5 * covparam[0])))
    rho_str = "rho [ {:2d} ] : {}".format(0, jnp.exp(-covparam[1]))
    for i in range(covparam.size - 2):
        rho_str += "\n    [ {:2d} ] : {}".format(i + 1,
                                                 jnp.exp(-covparam[i + 2]))
    print(rho_str)
