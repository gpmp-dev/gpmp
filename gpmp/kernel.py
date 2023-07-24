## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
import time
import warnings
import numpy as np
import gpmp.num as gnp
from scipy.optimize import minimize, OptimizeWarning
from math import exp, log, sqrt


## -- kernels


def exponential_kernel(h):
    """
    Exponential kernel.

    The exponential kernel is defined as:

    .. math::

        k(h) = \exp(-h)

    where `h` represents the distances between points.

    Parameters
    ----------
    h : gnp.array, shape (n,)
        An array of distances between points.

    Returns
    -------
    gnp.array, shape (n,)
        An array of the exponential kernel values corresponding to the input distances.
    """
    return gnp.exp(-h)


def matern32_kernel(h):
    """
    Matérn 3/2 kernel.

    The Matérn 3/2 kernel is defined as:

    .. math::

        K(h) = (1 + 2\sqrt{3/2}h) \exp(-2\sqrt{3/2}h)

    where `h` represents the distances between points.

    Parameters
    ----------
    h : gnp.array, shape (n,)
        An array of distances between points.

    Returns
    -------
    gnp.array, shape (n,)
        An array of the Matérn 3/2 kernel values corresponding to the input distances.
    """
    nu = 3. / 2.
    c = 2. * sqrt(nu)
    t = c * h

    return (1.0 + t) * gnp.exp(-t)


def maternp_kernel(p: int, h):
    """Matérn kernel with half-integer regularity nu = p + 1/2.

    The Matérn kernel is defined as in Stein 1999, page 50:

    .. math::
        K(h) = \frac{1}{\Gamma(\nu) 2^{\nu - 1}} (\sqrt{2 \nu} h)^\nu K_\nu(\sqrt{2 \nu} h)

    where `h` represents the distances between points, `nu` is the
    regularity of the kernel, `K_nu` is the modified Bessel function
    of the second kind of order `nu`.

    In the particular case of half-integer regularity (nu = p + 1/2),
    the Matérn kernel simplifies to a product of an exponential term
    and a polynomial term (Watson 1922, A treatise on the theory of
    Bessel functions, pp. 80, Abramowitz and Stegun 1965, pp. 374-379,
    443-444):

    .. math::
        K(h) = \exp(-2\sqrt{\nu}h) \frac{\Gamma(p+1)}{\Gamma(2p+1)}\sum_{i=0}^{p} \frac{(p+i)!}{i!(p-i)!} (4\sqrt{\nu}h)^{p-i}

    The implementation provided in this function uses this
    half-integer simplification.

    """
    h = gnp.inftobigf(h)
    c = 2.0 * sqrt(p + 0.5)
    twoch = 2.0 * c * h
    polynomial = gnp.ones(h.shape)
    a = gnp.gammaln(p + 1) - gnp.gammaln(2 * p + 1)
    for i in gnp.arange(p):
        log_combination = (
            a
            + gnp.gammaln(p + i + 1)
            - gnp.gammaln(i + 1)
            - gnp.gammaln(p - i + 1)
        )
        polynomial += gnp.exp(log_combination) * twoch**(p-i)

    return gnp.exp(-c * h) * polynomial


def maternp_covariance_ii_or_tt(x, p, param, pairwise=False):
    """
    Covariance between observations or predictands at x.

    The covariance matrix is computed using the Matérn kernel with half-integer regularity:

    .. math::

        K_{ij} = \sigma^2  K(h_{ij}) + \epsilon \delta_{ij}

    where `K(h_{ij})` is the Matérn kernel value for the distance `h_{ij}` between points `x_i` and `x_j`,
    `sigma^2` is the variance, `delta_{ij}` is the Kronecker delta, and `epsilon` is a small positive constant.

    Parameters
    ----------
    x : gnp.array, shape (nx, d)
        Observation points.
    p : int
        Half-integer regularity nu = p + 1/2.
    param : gnp.array, shape (1 + d,)
        sigma2 and range parameters.
    pairwise : bool, optional
        Whether to return a covariance matrix k(x_i, x_j),
        for i and j = 1 ... nx, if pairwise is False, or a covariance
        vector k(x_i, x_i) if pairwise is True. Default is False.

    Returns
    -------
    K : gnp.array
        Covariance matrix (nx, nx) or covariance vector if pairwise is True.
    """
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1:]
    nugget = 10.0 * gnp.finfo(gnp.float64).eps

    if pairwise:
        # return a vector of covariances
        K = sigma2 * gnp.ones((x.shape[0],))  # nx x 0
    else:
        # return a covariance matrix
        K = gnp.scaled_distance(loginvrho, x, x)  # nx x nx
        K = sigma2 * maternp_kernel(p, K) + nugget * gnp.eye(K.shape[0])

    return K


def maternp_covariance_it(x, y, p, param, pairwise=False):
    """
    Covariance between observations and prediction points.

    Parameters
    ----------
    x : ndarray, shape (nx, d)
        Observation points.
    y : ndarray, shape (ny, d)
        Observation points.
    p : int
        Half-integer regularity nu = p + 1/2.
    param : ndarray, shape (1 + d,)
        log(sigma2) and log(1/range) parameters.
    pairwise : bool, optional
        Whether to return a covariance matrix k(x_i, y_j),
        for i in 1 ... nx and j in 1 ... ny, if pairwise is False,
        or a covariance vector k(x_i, y_i) if pairwise is True. Default is False.

    Returns
    -------
    K : ndarray
        Covariance matrix (nx, ny) or covariance vector if pairwise is True.
    """
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1:]

    if pairwise:
        # return a vector of distances
        K = gnp.scaled_distance_elementwise(loginvrho, x, y)  # nx x 0
    else:
        # return a distance matrix
        K = gnp.scaled_distance(loginvrho, x, y)  # nx x ny

    K = sigma2 * maternp_kernel(p, K)

    return K


def maternp_covariance(x, y, p, param, pairwise=False):
    """
    Matérn covariance function with half-integer regularity nu = p + 1/2.

    The kernel is defined in terms of the Euclidean distance, between
    pairs of input points. For the Matérn kernel, the distance measure
    is scaled by a length scale parameter, which controls how much
    influence distant points have on each other. The kernel has two
    hyperparameters: the length scale and a smoothness parameter,
    which is typically an integer or half-integer value.
    
    Parameters
    ----------
    x : ndarray, shape (nx, d)
        Observation points.
    y : ndarray, shape (ny, d) or None
        Prediction points. If None, it is assumed that y is x.
    p : int
        Half-integer regularity nu = p + 1/2.
    param : ndarray, shape (1 + d)
        Covariance parameters
        [log(sigma2) log(1/rho_1) log(1/rho_2) ...].
    pairwise : bool, optional
        If True, return a covariance vector k(x_i, y_i). If False,
        return a covariance matrix k(x_i, y_j) for i in the range 1 to nx
        and j in the range 1 to ny. Default is False.

    Returns
    -------
    K : ndarray
        Covariance matrix (nx, ny) or covariance vector if pairwise is True.

    Notes
    -----
    An isotropic covariance is obtained if param = [log(sigma2) log(1/rho)]
    (only one length scale parameter).

    """
    if y is x or y is None:
        return maternp_covariance_ii_or_tt(x, p, param, pairwise)
    else:
        return maternp_covariance_it(x, y, p, param, pairwise)


## -- parameters


def anisotropic_parameters_initial_guess_with_zero_mean(model, xi, zi):
    """anisotropic initialization strategy with zero mean

    See anisotropic_parameters_initial_guess
    """
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi)

    delta = gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)
    d = xi_.shape[1]
    rho = (gnp.exp(gnp.gammaln(d/2+1))/gnp.pi**(d/2))**(1/d) * delta

    covparam = gnp.concatenate((gnp.array([log(1.0)]), -gnp.log(rho)))
    n = xi_.shape[0]
    sigma2_GLS = (
        1.0 / n * model.norm_k_sqrd_with_zero_mean(xi_, zi_.reshape((-1,)), covparam)
    )

    return gnp.concatenate((gnp.array([gnp.log(sigma2_GLS)]), -gnp.log(rho)))


def anisotropic_parameters_initial_guess(model, xi, zi):
    """
    Anisotropic initialization strategy.

    Parameters
    ----------
    model : object
        An instance of a Gaussian process model.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.

    Returns
    -------
    initial_params : ndarray
        Initial guess for anisotropic parameters.

    References
    ----------
    .. [1] Basak, S., Petit, S., Bect, J., & Vazquez, E. (2021).
       Numerical issues in maximum likelihood parameter estimation for
       Gaussian process interpolation. arXiv:2101.09747.
    """

    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi)

    delta = gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)
    d = xi_.shape[1]
    # the volume of a ball in dim d is V_d(R) = \frac{\pi^{d/2} R^d}{\Gamma(d/2+1)}
    # taking R = rho / 2, compute rho such that V_d(R) = 2 * delta
    rho = (gnp.exp(gnp.gammaln(d/2+1))/gnp.pi**(d/2))**(1/d) * delta
    
    covparam = gnp.concatenate((gnp.array([log(1.0)]), -gnp.log(rho)))
    n = xi_.shape[0]
    sigma2_GLS = 1.0 / n * model.norm_k_sqrd(xi_, zi_.reshape((-1,)), covparam)

    return gnp.concatenate((gnp.array([gnp.log(sigma2_GLS)]), -gnp.log(rho)))


def make_selection_criterion_with_gradient(selection_criterion, xi, zi):
    """
    Make selection criterion function with gradient.

    Parameters
    ----------
    selection_criterion : function
        Selection criterion function.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.

    Returns
    -------
    crit_jit : function
        Selection criterion function with gradient.
    dcrit : function
        Gradient of the selection criterion function.
    """
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi)

    # selection criterion
    def crit_(covparam):
        l = selection_criterion(covparam, xi_, zi_)
        return l

    crit_jit = gnp.jax.jit(crit_)

    dcrit = gnp.jax.jit(gnp.grad(crit_jit))

    return crit_jit, dcrit


def autoselect_parameters(p0,
                          criterion,
                          gradient,
                          bounds=None,
                          silent=True,
                          info=False,
                          method='SLSQP'):
    """
    Automatic parameters selection.

    Parameters
    ----------
    p0 : ndarray
        Initial guess of the parameters.
    criterion : function
        Selection criterion function.
    gradient : function
        Gradient of the selection criterion function.
    bounds : sequence, optional
        Sequence of (min, max) pairs for each parameter. 
        None is used to specify no bound.
    silent : bool, optional
        If True, suppresses output messages. Default is True.
    info : bool, optional
        If True, returns additional information. Default is False.

    Returns
    -------
    best : ndarray
        Best parameters found by the optimization.
    r : object, optional
        Additional information about the optimization (if info=True).
    """
    tic = time.time()
    if gnp._gpmp_backend_ == 'jax':
        # scipy.optimize.minimize cannot use jax arrays
        crit_asnumpy = criterion
        gradient_asnumpy = lambda p: gnp.to_np(gradient(p))
    elif gnp._gpmp_backend_ == 'torch':
        def crit_asnumpy(p):
            v = criterion(gnp.asarray(p))
            return v.detach().item()
        def gradient_asnumpy(p):
            g = gradient(gnp.asarray(p))
            if g is None:
                return gnp.zeros(p.shape)
            else:
                return g
    elif gnp._gpmp_backend_ == 'numpy':
        def crit_asnumpy(p):
            try:
                J = criterion(p)
            except:
                J = np.Inf
            return J
        gradient_asnumpy = None

    if method == 'L-BFGS-B':
        options = {
            "disp": False,
            "maxcor": 20,
            "ftol": 1e-06,
            "gtol": 1e-05,
            "eps": 1e-08,
            "maxfun": 15000,
            "maxiter": 15000,
            "iprint": -1,
            "maxls": 40,
            "finite_diff_rel_step": None,
        }
    elif method == 'SLSQP':
        options = {
            "disp": False,
            "ftol": 1e-06,
            "eps": 1e-08,
            "maxiter": 15000,
            "finite_diff_rel_step": None,
        }
    else:
        raise ValueError('Optmization method not implemented.')
    
    if silent is False:
        options["disp"] = True

    r = minimize(
        crit_asnumpy,
        p0,
        args=(),
        method=method,
        jac=gradient_asnumpy,
        bounds=bounds,
        tol=None,
        callback=None,
        options=options,
    )
    
    best = r.x
    if silent is False:
        print("Gradient")
        print("--------")
        if gradient_asnumpy is not None:
            print(gradient_asnumpy(best))
        else:
            print('gradient not available')    
        print(".")
    r.covparam0 = p0
    r.covparam = best
    r.selection_criterion = criterion
    r.time = time.time() - tic

    if info:
        return best, r
    else:
        return best


def select_parameters_with_reml(model, xi, zi, covparam0=None, info=False, verbosity=0):
    """Parameters selection with Restricted Maximum Likelihood (REML).

    Parameters
    ----------
    model : object
        Gaussian process model object.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    covparam0 : ndarray, shape (covparam_dim,), optional
        Initial guess for the covariance parameters. If None,
        anisotropic_parameters_initial_guess is used. Default is None.
    info : bool, optional
        If True, returns additional information. Default is False.
    verbosity : 0, 1, 2, optional
        If 0, suppresses output messages. Default is 0.

    Returns
    -------
    model : object
        Updated Gaussian process model object with optimized parameters.
    info_ret : dict, optional
        Additional information about the optimization (if info=True).

    """
    tic = time.time()

    if covparam0 is None:
        covparam0 = anisotropic_parameters_initial_guess(model, xi, zi)

    nlrl, dnlrl = make_selection_criterion_with_gradient(
        model.negative_log_restricted_likelihood, xi, zi
    )

    silent = True
    if verbosity == 1:
        print('Parameter selection...')
    elif verbosity == 2:
        silent = False

    covparam_reml, info_ret = autoselect_parameters(
        covparam0, nlrl, dnlrl, silent=silent, info=True
    )
    
    if verbosity == 1:
        print('done.')

    # NB: info is essentially a dict with attribute accessors

    model.covparam = gnp.asarray(covparam_reml)

    if info:
        info_ret["covparam0"] = covparam0
        info_ret["covparam"] = covparam_reml
        info_ret["selection_criterion"] = nlrl
        info_ret["time"] = time.time() - tic
        return model, info_ret
    else:
        return model


def update_parameters_with_reml(model, xi, zi, info=False):
    """
    Update model parameters with Restricted Maximum Likelihood (REML).

    Parameters
    ----------
    model : object
        Gaussian process model object.
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    info : bool, optional
        If True, returns additional information. Default is False.

    Returns
    -------
    model : object
        Updated Gaussian process model object with optimized parameters.
    info_ret : dict, optional
        Additional information about the optimization (if info=True).
    """
    tic = time.time()

    covparam0 = model.covparam

    nlrl, dnlrl = make_selection_criterion_with_gradient(
        model.negative_log_restricted_likelihood, xi, zi
    )

    covparam_reml, info_ret = autoselect_parameters(
        covparam0, nlrl, dnlrl, silent=True, info=True
    )

    model.covparam = covparam_reml

    if info:
        info_ret["covparam0"] = covparam0
        info_ret["covparam"] = covparam_reml
        info_ret["selection_criterion"] = nlrl
        info_ret["time"] = time.time() - tic
        return model, info_ret
    else:
        return model
    # --- end if
