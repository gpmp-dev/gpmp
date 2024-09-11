## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2024, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
import time
import numpy as np
import gpmp.num as gnp
from scipy.optimize import minimize, OptimizeWarning
from math import log, sqrt


## -- kernels


def exponential_kernel(h):
    """Exponential kernel.

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
    """Matérn 3/2 kernel.

    The Matérn 3/2 kernel is defined as:

    .. math::

        K(h) = (1 + 2\\sqrt{3/2}h) \\exp(-2\\sqrt{3/2}h)

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
    nu = 3.0 / 2.0
    c = 2.0 * sqrt(nu)
    t = c * h

    return (1.0 + t) * gnp.exp(-t)


def compute_gammaln(up_to_p):
    """Compute gammaln values."""
    return [gnp.asarray(gnp.gammaln(i)) for i in range(2 * up_to_p + 2)]


gln = []
pmax = -1


def maternp_kernel(p: int, h):
    """Matérn kernel with half-integer regularity nu = p + 1/2.

    The Matérn kernel is defined as in Stein 1999, page 50:

    .. math::

        K(h) = \\frac{1}{\\Gamma(\\nu) 2^{\\nu - 1}} (\\sqrt{2 \\nu} h)^\\nu K_\\nu(\\sqrt{2 \\nu} h)

    Where:
    - :math:`h` represents the distances between points.
    - :math:`\\nu` is the regularity of the kernel.
    - :math:`K_\\nu` is the modified Bessel function of the second kind of order :math:`\\nu`.

    The implementation provided in this function uses this half-integer simplification.

    In the particular case of half-integer regularity (nu = p + 1/2), the Matérn kernel
    simplifies to a product of an exponential term and a polynomial term (Watson 1922,
    A treatise on the theory of Bessel functions, pp. 80, Abramowitz and Stegun 1965, pp. 374-379, 443-444):

    .. math::

        K(h) = \\exp(-2\\sqrt{\\nu}h) \\frac{\\Gamma(p+1)}{\\Gamma(2p+1)} \\sum_{i=0}^{p} \\frac{(p+i)!}{i!(p-i)!} (4\\sqrt{\\nu}h)^{p-i}
    """
    global gln, pmax

    # Check if p exceeds pmax and compute gammaln cache if needed
    if p > pmax:
        gln = compute_gammaln(p)
        pmax = p

    h = gnp.inftobigf(h)
    c = 2.0 * sqrt(p + 0.5)
    twoch = 2.0 * c * h
    polynomial = gnp.ones(h.shape)

    for i in range(p):
        exp_log_combination = gnp.exp(
            gln[p + 1] - gln[2 * p + 1] + gln[p + i + 1] - gln[i + 1] - gln[p - i + 1]
        )
        twoch_pow = twoch ** (p - i)
        polynomial += exp_log_combination * twoch_pow

    return gnp.exp(-c * h) * polynomial


def maternp_covariance_ii_or_tt(x, p, param, pairwise=False):
    """Covariance between observations or predictands at x.

    The covariance matrix is computed using the Matérn kernel with half-integer regularity:

    .. math::

        K_{ij} = \\sigma^2  K(h_{ij}) + \\epsilon \\delta_{ij}

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
    nugget = 10.0 * sigma2 * gnp.finfo(gnp.float64).eps

    if pairwise:
        # return a vector of covariances
        K = sigma2 * gnp.ones((x.shape[0],))  # nx x 0
    else:
        # return a covariance matrix
        K = gnp.scaled_distance(loginvrho, x, x)  # nx x nx
        K = sigma2 * maternp_kernel(p, K) + nugget * gnp.eye(K.shape[0])

    return K


def maternp_covariance_it(x, y, p, param, pairwise=False):
    """Covariance between observations and prediction points.

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
    """Matérn covariance function with half-integer regularity nu = p + 1/2.

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


def anisotropic_parameters_initial_guess_zero_mean(model, xi, zi):
    """Anisotropic initialization strategy with zero mean.

    See anisotropic_parameters_initial_guess
    """
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi).reshape(-1, 1)
    n = xi_.shape[0]
    d = xi_.shape[1]

    delta = gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta
    covparam = gnp.concatenate((gnp.array([log(1.0)]), -gnp.log(rho)))
    sigma2_GLS = 1.0 / n * model.norm_k_sqrd_with_zero_mean(xi_, zi_, covparam)

    return gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))


def anisotropic_parameters_initial_guess_constant_mean(model, xi, zi):
    """Anisotropic initialization strategy with a parameterized constant mean.

    This function provides initial parameter guesses for an
    anisotropic Gaussian process with a parameterized constant mean.

    Parameters
    ----------
    model : object
        The Gaussian process model object.
    xi : array_like, shape (n, d)
        Input data points used for fitting the GP model, where `n` is
        the number of points and `d` is the dimensionality.
    zi : array_like, shape (n, )
        Output (response) values corresponding to the input data points xi.

    Returns
    -------
    mean_GLS : float
        The generalized least squares (GLS) estimator of the
        mean. Computed as:

        .. math::

            m_{GLS} = \frac{\mathbf{1}^T K^{-1} \mathbf{z}}{\mathbf{1}^T K^{-1} \mathbf{1}}

    concatenated parameters : array_like
        An array containing the initialized :math:`\sigma^2_{GLS}` and :math:`\rho` values.
        The estimator :math:`\sigma^2_{GLS}` is given by:

        .. math::

            \sigma^2_{GLS} = \frac{1}{n} \mathbf{z}^T K^{-1} \mathbf{z}
    """
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi).reshape((-1, 1))  # Ensure zi_ is a column vector
    n = xi_.shape[0]
    d = xi_.shape[1]

    delta = gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta

    covparam = gnp.concatenate((gnp.array([gnp.log(1.0)]), -gnp.log(rho)))
    zTKinvz, Kinv1, Kinvz = model.k_inverses(xi_, zi_, covparam)

    mean_GLS = gnp.sum(Kinvz) / gnp.sum(Kinv1)
    sigma2_GLS = (1.0 / n) * zTKinvz

    return mean_GLS.reshape(1), gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))


def anisotropic_parameters_initial_guess(model, xi, zi):
    """Anisotropic initialization strategy for parameters of a Gaussian process model.

    Given the observed data points and their values, this function
    computes an initial guess for the anisotropic parameters. The
    guess for :math:`\\sigma^2` is initialized using the Generalized
    Least Squares (GLS) estimate as described below.

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
        Initial guess for the anisotropic parameters, comprising the
        estimate for :math:`\\sigma^2_{GLS}` followed by the estimates
        for the anisotropic lengthscales.

    Notes
    -----
    The GLS estimate for :math:`\\sigma^2` is given by:

    .. math::

        \\sigma^2_{GLS} = \\frac{1}{n} \\mathbf{z}^T \\mathbf{K}^{-1} \\mathbf{z}

    Where:

    * :math:`n` is the number of data points.
    * :math:`\\mathbf{z}` is the vector of observed data.
    * :math:`\\mathbf{K}` is the covariance matrix associated with the data locations.

    Additionally, the function uses a relation (not from the
    reference) between :math:`\\rho` and the volume of a ball in
    dimension :math:`d` for initialization:

    .. math::

        V_d(R) = \\frac{\\pi^{d/2} R^d}{\\Gamma(d/2+1)}

    Where :math:`R` is defined as :math:`\\rho / 2`.

    .. [1] Basak, S., Petit, S., Bect, J., & Vazquez, E. (2021).
       Numerical issues in maximum likelihood parameter estimation for
       Gaussian process interpolation. arXiv:2101.09747.
    """

    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi).reshape(-1, 1)
    n = xi_.shape[0]
    d = xi_.shape[1]

    delta = gnp.max(xi_, axis=0) - gnp.min(xi_, axis=0)
    rho = gnp.exp(gnp.gammaln(d / 2 + 1) / d) / (gnp.pi**0.5) * delta

    covparam = gnp.concatenate((gnp.array([log(1.0)]), -gnp.log(rho)))
    sigma2_GLS = 1.0 / n * model.norm_k_sqrd(xi_, zi_, covparam)

    return gnp.concatenate((gnp.log(sigma2_GLS), -gnp.log(rho)))


def make_selection_criterion_with_gradient(
    model, selection_criterion, xi, zi, parameterized_mean=False, meanparam_len=1
):
    """Make selection criterion function with gradient.

    Parameters
    ----------
    model : object
        Instance of a Gaussian process model that needs parameter optimization.
    selection_criterion : function
        Selection criterion function. (See Notes.)
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    parameterized_mean : bool, optional
        Whether to use mean parameter in the selection criterion.
    meanparam_len : int, optional
        Length of the mean parameter, used only if parameterized_mean is True.

    Returns
    -------
    crit_jit : function
        Selection criterion function with gradient.
    dcrit : function
        Gradient of the selection criterion function.

    Notes
    -----
    The `criterion` function should follow one of the following two forms:

    - For models without a parameterized mean: `criterion(model,
      covparam, xi, zi)` where `covparam` are the covariance
      parameters.

    - For models with a parameterized mean: `criterion(model,
      meanparam, covparam, xi, zi)` where both `meanparam` and
      `covparam` are passed.

    The function will automatically handle the form of the criterion
    based on whether a mean parameter is used (controlled by the
    `parameterized_mean` flag in the criterion definition).

    """
    xi_ = gnp.asarray(xi)
    zi_ = gnp.asarray(zi)

    if parameterized_mean:
        # make a selection criterion with mean parameter
        def crit_(param):
            meanparam = param[:meanparam_len]
            covparam = param[meanparam_len:]
            l = selection_criterion(model, meanparam, covparam, xi_, zi_)
            return l

    else:
        # make a selection criterion without mean parameter
        def crit_(covparam):
            l = selection_criterion(model, covparam, xi_, zi_)
            return l

    crit_jit = gnp.jax.jit(crit_)
    dcrit = gnp.jax.jit(gnp.grad(crit_jit))

    return crit_jit, dcrit


def autoselect_parameters(
    p0,
    criterion,
    gradient,
    bounds=None,
    silent=True,
    info=False,
    method="SLSQP",
    method_options={},
):
    """Optimize parameters using a provided criterion and gradient function.

    This function automatically optimizes the parameters of a given model based on
    a specified criterion function and its gradient. Different optimization methods
    can be used based on the `method` argument.

    Parameters
    ----------
    p0 : ndarray
        Initial guess of the parameters.
    criterion : function
        Function that computes the selection criterion for a given parameter set.
        It should take an ndarray of parameters and return a scalar value.
    gradient : function
        Function that computes the gradient of the selection criterion with respect
        to the parameters. It should take an ndarray of parameters and return an ndarray
        of the same shape.
    bounds : sequence of tuple, optional
        A sequence of (min, max) pairs specifying bounds for each parameter.
        Use None to indicate no bounds. Default is None.
    silent : bool, optional
        If True, suppresses optimization output messages. Default is True.
    info : bool, optional
        If True, returns additional information about the optimization process.
        Default is False.
    method : str, optional
        Optimization method to use. Supported methods are 'L-BFGS-B' and 'SLSQP'.
        Default is 'SLSQP'.
    method_options : dict, optional, default {}
        User options for the optimization method.

    Returns
    -------
    best : ndarray
        Array of optimized parameters.
    r : OptimizeResult, optional
        A dictionary of optimization information (only if `info=True`). This includes
        details like the initial parameters, final parameters, selection criterion function,
        and total time taken for optimization.

    Notes
    -----
    The function uses the `minimize` method from `scipy.optimize` for optimization.
    Depending on the backend (`gnp._gpmp_backend_`), different preparations are made
    for the criterion and gradient functions to ensure compatibility.
    """
    # Track optimization start time
    tic = time.time()

    # Setup to record parameter and criterion history
    history_params = []
    history_criterion = []
    best_criterion = float("inf")
    best_params = None

    def record_history(p, criterion_value):
        nonlocal best_criterion, best_params
        history_params.append(p.copy())
        history_criterion.append(criterion_value)

        if criterion_value < best_criterion:
            best_criterion = criterion_value
            best_params = p.copy()

    # Determine which backend to use and configure the criterion and gradient functions
    if gnp._gpmp_backend_ == "jax":
        # scipy.optimize.minimize cannot use jax arrays
        def crit_asnumpy(p):
            J = criterion(p).item()
            record_history(p, J)
            return J

        gradient_asnumpy = lambda p: gnp.to_np(gradient(p))

    elif gnp._gpmp_backend_ == "torch":

        def crit_asnumpy(p):
            v = criterion(gnp.asarray(p))
            J = v.detach().item()
            record_history(p, J)
            return J

        def gradient_asnumpy(p):
            g = gradient(gnp.asarray(p))
            if g is None:
                return gnp.zeros(p.shape)
            else:
                return g

    elif gnp._gpmp_backend_ == "numpy":

        def crit_asnumpy(p):
            try:
                J = criterion(p)
            except Exception as e:
                J = np.Inf
            record_history(p, J)
            return J

        gradient_asnumpy = None

    # Set default optimization options and update with user-provided options
    options = {"disp": not silent}
    if method == "L-BFGS-B":
        options.update(
            {
                "maxcor": 20,
                "ftol": 1e-6,
                "gtol": 1e-5,
                "eps": 1e-8,
                "maxfun": 15000,
                "maxiter": 15000,
                "maxls": 40,
                "iprint": -1,
                "maxls": 40,
                "finite_diff_rel_step": None,
            }
        )
    elif method == "SLSQP":
        options.update(
            {"ftol": 1e-6, "eps": 1e-8, "maxiter": 15000, "finite_diff_rel_step": None}
        )
    else:
        raise ValueError("Optimization method not implemented.")

    options.update(method_options)

    # Perform the minimization
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

    # Ensure that the best values are returned
    if r.fun > best_criterion:
        r.x = best_params
        r.fun = best_criterion
        r.best_value_returned = False
    else:
        r.best_value_returned = True

    # Set additional information about the optimization process
    r.history_params = history_params
    r.history_criterion = history_criterion
    r.initial_params = p0
    r.final_params = r.x
    r.selection_criterion = criterion
    r.total_time = time.time() - tic

    if not silent:
        print("Optimization completed. Best found criterion:", best_criterion)
        print("Gradient")
        print("--------")
        if gradient_asnumpy is not None:
            print(gradient_asnumpy(r.x))
        else:
            print("gradient not available")
        print(".")

    if info:
        return r.x, r
    else:
        return r.x


def select_parameters_with_criterion(
    model,
    criterion,
    xi,
    zi,
    meanparam0=None,
    covparam0=None,
    parameterized_mean=False,
    meanparam_len=1,
    info=False,
    verbosity=0,
):
    """Optimize Gaussian process model parameters using a specified
    selection criterion.

    This function performs parameter optimization for a Gaussian
    process model using the provided selection criterion. It can use a
    provided initial guess for the covariance parameters or employ a
    default initialization strategy. Additional information about the
    optimization can be obtained by setting the `info` parameter to
    True.

    Parameters
    ----------
    model : object
        Instance of a Gaussian process model that needs parameter optimization.
    criterion : function
        The selection criterion function (e.g., ML, REML or REMAP). The function must follow one of two forms:

        - For models without a parameterized mean:
          `criterion(model, covparam, xi, zi)` where `covparam` are the covariance parameters.

        - For models with a parameterized mean:
          `criterion(model, meanparam, covparam, xi, zi)` where both `meanparam` and
          `covparam` are passed.
    xi : ndarray, shape (n, d)
        Locations of the observed data points in the input space.
    zi : ndarray, shape (n,)
        Observed values corresponding to the data points `xi`.
    meanparam0 : ndarray, shape (meanparam_len,), optional
        Initial guess for the mean parameters. Required if `parameterized_mean=True`. Default is None.
    covparam0 : ndarray, shape (covparam_dim,), optional
        Initial guess for the covariance parameters. If not provided, the function
        defaults to the `anisotropic_parameters_initial_guess` method for initialization.
        Default is None.
    parameterized_mean : bool, optional
        Whether the mean is parameterized and included in the selection criterion.
        Default is False.
    meanparam_len : int, optional
        Length of the mean parameter vector if `parameterized_mean` is True. Default is 1.
    info : bool, optional
        Controls the return of additional optimization information. If set to True, the
        function returns an info dictionary with details about the optimization process.
        Default is False.
    verbosity : int, optional, values in {0, 1, 2}
        Sets the verbosity level for the function.
        - 0: No output messages (default).
        - 1: Minimal output messages.
        - 2: Detailed output messages.

    Returns
    -------
    model : object
        The input Gaussian process model object, updated with the optimized parameters.
    info_ret : dict, optional
        A dictionary with additional details about the optimization process. It contains
        fields like initial parameters (`covparam0`), final optimized parameters
        (`covparam`), selection criterion used, and the total time taken for optimization.
        This dictionary is only returned if `info` is set to True.

    Notes
    -----
    The `criterion` function should follow one of the following two forms:

    - For models without a parameterized mean:
      `criterion(model, covparam, xi, zi)` where `covparam` are the covariance parameters.

    - For models with a parameterized mean:
      `criterion(model, meanparam, covparam, xi, zi)` where both `meanparam` and `covparam` are passed.
    """
    tic = time.time()

    if covparam0 is None:
        covparam0 = anisotropic_parameters_initial_guess(model, xi, zi)

    # If the model has a parameterized mean, we need an initial guess for the mean parameters
    if parameterized_mean:
        if meanparam0 is None:
            raise ValueError(
                "meanparam0 must be provided when parameterized_mean is True. Use anisotropic_parameters_initial_guess_constant_mean if needed."
            )
        param0 = gnp.concatenate([meanparam0, covparam0])
    else:
        param0 = covparam0

    # Create the criterion and its gradient using the passed criterion function
    criterion_func, criterion_grad = make_selection_criterion_with_gradient(
        model,
        criterion,
        xi,
        zi,
        parameterized_mean=parameterized_mean,
        meanparam_len=meanparam_len,
    )

    # Optimize parameters using the provided criterion
    silent = True
    if verbosity == 1:
        print("Parameter selection using custom criterion...")
    elif verbosity == 2:
        silent = False

    param_opt, info_ret = autoselect_parameters(
        param0, criterion_func, criterion_grad, silent=silent, info=True
    )

    if verbosity == 1:
        print("done.")

    # Split the optimized parameters into meanparam and covparam if the mean is parameterized
    if parameterized_mean:
        meanparam_opt = param_opt[:meanparam_len]
        covparam_opt = param_opt[meanparam_len:]
        model.meanparam = gnp.asarray(meanparam_opt)
    else:
        meanparam_opt = None
        covparam_opt = param_opt

    model.covparam = gnp.asarray(covparam_opt)

    # NB: info_ret is essentially a dict with attribute accessors
    if info:
        info_ret["meanparam0"] = meanparam0
        info_ret["covparam0"] = covparam0
        info_ret["meanparam"] = meanparam_opt
        info_ret["covparam"] = covparam_opt
        info_ret["selection_criterion"] = criterion_func
        info_ret["time"] = time.time() - tic
        return model, info_ret
    else:
        return model


def update_parameters_with_criterion(
    model, criterion, xi, zi, parameterized_mean=False, meanparam_len=1, info=False
):
    """Update model parameters using a specified selection criterion.

    Parameters
    ----------
    model : object
        Gaussian process model.
    criterion : function
        The selection criterion function (e.g., REML or REMAP).
    xi : ndarray, shape (n, d)
        Locations of the observed data points.
    zi : ndarray, shape (n,)
        Observed values at the data points.
    parameterized_mean : bool, optional
        Whether the mean is parameterized and included in the selection criterion.
        Default is False.
    meanparam_len : int, optional
        Length of the mean parameter vector if `parameterized_mean` is True. Default is 1.
    info : bool, optional
        If True, returns additional information. Default is False.

    Returns
    -------
    model : object
        Updated Gaussian process model object with optimized parameters.
    info_ret : object, optional
        Additional information about the optimization (if info=True).
    """

    return select_parameters_with_criterion(
        model,
        criterion,
        xi,
        zi,
        meanparam0=model.meanparam if parameterized_mean else None,
        covparam0=model.covparam,
        parameterized_mean=parameterized_mean,
        meanparam_len=meanparam_len,
        info=info,
    )


def negative_log_likelihood_zero_mean(model, covparam, xi, zi):
    """Wrapper to core.model.negative_log_likelihood_zero_mean."""
    return model.negative_log_likelihood_zero_mean(covparam, xi, zi)


def negative_log_likelihood(model, meanparam, covparam, xi, zi):
    """Wrapper to core.model.negative_log_likelihood."""
    return model.negative_log_likelihood(meanparam, covparam, xi, zi)


def negative_log_restricted_likelihood(model, covparam, xi, zi):
    """Wrapper to core.model.negative_log_restricted_likelihood."""
    return model.negative_log_restricted_likelihood(covparam, xi, zi)


def select_parameters_with_reml(model, xi, zi, covparam0=None, info=False, verbosity=0):
    """Optimize Gaussian process model parameters using Restricted Maximum Likelihood (REML).

    See select_parameters_with_criterion()
    """
    return select_parameters_with_criterion(
        model,
        negative_log_restricted_likelihood,
        xi,
        zi,
        covparam0=covparam0,
        info=info,
        verbosity=verbosity,
    )


def update_parameters_with_reml(model, xi, zi, info=False):
    """Update model parameters using Restricted Maximum Likelihood (REML).

    See update_parameters_with_criterion
    """
    return update_parameters_with_criterion(
        model, model.negative_log_restricted_likelihood, xi, zi, info=info
    )


def log_prior_jeffrey_variance(covparam, lbda=1.0):
    """Compute a log prior using Jeffrey's prior on the variance parameter.

    This function assumes a Jeffrey's prior on the variance parameter:

    .. math::
        \pi(\sigma^2) \propto \left(\frac{1}{\sigma^2}\right)^\lambda

    Parameters
    ----------
    covparam : ndarray
        Parameter array, where the first element is the
        log(variance) (i.e., log(sigma^2)), and the remaining elements
        are the log of the length-scale parameters (log(1/rho)).
    lbda : float, optional
        Scaling factor for the log prior. Default is 1.0.

    Returns
    -------
    log_p : float
        The logarithm of the prior probability.

    """
    log_sigma2 = covparam[0]

    # Jeffrey's prior: p(sigma^2) \propto (1 / sigma^2)^lbda => log p(sigma^2) = - lbda * log(sigma^2)
    log_prior_sigma2 = - lbda * log_sigma2

    return log_prior_sigma2


def neg_log_restricted_posterior_with_jeffreys_prior(model, covparam, xi, zi):
    """Compute the negative log restricted posterior using Jeffrey's
    prior on the variance parameter.

    This function combines the negative log restricted likelihood with
    a Jeffrey's prior on the variance parameter:

    .. math::
        \pi(\sigma^2) \propto \frac{1}{\sigma^2}

    Parameters
    ----------
    model : object
        Gaussian process model object.
    covparam : ndarray
        The covariance parameter array, where the first element is the
        log(variance) (i.e., log(sigma^2)), and the remaining elements
        are the log of the length-scale parameters (log(1/rho)).
    xi : ndarray
        The input data points.
    zi : ndarray
        The observed values at the input data points.

    Returns
    -------
    float
        The value of the negative log restricted posterior.

    """
    # Compute the negative log restricted likelihood
    nlrl = model.negative_log_restricted_likelihood(covparam, xi, zi)

    # Compute the log prior using Jeffrey's prior on the variance
    log_prior = log_prior_jeffrey_variance(covparam)

    # Posterior
    neg_log_posterior = nlrl - log_prior

    return neg_log_posterior


def select_parameters_with_remap(
    model, xi, zi, covparam0=None, info=False, verbosity=0
):
    """Optimize Gaussian process model parameters using Restricted Maximum A Posteriori (REMAP)."""
    return select_parameters_with_criterion(
        model,
        neg_log_restricted_posterior_with_jeffreys_prior,
        xi,
        zi,
        covparam0=covparam0,
        info=info,
        verbosity=verbosity,
    )


def update_parameters_with_remap(model, xi, zi, info=False):
    """Update model parameters using Restricted Maximum A Posteriori (REMAP)."""
    return update_parameters_with_criterion(
        model, neg_log_restricted_posterior_with_jeffreys_prior, xi, zi, info=info
    )
