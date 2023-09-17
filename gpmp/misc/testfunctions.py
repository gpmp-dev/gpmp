# coding: utf-8
## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
import math
import numpy as np


def twobumps(x):
    """
    Computes the response Z of the TwoBumps function at X.

    The TwoBumps function is defined as:

       TwoBumps(x) = - (0.7x + sin(5x + 1) + 0.1 sin(10x))

    Parameters
    ----------
    x : numpy.ndarray
        Input array of shape (n,)

    Returns
    -------
    numpy.ndarray
        Output array of shape (n,)
    """
    z = -(0.7 * x + (np.sin(5 * x + 1)) + 0.1 * (np.sin(10 * x)))
    return z.reshape([-1])


def wave(x):
    """Computes the Wave function.
    
    The Wave function is a popular test function used in optimization problems. It has multiple local minima and a single global minimum. It is defined as follows:
    
    .. math::
        f(x) = e^{1.8(x_1 + x_2)} + 3x_1 + 6x_2^2 + 3\sin(4\pi x_1)
        
    where x is a 2-dimensional numpy array in the range [-1, 1] x [-1, 1].
    
    Parameters
    ----------
    x : numpy.ndarray
        A 2-dimensional numpy array containing the input values of the function. The shape of the array should be (n, 2), where n is the number of input points.
        
    Returns
    -------
    numpy.ndarray
        A 1-dimensional numpy array containing the output values of the function. The shape of the array is (n,).
    """    
    z = (
        np.exp(1.8 * (x[:, 0] + x[:, 1]))
        + 3 * x[:, 1]
        + 6 * x[:, 1] ** 2
        + 3 * np.sin(4 * np.pi * x[:, 0])
    )

    return z


def braninhoo(x):
    """
    The Branin-Hoo function is a classical test function for global optimization algorithms, which belongs to the well-known Dixon-Szego test set. It is usually minimized over [-5; 10] x [0; 15].

    Parameters
    ----------
    x : numpy.array
        2D array of shape (n, 2) where each row represents a point in the 2D space to evaluate the function.

    Returns
    -------
    numpy.array
        A 1D array of shape (n,) containing the Branin-Hoo function values evaluated at the input points.

    Notes
    -----
    .. [1] Branin, F. H. and Hoo, S. K. (1972), A Method for Finding Multiple
        Extrema of a Function of n Variables, in Numerical methods of
        Nonlinear Optimization (F. A. Lootsma, editor, Academic Press,
        London), 231-237.

    .. [2] Dixon L.C.W., Szego G.P., Towards Global Optimization 2, North-
        Holland, Amsterdam, The Netherlands (1978)

    .. [3] Surjanovic, S. and Bingham D. (2013), Branin Function,
        https://www.sfu.ca/~ssurjano/Code/braninm.html
    """
    a = 5.1 / (4 * math.pi**2)
    b = 5 / math.pi
    c = 10 * (1 - 1 / (8 * math.pi))

    z = (x[:, 1] - a * x[:, 0] ** 2 + b * x[:, 0] - 6) ** 2 + c * np.cos(x[:, 0]) + 10

    return z


def hartmann4(x):
    """Hartmann 4-dimensional function [1, 2]

    The 4-dimensional Hartmann function is a multimodal function defined
    on the unit hypercube (i.e., xi in (0, 1), for all i = 1, ..., 4). It is
    commonly used as a test problem in global optimization.

    Parameters
    ----------
    x : numpy.ndarray
        An array of shape (n_samples, 4) containing the input points.

    Returns
    -------
    numpy.ndarray
        An array of shape (n_samples,) containing the function values at
        the input points.

    Notes
    -----
    The Hartmann 4-dimensional function has 6 local minima and one
    global minimum. The global minimum is located at x* = [0.20169,
    0.15001, 0.47687, 0.27533] and has a function value of f(x*) = -3.86278.
    The function is generally considered to be difficult to optimize.

    .. [1] Dixon, L. C. W., & Szego, G. P. (1978). The global optimization
        problem: an introduction. Towards global optimization, 2, 1-15.

    .. [2] Picheny, V., Wagner, T., & Ginsbourger, D. (2012). A benchmark
        of kriging-based infill criteria for noisy optimization.
        Based on https://www.sfu.ca/\~ssurjano/hart6.html

    Authors: Sonja Surjanovic and Derek Bingham, Simon Fraser University

    Original copyright notice: Copyright 2013. Derek Bingham, Simon Fraser University.
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    outer = 0
    for ii in range(4):
        inner = 0
        for jj in range(4):
            xj = x[:, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij * (xj - Pij) ** 2
        new = alpha[ii] * np.exp(-inner)
        outer = outer + new

    z = (1.1 - outer) / 0.839

    return z


def hartmann6(x):
    """Hartmann 6-dimensional function [1]

    The 6-dimensional Hartmann function has 6 local minima and a
    a global minimum f(x*) = -3.32237.

    .. math::

        f(x) = - \\sum_{i=1}^4 \\alpha_i \\exp \\bigl(-\\sum_{j=1}^6 A_{ij}(x_j - P_{ij})^2 \\bigr) 

    where :math:`x_i \in (0, 1)` for all :math:`i = 1, \ldots, 6`.

    Parameters
    ----------
    x : numpy.ndarray
        An array of shape (n, 6) containing the input values.

    Returns
    -------
    numpy.ndarray
        An array of shape (n,) containing the function values.

    Notes
    -----
    
    .. [1] Dixon, L. C. W., & Szego, G. P. (1978). The global
        optimization problem: an introduction. Towards global
        optimization, 2, 1-15.

    .. [2] Picheny, V., Wagner, T., & Ginsbourger, D. (2012). A benchmark
        of kriging-based infill criteria for noisy optimization.
        Based on https://www.sfu.ca/~ssurjano/hart6.html

    Authors: Sonja Surjanovic and Derek Bingham, Simon Fraser University

    Original copyright notice: Copyright 2013. Derek Bingham, Simon Fraser University.
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    outer = 0
    for ii in range(4):
        inner = 0.0
        for jj in range(6):
            xj = x[:, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij * (xj - Pij) ** 2
        new = alpha[ii] * np.exp(-inner)
        outer = outer + new

    z = -outer

    return z


def borehole(x):
    """Compute the water flow rate through a borehole.

    The Borehole function [1] models water flow through a borehole. Its simplicity
    and quick evaluation makes it a commonly used function for testing a wide variety
    of methods in computer experiments.

    Parameters
    ----------
    x : numpy.ndarray of shape (n_samples, 8)
        The input variables and their usual input ranges:
    
        * rw : radius of borehole in meters, range [0.05, 0.15]
        * r : radius of influence in meters, range [100, 50000]
        * Tu : transmissivity of upper aquifer in m^2/yr, range [63070, 115600]
        * Hu : potentiometric head of upper aquifer in meters, range [990, 1110]
        * Tl : transmissivity of lower aquifer in m^2/yr, range [63.1, 116]
        * Hl : potentiometric head of lower aquifer in meters, range [700, 820]
        * L : length of borehole in meters, range [1120, 1680]
        * Kw : hydraulic conductivity of borehole in m/yr, range [9855, 12045]

    Returns
    -------
    numpy.ndarray of shape (n_samples,)
        The water flow rate in m^3/yr.

    Notes
    -----
    .. [1] Harper, W. V., & Gupta, S. K. (1983). Sensitivity/uncertainty analysis of
        a borehole scenario comparing Latin Hypercube Sampling and deterministic
        sensitivity approaches (No. BMI/ONWI-516). Battelle Memorial Inst., Columbus,
        OH (USA). Office of Nuclear Waste Isolation.

    The distributions of the input random variables are:
    
    * rw ~ N(0.10, 0.0161812)
    * r ~ Lognormal(7.71, 1.0056)
    * Tu ~ Uniform[63070, 115600]
    * Hu ~ Uniform[990, 1110]
    * Tl ~ Uniform[63.1, 116]
    * Hl ~ Uniform[700, 820]
    * L ~ Uniform[1120, 1680]
    * Kw ~ Uniform[9855, 12045]

    Above, N(µ, s^2) is the Normal distribution with mean µ and variance s^2.
    Lognormal(µ, s) is the Lognormal distribution of a variable, such that the natural
    logarithm of the variable has a N(µ, s^2) distribution.

    Authors: Sonja Surjanovic and Derek Bingham, Simon Fraser University

    Original copyright notice: Copyright 2013. Derek Bingham, Simon Fraser University.
    """
    rw = x[:, 0]
    r = x[:, 1]
    Tu = x[:, 2]
    Hu = x[:, 3]
    Tl = x[:, 4]
    Hl = x[:, 5]
    L = x[:, 6]
    Kw = x[:, 7]

    frac1 = 2 * np.pi * Tu * (Hu - Hl)

    frac2a = 2 * L * Tu / (np.log(r / rw) * rw**2 * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r / rw) * (1 + frac2a + frac2b)

    z = frac1 / frac2

    return z


def detpep8d(x):
    """Dette & Pepelyshev (2010) 8-Dimensional Function

    This function is used for the comparison of computer experiment
    designs. It is highly curved in some variables and less in others
    [1].

    Input Domain:

    The function is evaluated on the hypercube :math:`x_i \in [0, 1], i = 1, \ldots, 8`.

    Parameters
    ----------
    x : numpy.ndarray
        2D array of shape (n, 8) containing n samples with 8 variables each

    Returns
    -------
    numpy.ndarray
        1D array of shape (n,) containing the function values for each input sample

    Notes
    -----
    .. [1] Dette, H., & Pepelyshev, A. (2010). Generalized Latin
        hypercube design for computer experiments. Technometrics,
        52(4).

    Based on https://www.sfu.ca/~ssurjano/detpep108d.html

    Authors: Sonja Surjanovic and Derek Bingham, Simon Fraser University

    Original copyright notice: Copyright 2013. Derek Bingham, Simon Fraser University.
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    term1 = 4 * (x1 - 2 + 8 * x2 - 8 * x2**2) ** 2
    term2 = (3 - 4 * x2) ** 2
    term3 = 16 * np.sqrt(x3 + 1) * (2 * x3 - 1) ** 2

    outer = 0
    for ii in range(4, 9):
        inner = 0.0
        for jj in range(3, ii + 1):
            xj = x[:, jj - 1]
            inner = inner + xj

    new = ii * np.log(1 + inner)
    outer = outer + new

    z = term1 + term2 + term3 + outer

    return z
