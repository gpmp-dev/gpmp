#coding: utf-8
## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
import math
import numpy as np


def twobumps(x):
    '''
    Computes the response Z of the TwoBumps function at X.

    The TwoBumps function is defined as:

       TwoBumps(x) = - (0.7x + sin(5x + 1) + 0.1 sin(10x))

    for x in [-1.0; 1.0].
    '''
    z = -(0.7 * x + (np.sin(5 * x + 1)) + 0.1 * (np.sin(10 * x)))
    return z.reshape([-1])


def wave(x):
    """Computes the Wave function
    
    .. math ::

        f: x & = (x1, x2) \in [-1; 1] x [-1; 1] |-> \\\\
        & exp(1.8 * (x1 + x2)) + 3 * x1 + 6 * x2.^2 + 3 * sin(4 * pi * x1)

    Parameters
    ----------
    x : numpay.array
        _description_

    Returns
    -------
    numpay.array
        _description_
    """
    z = np.exp(1.8 * (x[:, 0] + x[:, 1])) \
        + 3 * x[:, 1] \
        + 6 * x[:, 1]**2 \
        + 3 * np.sin(4 * np.pi * x[:, 0])

    return z


def braninhoo(x):
    """the Branin-Hoo function

    The Branin-Hoo function [1] is a classical test
    function for global optimization algorithms, which belongs to the
    well-known Dixon-Szego test set [2]. It is usually
    minimized over [-5; 10] x [0; 15].

    Parameters
    ----------
    x : numpy.array
        _description_

    Returns
    -------
    numpy.array
        _description_
    
    References
    ----------
    [1] Branin, F. H. and Hoo, S. K. (1972), A Method for Finding Multiple
        Extrema of a Function of n Variables, in Numerical methods of
        Nonlinear Optimization (F. A. Lootsma, editor, Academic Press,
        London), 231-237.

    [2] Dixon L.C.W., Szego G.P., Towards Global Optimization 2, North-
        Holland, Amsterdam, The Netherlands (1978)

    [3] Surjanovic, S. and Bingham D. (2013), Branin Function,
        https://www.sfu.ca/~ssurjano/Code/braninm.html
    """

    a = 5.1 / (4 * math.pi**2)
    b = 5 / math.pi
    c = 10 * (1 - 1 / (8 * math.pi))

    z = (x[:, 1] - a * x[:, 0]**2 + b * x[:, 0] - 6)**2 \
        + c * np.cos(x[:, 0]) + 10

    return z


def hartmann4(x):
    """Hartmann 4-dimensional function [1, 2]
    The 4-dimensional Hartmann is a multimodal function

    Input Domain:
    xi ∈ (0, 1), for all i = 1, …, 6.

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    References
    ----------
    [1] Dixon, L. C. W., & Szego, G. P. (1978). The global
        optimization problem: an introduction. Towards global
        optimization, 2, 1-15.

    [2] Picheny, V., Wagner, T., & Ginsbourger, D. (2012). A benchmark
        of kriging-based infill criteria for noisy optimization.
        Based on https://www.sfu.ca/~ssurjano/hart6.html

    Notes
    -----
    Authors: Sonja Surjanovic and Derek Bingham, Simon Fraser University

    Original copyright notice:
   
    Copyright 2013. Derek Bingham, Simon Fraser University.
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124,  8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091,  381]])

    outer = 0
    for ii in range(4):
        inner = 0
        for jj in range(4):
            xj = x[:, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij * (xj - Pij)**2
        new = alpha[ii] * np.exp(-inner)
        outer = outer + new

    z = (1.1 - outer) / 0.839

    return z


def hartmann6(x):
    """Hartmann 6-dimensional function [1]

    The 6-dimensional Hartmann function has 6 local minima and a 
    a global minimum f(x*) = -3.32237

    Input Domain:
    xi ∈ (0, 1), for all i = 1, …, 6.

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    References
    ----------
    [1] Dixon, L. C. W., & Szego, G. P. (1978). The global
        optimization problem: an introduction. Towards global
        optimization, 2, 1-15.

    [2] Picheny, V., Wagner, T., & Ginsbourger, D. (2012). A benchmark
        of kriging-based infill criteria for noisy optimization.
        Based on https://www.sfu.ca/~ssurjano/hart6.html

    Notes
    -----
    Authors: Sonja Surjanovic and Derek Bingham, Simon Fraser University

    Original copyright notice:
   
    Copyright 2013. Derek Bingham, Simon Fraser University.
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124,  8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])

    outer = 0
    for ii in range(4):
        inner = 0.0
        for jj in range(6):
            xj = x[:, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij * (xj - Pij)**2
        new = alpha[ii] * np.exp(-inner)
        outer = outer + new

    z = - outer

    return z


def borehole(x):
    """Borehole function

    Dimensions: 8

    The Borehole function [1] models water flow through a borehole. Its
    simplicity and quick evaluation makes it a commonly used function
    for testing a wide variety of methods in computer experiments.
    
    The response is water flow rate, in m3/yr.

    Input Domain and Distributions:

    The input variables and their usual input ranges are:

    rw ∈ [0.05, 0.15] 	radius of borehole (m)
    r ∈ [100, 50000] 	radius of influence (m)
    Tu ∈ [63070, 115600]    	transmissivity of upper aquifer (m2/yr)
    Hu ∈ [990, 1110] 	potentiometric head of upper aquifer (m)
    Tl ∈ [63.1, 116] 	transmissivity of lower aquifer (m2/yr)
    Hl ∈ [700, 820] 	potentiometric head of lower aquifer (m)
    L ∈ [1120, 1680] 	length of borehole (m)
    Kw ∈ [9855, 12045] 	hydraulic conductivity of borehole (m/yr)

    For the purposes of uncertainty quantification, the distributions of the input random variables are:

    rw ~ N(μ=0.10, σ=0.0161812)
    r ~ Lognormal(μ=7.71, σ=1.0056)   
    Tu ~ Uniform[63070, 115600]
    Hu ~ Uniform[990, 1110]
    Tl ~ Uniform[63.1, 116]
    Hl ~ Uniform[700, 820]
    L ~ Uniform[1120, 1680]
    Kw ~ Uniform[9855, 12045]

    Above, N(μ, σ) is the Normal distribution with mean μ and variance
    σ2. Lognormal(μ, σ) is the Lognormal distribution of a variable,
    such that the natural logarithm of the variable has a N(μ, σ)
    distribution.

    Parameters
    ----------
    x : _type_
        [rw, r, Tu, Hu, Tl, Hl, L, Kw]

    Returns
    -------
    z : _type_
        water flow rate

    References
    ----------
    [1] Harper, W. V., & Gupta, S. K. (1983). Sensitivity/uncertainty
        analysis of a borehole scenario comparing Latin Hypercube
        Sampling and deterministic sensitivity approaches
        (No. BMI/ONWI-516). Battelle Memorial Inst., Columbus, OH
        (USA). Office of Nuclear Waste Isolation

    Based on https://www.sfu.ca/~ssurjano/borehole.html
    
    Notes
    -----
    Authors: Sonja Surjanovic and Derek Bingham, Simon Fraser University

    Original copyright notice:
   
    Copyright 2013. Derek Bingham, Simon Fraser University.

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
    Dimensions: 8

    This function is used for the comparison of computer experiment
    designs. It is highly curved in some variables and less in others
    [1].

    Input Domain:

    The function is evaluated on the hypercube xi ∈ [0, 1], for all i = 1, …, 8.

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    References
    ----------
    [1] Dette, H., & Pepelyshev, A. (2010). Generalized Latin
        hypercube design for computer experiments. Technometrics,
        52(4).

    Based on https://www.sfu.ca/~ssurjano/detpep108d.html

    Notes
    -----
    Authors: Sonja Surjanovic and Derek Bingham, Simon Fraser University

    Original copyright notice:
   
    Copyright 2013. Derek Bingham, Simon Fraser University.
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    term1 = 4 * (x1 - 2 + 8 * x2 - 8 * x2**2)**2
    term2 = (3 - 4 * x2)**2
    term3 = 16 * np.sqrt(x3 + 1) * (2 * x3 - 1)**2

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
