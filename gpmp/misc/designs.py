## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import cdist, pdist


def maxdist(sample):
    """
    Calculate the maximum distance (diameter) between any pair of points in the sample.

    Parameters
    ----------
    sample : numpy.ndarray
        Array of points in the sample.

    Returns
    -------
    float
        Maximum distance between any pair of points in the sample.
    """
    D = pdist(sample)
    maxdist = np.max(D)
    return maxdist


def mindist(sample):
    """
    Calculate the minimum distance (separation) between any pair of points in the sample.

    Parameters
    ----------
    sample : numpy.ndarray
        Array of points in the sample.

    Returns
    -------
    float
        Minimum distance between any pair of points in the sample.
    """
    D = pdist(sample)
    mindist = np.min(D)
    return mindist


def discrepancy(sample):
    """
    Calculate the discrepancy of the sample.

    Parameters
    ----------
    sample : numpy.ndarray
        Array of points in the sample.

    Returns
    -------
    float
        Discrepancy value of the sample.
    """
    return qmc.discrepancy(sample)


def filldist_approx(sample, box, n=int(1e6), x=None):
    """
    Approximate the fill distance using a random uniform discretization of the box.

    Parameters
    ----------
    sample : numpy.ndarray
        Array of points in the sample.
    box : list of lists
        List of lists containing the lower and upper bounds of the box.
    n : int, optional
        Number of points in the random uniform discretization, default is 1e6.
    x : numpy.ndarray, optional
        Points in the random uniform discretization, default is None.

    Returns
    -------
    float
        Approximated fill distance.
    """
    dim = sample.shape[1]
    if x is None:
        x = randunif(dim, n, box)
    else:
        n = x.shape[0]
    filldist = 0
    for i in range(n):
        D = cdist(sample, x)
        d = np.min(D)
        if d > filldist:
            filldist = d
    return filldist


def scale(sample_standard, box):
    """
    Map a standard sample in [0, 1]^dim to the given box.

    Parameters
    ----------
    sample_standard : numpy.ndarray
        Array of points in the standard sample.
    box : list of lists
        List of lists containing the lower and upper bounds of the box.

    Returns
    -------
    numpy.ndarray
        Sample points mapped to the given box.
    """
    l_bounds, u_bounds = box[0], box[1]
    sample_box = qmc.scale(sample_standard, l_bounds, u_bounds)
    return sample_box


def regulargrid(dim, n, box):
    """
    Build a regular grid in the dim-dimensional hyperrectangle.

    If n is an integer, a grid of size n^dim is built;

    If n is a list of length dim, a grid of size prod(n) is built,
    with n_i points on coordinate i.

    The dim-dimensional hyperrectangle is specified by the argument
    box, which is a 2 x dim array where box_(1, i) and box_(2, i) are
    the lower- and upper-bound of the interval on the i^th coordinate.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    n : int or list
        Number of points per dimension or a list with the number of points per dimension.
    box : list of lists
        List of lists containing the lower and upper bounds of the box.

    Returns
    -------
    x : numpy.ndarray
        Regular grid in the dim-dimensional hyperrectangle.
    """

    # Read argument 'n'
    if not isinstance(n, list):
        n = [n for i in range(dim)]

    # Read argument 'box'
    xmin, xmax = box[0], box[1]

    # levels
    levels = [np.linspace(xmin[i], xmax[i], n[i]) for i in range(dim)]

    # Construct a full factorial design x
    Xv = np.meshgrid(*levels, copy=True, sparse=False, indexing="ij")
    Xv = np.array(Xv)

    N = np.prod(n)
    x = np.zeros((N, dim))
    for i in range(dim):
        x[:, i] = Xv[i].reshape(
            N,
        )

    return x


def randunif(dim, n, box):
    """
    Generate a random uniform sample in the specified box.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    n : int
        Number of points in the sample.
    box : list of lists
        List of lists containing the lower and upper bounds of the box.

    Returns
    -------
    numpy.ndarray
        Random uniform sample in the specified box.
    """
    sample = np.random.rand(n, dim)
    sample = scale(sample, box)

    return sample


def ldrandunif(dim, n, box, max_iter=50):
    """
    Generate a low discrepancy random uniform sample in the specified box.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    n : int
        Number of points in the sample.
    box : list of lists
        List of lists containing the lower and upper bounds of the box.
    max_iter : int, optional
        Maximum number of iterations for optimization, default is 50.

    Returns
    -------
    numpy.ndarray
        Low discrepancy random uniform sample in the specified box.

    Notes
    -----
    FIXME: optimization method
    """
    mindiscrepany = 1e6  # large number
    for i in range(max_iter):
        sample = np.random.rand(n, dim)
        d = discrepancy(sample)
        if d < mindiscrepany:
            mindiscrepany = d
            sample_ld = sample

    sample_ld = scale(sample_ld, box)

    return sample_ld


def maximinlhs(dim, n, box, max_iter=1000):
    """
    Generate a maximin Latin Hypercube Sample (LHS) within the specified box.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    n : int
        Number of points in the sample.
    box : list of lists
        List of lists containing the lower and upper bounds of the box.
    max_iter : int, optional
        Maximum number of iterations for finding the sample with the maximum minimum distance, default is 1000.

    Returns
    -------
    numpy.ndarray
        Maximin Latin Hypercube Sample within the specified box.
    """
    sampler = qmc.LatinHypercube(d=dim, optimization=None)

    maximindist = 0
    for i in range(max_iter):
        sample = sampler.random(n)
        d = mindist(sample)
        if d > maximindist:
            maximindist = d
            sample_maximin = sample

    sample_maximin = scale(sample_maximin, box)

    return sample_maximin


def maximinldlhs(dim, n, box):
    """
    Generate a maximin low-discrepancy Latin Hypercube Sample (LHS) within the specified box.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    n : int
        Number of points in the sample.
    box : list of lists
        List of lists containing the lower and upper bounds of the box.

    Returns
    -------
    numpy.ndarray
        Maximin low-discrepancy Latin Hypercube Sample within the specified box.
    """
    sampler = qmc.LatinHypercube(d=dim, optimization="random-cd")

    max_iter = 10
    maximindist = 0
    for i in range(max_iter):
        sample = sampler.random(n)
        d = mindist(sample)
        if d > maximindist:
            maximindist = d
            sample_maximin = sample

    sample_maximin = scale(sample_maximin, box)

    return sample_maximin
