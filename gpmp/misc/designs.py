## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
import numpy as np
from scipy.stats import qmc
from ..kernel import distance


def maxdist(sample):
    '''maximum distance / diameter'''
    D = distance(sample, sample)
    maxdist = np.max(D)
    return maxdist


def mindist(sample):
    '''minimum / separation distance'''
    D = distance(sample, sample)
    mindist = np.min(D)
    return mindist


def discrepancy(sample):
    '''discrepancy'''
    return qmc.discrepancy(sample)


def filldist_approx(sample, box, n=int(1e6), x=None):
    '''fill distance approximated using a random uniform discretization
    of box

    '''
    dim = sample.shape[1]
    if x is None:
        x = randunif(dim, n, box)
    else:
        n = x.shape[0]
    filldist = 0
    for i in range(n):
        D = distance(sample, x)
        d = np.min(D)
        if d > filldist:
            filldist = d
    return filldist


def scale(sample_standard, box):
    '''map a standard sample in [0,1]^dim to box'''
    l_bounds, u_bounds = box[0], box[1]
    sample_box = qmc.scale(sample_standard, l_bounds, u_bounds)
    return sample_box


def regulargrid(dim, n, box):
    """Builds a regular grid

    Builds a regular grid in the DIM-dimensional hyperrectangle 
    \Prod [xmin_i; xmax_i]. 

    If n is an integer, a grid of size n^dim is built;

    If n is a list of length dim, a grid of size prod(n) is built,
     with n_i points on coordinate i.

    The dim-dimensional hyperrectangle is specified by the argument
    box, which is a 2 x dim array where box_(1, i) and box_(2, i) are
    the lower- and upper-bound of the interval on the i^th coordinate.

    Parameters
    ----------
    dim : _type_
        _description_
    n : _type_
        _description_
    box : _type_
        _description_

    Returns
    -------
    x : _type_
        regulargrid (dim, n, box)
    """

    # Read argument 'n'
    if not isinstance(n, list):
        n = [n for i in range(dim)]

    # Read argument 'box'
    xmin, xmax = box[0], box[1]

    # levels
    levels = [np.linspace(xmin[i], xmax[i], n[i]) for i in range(dim)]

    # Construct a full factorial design x
    Xv = np.meshgrid(*levels, copy=True, sparse=False, indexing='ij')
    Xv = np.array(Xv)

    N = np.prod(n)
    x = np.zeros((N, dim))
    for i in range(dim):
        x[:, i] = Xv[i].reshape(N, )

    return x


def randunif(dim, n, box):
    '''random uniform sample in box'''
    sample = np.random.rand(n, dim)
    sample = scale(sample, box)
    
    return sample

def ldrandunif(dim, n, box):
    """low discrepancy random uniform sample in box

    Parameters
    ----------
    dim : int
        _description_
    n : int
        _description_
    box : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    
    Notes
    -----
    FIXME: optimization method
    """
    max_iter = 100
    mindiscrepany = 1e6  # large number
    for i in range(max_iter):
        sample = np.random.rand(n, dim)
        d = discrepancy(sample)
        if d < mindiscrepany:
            mindiscrepany = d
            sample_ld = sample

    sample_ld = scale(sample_ld, box)

    return sample_ld


def maximinldlhs(dim, n, box):
    ''' maximin low-discrepancy Latin hypercube sampling '''
    sampler = qmc.LatinHypercube(d=dim, optimization='random-cd')

    max_iter = 20
    maximindist = 0
    for i in range(max_iter):
        sample = sampler.random(n)
        d = mindist(sample)
        if d > maximindist:
            maximindist = d
            sample_maximin = sample

    sample_maximin = scale(sample_maximin, box)

    return sample_maximin
