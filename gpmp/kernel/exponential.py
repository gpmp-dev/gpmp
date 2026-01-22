# gpmp/kernel/exponential.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import gpmp.num as gnp

def exponential_kernel(h):
    """Exponential kernel.

    .. math::
        k(h) = \\exp(-h)

    Parameters
    ----------
    h : gnp.array, shape (n,)
        Distances between points.

    Returns
    -------
    gnp.array, shape (n,)
        Kernel values.
    """
    return gnp.exp(-h)
