# gpmp/modeldiagnosis/utils.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Utilities for GP model diagnosis.

Defines
-------
sigma_rho_from_covparam
    Extract sigma and rho values from a covariance parameter vector.
describe_array
    Build a DataFrame of basic descriptive statistics for an array.
pretty_print_dictionary
    Print a dictionary with aligned keys and formatted floats.

Compatibility
-------------
The legacy name `pretty_print_dictionnary` is kept as an alias.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

import gpmp.num as gnp
from gpmp.misc.dataframe import DataFrame, ftos


def sigma_rho_from_covparam(covparam: Any) -> Dict[str, Any]:
    """
    Extract sigma and rho parameters from a covariance parameter vector.

    Assumes the convention:
    - covparam[0] = log(sigma^2)
    - covparam[i] = log(1/rho_{i-1}) for i >= 1

    Parameters
    ----------
    covparam : array-like, shape (p,)
        Covariance parameters.

    Returns
    -------
    dict
        Dictionary with keys:
        - "sigma": sigma
        - "rho0", "rho1", ... : rho values
    """
    covparam = gnp.asarray(covparam).reshape(-1)

    out: Dict[str, Any] = {}
    out["sigma"] = gnp.exp(0.5 * covparam[0])

    n_rho = int(covparam.shape[0]) - 1
    for i in range(n_rho):
        out[f"rho{i:d}"] = gnp.exp(-covparam[i + 1])

    return out


def describe_array(x, rownames, sigma_factor=None):
    """
    Build simple descriptive statistics for an array.

    Parameters
    ----------
    x : array_like
        Input data. Shape (n,) or (n, d).
    rownames : list of str
        Row names for the output DataFrame. Length 1 if x is 1D, else d.
    sigma_factor : float or array_like, optional
        Scaling used for the last column. If scalar, the same factor is applied
        to all dimensions. If array_like, must have length d.

    Returns
    -------
    DataFrame
        Statistics per dimension.
    """
    x = np.asarray(x)
    dim = 1 if x.ndim == 1 else x.shape[1]

    if sigma_factor is None:
        colnames = ["min", "max", "delta", "mean", "std"]
        data = np.empty((dim, 5), dtype=float)
    else:
        colnames = ["min", "max", "delta", "mean", "std", "delta_over_sigma"]
        data = np.empty((dim, 6), dtype=float)

    xmin = np.min(x, axis=0)
    xmax = np.max(x, axis=0)
    xdelta = xmax - xmin
    xmean = np.mean(x, axis=0)
    xstd = np.std(x, axis=0)

    data[:, 0] = np.atleast_1d(xmin).astype(float, copy=False)
    data[:, 1] = np.atleast_1d(xmax).astype(float, copy=False)
    data[:, 2] = np.atleast_1d(xdelta).astype(float, copy=False)
    data[:, 3] = np.atleast_1d(xmean).astype(float, copy=False)
    data[:, 4] = np.atleast_1d(xstd).astype(float, copy=False)

    if sigma_factor is not None:
        sf = np.asarray(sigma_factor, dtype=float)
        if sf.ndim == 0:
            sf = np.full((dim,), float(sf))
        else:
            sf = sf.reshape(-1)
            if sf.size != dim:
                raise ValueError(
                    "sigma_factor must be a scalar or have length equal to the number of columns in x."
                )
        data[:, 5] = data[:, 2] * sf

    return DataFrame(data, colnames, rownames)


def pretty_print_dictionary(d: Dict[str, Any], fp: int = 4) -> None:
    """
    Print a dictionary with aligned keys.

    Parameters
    ----------
    d : dict
        Values can be scalars or backend arrays with one element.
    fp : int, optional
        Number of decimal places for float formatting.

    Returns
    -------
    None
    """
    if not d:
        return

    max_key_length = max(15, max(len(str(k)) for k in d.keys()) + 2)

    for k, v in d.items():
        if not gnp.isscalar(v):
            try:
                v = v.item()
            except Exception:
                pass

        if isinstance(v, float):
            fmt = f"{{:>{max_key_length}s}}: {{:s}}"
            print(fmt.format(str(k), ftos(v, fp)))
        else:
            print(f"{str(k):>{max_key_length}s}: {v}")


def pretty_print_dictionnary(d: Dict[str, Any], fp: int = 4) -> None:
    """
    Backward-compatible alias for pretty_print_dictionary.

    Parameters
    ----------
    d : dict
    fp : int, optional

    Returns
    -------
    None
    """
    pretty_print_dictionary(d, fp=fp)


__all__ = [
    "sigma_rho_from_covparam",
    "describe_array",
    "pretty_print_dictionary",
    "pretty_print_dictionnary",
]
