# gpmp/core/utils.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Small utilities used across `gpmp.core` modules.

This file hosts:
- Shape/type validation & conversion helpers for (xi, zi, xt)
- Model mean validation at construction time
- Return inf scalar / tensor helper
"""
from typing import Optional, Tuple
import gpmp.num as gnp


def ensure_shapes_and_type(
    *,
    xi=None,
    zi=None,
    xt=None,
    convert: bool = True,
):
    """Validate and adjust shapes/types of input arrays.

    Parameters
    ----------
    xi : array_like, optional
        Observation points (n, d).
    zi : array_like, optional
        Observed values (n,) or (n, 1).
    xt : array_like, optional
        Prediction points (m, d).
    convert : bool, optional
        Convert arrays to backend type (default True).

    Returns
    -------
    tuple
        (xi, zi, xt) with proper shapes and types.

    Notes
    -----
    - If `zi` is provided as a 2D column (n,1), it is reshaped to (n,).
    - If `convert` is True, inputs are converted via `gnp.asarray`.
    - Basic dimensionality checks are enforced:
        * xi is 2D
        * xt is 2D
        * zi is 1D or a single-column 2D
        * xi.shape[0] == zi.shape[0] (when both given)
        * xi.shape[1] == xt.shape[1] (when both given)
    """
    if xi is not None:
        assert len(xi.shape) == 2, "xi should be a 2D array"

    if zi is not None:
        if len(zi.shape) == 2:
            assert zi.shape[1] == 1, "zi should only have one column if it's a 2D array"
            zi = zi.reshape(-1)  # (n,1) -> (n,)
        else:
            assert len(zi.shape) == 1, "zi should be 1D or a 2D column array"

    if xt is not None:
        assert len(xt.shape) == 2, "xt should be a 2D array"

    if xi is not None and zi is not None:
        assert xi.shape[0] == zi.shape[0], "xi and zi must have the same number of rows"
    if xi is not None and xt is not None:
        assert xi.shape[1] == xt.shape[1], "xi and xt must have the same number of columns"

    if convert:
        if xi is not None:
            xi = gnp.asarray(xi)
        if zi is not None:
            zi = gnp.asarray(zi)
        if xt is not None:
            xt = gnp.asarray(xt)

    return xi, zi, xt


def validate_model_mean(meantype: str, mean, meanparam):
    """Validate model initialization inputs for the mean component.

    Parameters
    ----------
    meantype : {'zero','parameterized','linear_predictor'}
        Type of mean to be used by the model.
    mean : callable or None
        Mean function (ignored if meantype == 'zero').
    meanparam : array_like or None
        Parameters of the mean function (required for 'parameterized').

    Raises
    ------
    ValueError
        If `meantype` is invalid or inconsistent with `mean`/`meanparam`.
    TypeError
        If a callable `mean` is required but not provided.

    Notes
    -----
    - For 'zero', `mean` must be None.
    - For 'parameterized' and 'linear_predictor', `mean` must be callable.
    """
    if meantype not in {"zero", "parameterized", "linear_predictor"}:
        raise ValueError("meantype must be one of 'zero', 'parameterized', or 'linear_predictor'")

    if meantype == "zero" and mean is not None:
        raise ValueError("For meantype 'zero', mean must be None")

    # parameterized or linear_predictor
    if meantype in ["parameterized", "linear_predictor"] and not callable(mean):
        raise TypeError(
            "For meantype 'parameterized' or 'linear_predictor', mean must be a callable function"
        )


def return_inf():
    """Backend-specific +∞ scalar/tensor."""
    if gnp._gpmp_backend_ == "jax" or gnp._gpmp_backend_ == "numpy":
        return gnp.inf
    elif gnp._gpmp_backend_ == "torch":
        # Use LinAlgError instead of raising RuntimeError for linalg operations
        # https://github.com/pytorch/pytorch/issues/64785
        # https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
        # extype, value, tb = __import__("sys").exc_info()
        # __import__("traceback").print_exc()
        # __import__("pdb").post_mortem(tb)
        inf_tensor = gnp.tensor(float("inf"), requires_grad=True)
        return inf_tensor  # returns inf with None gradient
