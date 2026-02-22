# gpmp/num/shared.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""Backend-independent helpers for gpmp.num."""

from typing import Any, Callable, Union

from gpmp.config import get_config

Scalar = Union[int, float]
ArrayLike = Any


def get_dtype():
    return get_config().dtype_resolved


def compute_gammaln(up_to_p: int) -> ArrayLike:
    """
    Return gammaln(k) for k = 0, ..., 2*up_to_p + 1 as a 1D backend array.
    Grows and caches a single table in _config.caches["gammaln"]["table"].
    """
    import gpmp.num as gnp

    n = 2 * up_to_p + 2
    cache = get_config().caches.setdefault("gammaln", {})
    table = cache.get("table")

    if table is None:
        table = gnp.asarray(gnp.gammaln(gnp.arange(n)))
        cache["table"] = table
    elif table.shape[0] < n:
        old_n = table.shape[0]
        tail = gnp.asarray(gnp.gammaln(gnp.arange(old_n, n)))
        table = gnp.concatenate((table, tail))
        cache["table"] = table

    return table[:n]


def derivative_finite_diff(
    f: Callable[[Scalar], ArrayLike], x: Scalar, h: Scalar
) -> ArrayLike:
    """
    5-point central difference derivative of f w.r.t. scalar x.
    f(x) must return a NumPy (or similar) array/matrix/tensor.
    """
    f_x_p2 = f(x + 2 * h)
    f_x_p1 = f(x + h)
    f_x_m1 = f(x - h)
    f_x_m2 = f(x - 2 * h)
    return (-f_x_p2 + 8 * f_x_p1 - 8 * f_x_m1 + f_x_m2) / (12.0 * h)


def try_with_postmortem(
    func: Callable[..., ArrayLike], *args: Any, **kwargs: Any
) -> ArrayLike:
    """
    Execute ``func(*args, **kwargs)`` and drop into pdb post-mortem on failure.
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        extype, value, tb = __import__("sys").exc_info()
        __import__("traceback").print_exc()
        __import__("pdb").post_mortem(tb)
