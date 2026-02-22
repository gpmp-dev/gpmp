# gpmp/num/numpy_backend.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""NumPy numerical backend for GPmp.

This module defines the NumPy implementation of the gpmp.num API.
"""

import os
import warnings
import builtins
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from gpmp.config import _normalize_dtype_spec, get_config, init_backend, get_logger
from .shared import derivative_finite_diff

Scalar = Union[int, float]
ArrayLike = Any
CriterionCallable = Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
LoaderLike = Iterable[Tuple[ArrayLike, ArrayLike]]

_gpmp_backend_: str = init_backend()
_config = get_config()
_logger = get_logger()
_logger.info("Using backend: %s", _gpmp_backend_)
_DTYPE_SPEC = _normalize_dtype_spec(_config.dtype)

_LINALG_ERROR_KEYWORDS = (
    "singular",
    "not positive definite",
    "not positive-definite",
    "cholesky",
    "decomposition",
    "factorization",
    "matrix is not invertible",
    "matrix inversion",
    "inverse",
    "svd did not converge",
    "ill-conditioned",
    "linalg",
    "lapack",
    "cusolver",
    "array must not contain infs or nans",
)


# -----------------------------------------------------
#
#                      NUMPY
#
# -----------------------------------------------------

import numpy
from numpy import array, empty
from numpy.typing import NDArray

_np_dtype = numpy.float64
_config.dtype_resolved = _np_dtype

ndarray = NDArray[numpy.floating]
from numpy import (
    copy,
    array_equal,
    reshape,
    where,
    any,
    isscalar,
    isnan,
    isinf,
    isfinite,
    isclose,
    allclose,
    unique,
    nan_to_num,
    hstack,
    vstack,
    stack,
    tile,
    concatenate,
    split,
    expand_dims,
    empty,
    empty_like,
    zeros_like,
    zeros,
    ones,
    full,
    full_like,
    eye,
    diag,
    arange,
    linspace,
    logspace,
    meshgrid,
    floor,
    ceil,
    abs,
    sqrt,
    exp,
    log,
    log10,
    log1p,
    sin,
    cos,
    tan,
    tanh,
    diff,
    sum,
    cumsum,
    prod,
    mean,
    std,
    var,
    cov,
    percentile,
    sort,
    min,
    max,
    argmin,
    argmax,
    minimum,
    maximum,
    clip,
    einsum,
    matmul,
    trace,
    inner,
    all,
    logical_not,
    logical_and,
    logical_or,
)
from numpy.linalg import norm, cond, cholesky, qr, svd, inv
from numpy import pi, inf
from numpy import finfo, float64
from scipy.special import gammaln
from scipy.linalg import solve, solve_triangular, cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import norm as normal
from scipy.stats import multivariate_normal as scipy_mvnormal

# ..................................................

eps = finfo(_np_dtype).eps
fmax = numpy.finfo(_np_dtype).max

def safe_inf():
    return inf

def safe_neginf():
    return -inf

# ..................................................

def _is_linalg_exception(exc: Exception) -> bool:
    if isinstance(exc, numpy.linalg.LinAlgError):
        return True
    msg = str(exc).lower()
    return builtins.any(keyword in msg for keyword in _LINALG_ERROR_KEYWORDS)

# ..................................................

def array(x, dtype=None):
    if dtype is not None:
        return numpy.array(x, dtype=dtype)
    out = numpy.array(x)
    if numpy.issubdtype(out.dtype, numpy.floating):
        return out.astype(_np_dtype, copy=False)
    return out

def asarray(x, dtype=None):
    if dtype is not None:
        return numpy.asarray(x, dtype=dtype)
    if isinstance(x, numpy.ndarray):
        if numpy.issubdtype(x.dtype, numpy.floating):
            return x.astype(_np_dtype, copy=False)
        return x
    elif isinstance(x, (int, float)):
        dt = _np_dtype if isinstance(x, float) else None
        return numpy.array([x], dtype=dt)
    else:
        out = numpy.asarray(x)
        if numpy.issubdtype(out.dtype, numpy.floating):
            return out.astype(_np_dtype, copy=False)
        return out

def empty(shape, dtype=None):
    return numpy.empty(shape, dtype=_np_dtype if dtype is None else dtype)

def zeros(shape, dtype=None):
    return numpy.zeros(shape, dtype=_np_dtype if dtype is None else dtype)

def ones(shape, dtype=None):
    return numpy.ones(shape, dtype=_np_dtype if dtype is None else dtype)

def full(shape, fill_value, dtype=None):
    return numpy.full(
        shape, fill_value, dtype=_np_dtype if dtype is None else dtype
    )

def eye(n, m=None, k=0, dtype=None):
    return numpy.eye(n, M=m, k=k, dtype=_np_dtype if dtype is None else dtype)

def transpose(x, dim0, dim1):
    """Torch-style transpose: swap two dimensions."""
    return numpy.swapaxes(x, dim0, dim1)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    return numpy.linspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=_np_dtype if dtype is None else dtype,
        axis=axis,
    )

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    return numpy.logspace(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=_np_dtype if dtype is None else dtype,
        axis=axis,
    )

def asdouble(x):
    return numpy.asarray(x).astype(float64, copy=False)

def asint(x):
    return numpy.asarray(x).astype(int, copy=False)

def to_np(x):
    return x

def to_scalar(x):
    return x.item()

def isarray(x):
    return isinstance(x, numpy.ndarray)

def inftobigf(a, bigf=fmax / 1000.0):
    a = where(numpy.isinf(a), numpy.full_like(a, bigf), a)
    return a

# ..................................................

def grad(f: Callable[[ArrayLike], ArrayLike]) -> Callable[[ArrayLike], ArrayLike]:
    """
    Return function that computes gradient of scalar f via finite differences.

    Uses 5-point central difference formula for accuracy.
    Suitable for low to moderate dimensional problems.

    Parameters
    ----------
    f : callable
        Scalar-valued function taking an array and returning a scalar.

    Returns
    -------
    callable
        Function grad_f(x) that computes nabla f(x) using finite differences.
    """

    def grad_f(x: ArrayLike) -> ArrayLike:
        x_arr = asarray(x)
        grad_vec = zeros_like(x_arr)
        h = 1e-5  # step size for finite differences

        for i in range(x_arr.shape[0]):

            def f_i(xi_scalar):
                x_copy = copy(x_arr)
                x_copy[i] = xi_scalar
                return f(x_copy)

            # derivative_finite_diff expects scalar input
            grad_vec[i] = derivative_finite_diff(f_i, float(x_arr[i]), h)

        return grad_vec

    return grad_f

def value_and_grad(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    *,
    h: _np_dtype = 1e-5,
) -> Tuple[ArrayLike, ArrayLike]:
    """Returns (y, grad_y) where y = f(x) is scalar.  Uses
    derivative_finite_diff on each coordinate (expects scalar
    input).

    """

    def _coerce_scalar_like(y_):
        if isscalar(y_):
            return y_
        if isarray(y_):
            if y_.ndim == 0:
                return y_
            if y_.size == 1:
                return reshape(y_, ())
        raise ValueError("f(x) must return a scalar.")

    y = _coerce_scalar_like(f(x))
    grad = zeros_like(x, dtype=_np_dtype)
    x_tmp = x.copy()
    for idx in range(x.shape[0]):
        xi = x[idx]

        def f_i(xi_scalar: _np_dtype):
            x_tmp[idx] = xi_scalar
            return _coerce_scalar_like(f(x_tmp))

        grad[idx] = derivative_finite_diff(f_i, xi, h)
        x_tmp[idx] = x[idx]  # restore
    return y, grad

class DifferentiableSelectionCriterion:
    def __init__(self, crit: CriterionCallable, x: ArrayLike, z: ArrayLike):
        self.crit = crit
        self.x, self.z = x, z
        self.gradient = None

    def __call__(self, p: ArrayLike) -> ArrayLike:
        return self.evaluate(p)

    def evaluate(self, p: ArrayLike) -> _np_dtype:
        return self.crit(p, self.x, self.z)

    def evaluate_no_grad(self, p: ArrayLike) -> _np_dtype:
        return self.evaluate(p)

    def evaluate_pre_grad(self, p: ArrayLike) -> _np_dtype:
        try:
            return self.crit(p, self.x, self.z)
        except Exception as exc:
            if _is_linalg_exception(exc):
                return inf
            raise


class BatchDifferentiableSelectionCriterion:
    def __init__(
        self,
        crit: CriterionCallable,
        loader: LoaderLike,
        reduction: str = "mean",
        batches_per_eval: int = 0,
    ):
        """
        Batch differentiable function for parameter optimization.

        Parameters
        ----------
        crit : callable
            Selection criterion of the form f(p, xb, zb) -> scalar, where:
              - p : parameter array
              - xb : batch input
              - zb : batch output/targets
            Returns a scalar loss.
        loader : iterable
            Iterable yielding batches (xb, zb).
        reduction : str, optional
            Reduction mode: 'mean' (default) or 'sum'.
        batches_per_eval : int, default 0
            0  -> run over the whole loader each call (default behavior).
            >0 -> run over exactly this many batches per call, cycling when
                  the iterator is exhausted.
        """
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        if batches_per_eval < 0:
            raise ValueError("batches_per_eval must be ≥ 0")
        self.crit = crit
        self.loader = loader
        self.reduction = reduction
        self.bpe = int(batches_per_eval)
        self._batch_iter = iter(loader) if self.bpe > 0 else None
        self.gradient = None

    def __call__(self, p: ArrayLike) -> ArrayLike:
        return self.evaluate_no_grad(p)

    def _batches(self):
        if self.bpe == 0:
            yield from self.loader
        else:
            for _ in range(self.bpe):
                try:
                    yield next(self._batch_iter)
                except StopIteration:
                    self._batch_iter = iter(self.loader)
                    yield next(self._batch_iter)

    def evaluate(self, p: ArrayLike) -> _np_dtype:
        try:
            total_loss = 0.0
            n_samples = 0
            for xb, zb in self._batches():
                batch_size = xb.shape[0]
                total_loss += self.crit(p, xb, zb) * batch_size
                n_samples += batch_size
            if n_samples == 0:
                raise ValueError("Loader is empty.")
            if self.reduction == "mean":
                total_loss /= n_samples
            return total_loss
        except Exception as exc:
            if _is_linalg_exception(exc):
                return inf
            raise

    def evaluate_pre_grad(self, p: ArrayLike) -> _np_dtype:
        return self.evaluate(p)

    def evaluate_no_grad(self, p: ArrayLike) -> _np_dtype:
        return self.evaluate(p)

# ..................................................

def scaled_distance(loginvrho: ArrayLike, x: ArrayLike, y: ArrayLike) -> ArrayLike:
    invrho = exp(loginvrho)
    xs = invrho * x
    ys = invrho * y
    return cdist(xs, ys)

def scaled_distance_elementwise(
    loginvrho: ArrayLike, x: ArrayLike, y: Optional[ArrayLike]
) -> ArrayLike:
    if x is y or y is None:
        d = zeros((x.shape[0],))
    else:
        invrho = exp(loginvrho)
        d = sqrt(sum((invrho * (x - y)) ** 2, axis=1))
    return d

# ..................................................

def logdet(A):
    sign, logabsdet = numpy.linalg.slogdet(A)
    if sign <= 0:
        raise ValueError(
            "Matrix is not positive definite (or has non-positive determinant)."
        )
    return logabsdet

def cholesky_inv(A):
    # FIXME: slow!
    # n = A.shape[0]
    # C, lower = cho_factor(A)
    # Ainv = cho_solve((C, lower), eye(n))
    return inv(A)

def cholesky_solve(A, b):
    L = cholesky(A)
    y = solve_triangular(L, b, lower=True)
    x = solve_triangular(L.T, y, lower=False)
    return x, L

# ..................................................

# Build one global RNG (or let the user set the seed somewhere):
_np_rng = numpy.random.default_rng(seed=1234)

def set_seed(seed: int) -> None:
    """Set the global NumPy generator seed."""
    global _np_rng
    _np_rng = numpy.random.default_rng(seed=seed)

def rand(*shape: int) -> ArrayLike:
    try:
        return _np_rng.random(shape, dtype=_np_dtype)
    except TypeError:
        return _np_rng.random(shape).astype(_np_dtype, copy=False)

def randn(*shape: int) -> ArrayLike:
    return _np_rng.normal(loc=0, scale=1, size=shape).astype(_np_dtype, copy=False)

def choice(
    a: ArrayLike,
    size: Optional[int] = None,
    replace: bool = True,
    p: Optional[ArrayLike] = None,
) -> ArrayLike:
    return _np_rng.choice(a, size=size, replace=replace, p=p)

def permutation(x: ArrayLike) -> ArrayLike:
    return _np_rng.permutation(x)

class multivariate_normal:
    @staticmethod
    def _mean_array(mean, d: int):
        m = numpy.asarray(mean)
        if m.ndim == 0:
            return numpy.full((d,), float(m), dtype=_np_dtype)
        m = m.astype(_np_dtype, copy=False).reshape(-1)
        if m.size != d:
            raise ValueError("mean has incompatible length.")
        return m

    @staticmethod
    def rvs(mean=0.0, cov=1.0, n=1):
        # Check if cov is a scalar or 1x1 array, and use norm if so
        if isscalar(cov) or (isinstance(cov, numpy.ndarray) and cov.size == 1):
            return normal.rvs(mean, numpy.sqrt(cov), size=n).astype(
                _np_dtype, copy=False
            )

        # For dxd covariance matrix, use multivariate_normal
        cov = numpy.asarray(cov)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be a scalar or a square 2D matrix.")
        d = cov.shape[0]
        mean_array = multivariate_normal._mean_array(mean, d)
        return numpy.asarray(
            scipy_mvnormal.rvs(mean=mean_array, cov=cov, size=n), dtype=_np_dtype
        )

    @staticmethod
    def logpdf(x, mean=0.0, cov=1.0):
        # Check if cov is a scalar or 1x1 array, and use norm if so
        if numpy.isscalar(cov) or (
            isinstance(cov, numpy.ndarray) and cov.size == 1
        ):
            return normal.logpdf(x, mean, numpy.sqrt(cov))

        # For dxd covariance matrix, use multivariate_normal
        x = numpy.asarray(x)
        cov = numpy.asarray(cov)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be a scalar or a square 2D matrix.")
        d = cov.shape[0]
        if x.ndim == 1:
            if x.shape[0] != d:
                raise ValueError("x has incompatible length.")
        else:
            if x.shape[-1] != d:
                raise ValueError("x has incompatible last dimension.")
        mean_array = multivariate_normal._mean_array(mean, d)
        return scipy_mvnormal.logpdf(x, mean=mean_array, cov=cov)

    @staticmethod
    def cdf(x, mean=0.0, cov=1.0):
        # Check if cov is a scalar or 1x1 array, and use norm for the univariate case
        if isscalar(cov) or (isinstance(cov, numpy.ndarray) and cov.size == 1):
            return normal.cdf(x, mean, sqrt(cov))

        # For dxd covariance matrix, use multivariate_normal
        x = numpy.asarray(x)
        cov = numpy.asarray(cov)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be a scalar or a square 2D matrix.")
        d = cov.shape[0]
        if x.ndim == 1:
            if x.shape[0] != d:
                raise ValueError("x has incompatible length.")
        else:
            if x.shape[-1] != d:
                raise ValueError("x has incompatible last dimension.")
        mean = multivariate_normal._mean_array(mean, d)
        return scipy_mvnormal.cdf(x, mean=mean, cov=cov)
