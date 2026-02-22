# gpmp/num/torch_backend.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""Torch numerical backend for GPmp.

This module defines the Torch implementation of the gpmp.num API.
"""

import os
import warnings
import builtins
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from gpmp.config import _normalize_dtype_spec, get_config, init_backend, get_logger

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
#                      TORCH
#
# -----------------------------------------------------

import torch
import numpy
import numpy as np

_torch_dtype = torch.float64
torch.set_default_dtype(_torch_dtype)
_config.dtype_resolved = _torch_dtype

TensorLike = Union[torch.Tensor, float, int]

from torch import tensor, is_tensor

ndarray = torch.Tensor

from torch import (
    reshape,
    where,
    any,
    isnan,
    isinf,
    isfinite,
    isclose,
    allclose,
    hstack,
    vstack,
    stack,
    tile,
    concatenate,
    empty,
    empty_like,
    zeros_like,
    zeros,
    ones,
    full,
    eye,
    diag,
    arange,
    linspace,
    logspace,
    meshgrid,
    floor,
    ceil,
    abs,
    det,
    logdet,
    argmax,
    argmin,
    einsum,
    matmul,
    trace,
    inner,
    logical_not,
    logical_and,
    logical_or,
)
from torch.linalg import cond, cholesky, qr, inv
from torch import rand, randn
from torch import pi, inf
from torch import finfo, float64
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

# ..................................................

eps = finfo(_torch_dtype).eps
fmax = finfo(_torch_dtype).max


def safe_inf():
    """
    Use LinAlgError instead of raising RuntimeError for linalg operations
    https://github.com/pytorch/pytorch/issues/64785
    https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
    extype, value, tb = __import__("sys").exc_info()
    __import__("traceback").print_exc()
    __import__("pdb").post_mortem(tb)
    """
    inf_tensor = tensor(float("inf"), requires_grad=True)
    return inf_tensor  # returns inf with None gradient


def safe_neginf():
    neginf_tensor = tensor(-float("inf"), requires_grad=True)
    return neginf_tensor


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    if copy:
        x_target = x.clone()
    else:
        x_target = None
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf, out=x_target)


# ..................................................

_torch_linalg_error = (
    (torch.linalg.LinAlgError,) if hasattr(torch.linalg, "LinAlgError") else tuple()
)


def _is_linalg_exception(exc: Exception) -> bool:
    if _torch_linalg_error and isinstance(exc, _torch_linalg_error):
        return True
    msg = str(exc).lower()
    return builtins.any(keyword in msg for keyword in _LINALG_ERROR_KEYWORDS)

# ..................................................

def gammaln(x):
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    if t.is_floating_point() and t.dtype != _torch_dtype:
        t = t.to(dtype=_torch_dtype)
    elif not t.is_floating_point():
        t = t.to(dtype=_torch_dtype)
    return torch.lgamma(t)


def copy(x):
    t = asarray(x)
    return t.clone().detach()


def array_equal(x, y):
    return torch.equal(x, y)


def expand_dims(tensor, axis):
    return tensor.unsqueeze(axis)

# ..................................................

def array(x, dtype=None):
    return asarray(x, dtype=dtype)


def _resolve_torch_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is float:
        return _torch_dtype
    if dtype is int:
        return torch.long
    if dtype is bool:
        return torch.bool
    s = str(dtype).lower()
    if "float32" in s:
        return torch.float32
    if "float64" in s or "double" in s:
        return torch.float64
    if "int64" in s or "long" in s:
        return torch.int64
    if "int32" in s:
        return torch.int32
    if "bool" in s:
        return torch.bool
    return dtype


def asarray(x, dtype=None):
    dtype = _resolve_torch_dtype(dtype)
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            return x if x.dtype == dtype else x.to(dtype=dtype)
        if x.is_floating_point() and x.dtype != _torch_dtype:
            return x.to(dtype=_torch_dtype)
        return x
    if isinstance(x, numpy.ndarray):
        try:
            x_ = torch.from_numpy(x)
        except (TypeError, ValueError):
            x_ = torch.as_tensor(x)
        if dtype is not None:
            return x_ if x_.dtype == dtype else x_.to(dtype=dtype)
        if x_.is_floating_point() and x_.dtype != _torch_dtype:
            return x_.to(dtype=_torch_dtype)
        return x_
    if isinstance(x, (int, float)):
        if dtype is None and isinstance(x, float):
            dtype = _torch_dtype
        return torch.tensor(x, dtype=dtype)
    x_ = torch.as_tensor(x)
    if dtype is not None:
        return x_ if x_.dtype == dtype else x_.to(dtype=dtype)
    if x_.is_floating_point() and x_.dtype != _torch_dtype:
        x_ = x_.to(dtype=_torch_dtype)
    return x_


def asdouble(x):
    return asarray(x).to(torch.double)


def asint(x):
    return asarray(x).to(torch.int)


def to_np(x):
    if is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def to_scalar(x):
    return x.item()


def isarray(x):
    return torch.is_tensor(x)


def isscalar(x):
    if torch.is_tensor(x):
        return x.reshape(-1).size()[0] == 1
    elif isinstance(x, (int, float)):
        return True
    else:
        return False


def scalar_safe(f):
    def f_(x):
        # Fast path: already a tensor. Do not recast here.
        if torch.is_tensor(x):
            return f(x)

        # Scalars / lists / numpy arrays: convert once.
        # Use dtype=_torch_dtype so scalar literals become float64 consistently.
        t = torch.as_tensor(x, dtype=_torch_dtype)
        return f(t)

    return f_


# def scalar_safe(f):
#     def f_(x):
#         return f(asarray(x))

#     return f_


# ..................................................

def _sizes_from_bounds(bounds, length):
    """
    Convert 1-D tensor of strictly increasing boundary indices
    into chunk-size tensor expected by torch.split.

    Example
    -------
    length = 12, bounds = [3, 6, 9]  ->  sizes = [3, 3, 3, 3]
    """
    # sizes = [bounds[0], diff(bounds), length - bounds[-1]]
    return torch.cat(
        (
            bounds[:1],
            torch.diff(bounds),
            torch.tensor(
                [length - bounds[-1]], device=bounds.device, dtype=bounds.dtype
            ),
        )
    )


def split(x, indices_or_sections, dim=0):
    """
    Accepts either
    * **int n** -> chunks of size *n*   (Torch default)
    * **list / 1-D tensor of sizes**
    * **list / 1-D tensor of boundary indices** (NumPy style)

    Heuristic: if the sum of the numbers == `x.size(dim)`, treat them
    as *sizes*, otherwise as *boundary indices*.
    """
    # fast path: integer size
    if isinstance(indices_or_sections, int):
        return torch.split(x, indices_or_sections, dim=dim)
    # convert to 1-D long tensor on same device
    if torch.is_tensor(indices_or_sections):
        vec = indices_or_sections.to(dtype=torch.long, device=x.device)
    else:
        vec = torch.as_tensor(indices_or_sections, dtype=torch.long, device=x.device)
    if vec.numel() == 0:
        raise ValueError("indices_or_sections must contain at least one element")
    # decide “sizes” vs “bounds”
    total = vec.sum()
    length = x.size(dim)
    if total == length:
        sizes = vec  # already sizes
    else:
        sizes = _sizes_from_bounds(vec, length)
    # torch.split needs a list
    return torch.split(x, sizes.tolist(), dim=dim)


def transpose(x, dim0, dim1):
    """Torch-style transpose: swap two dimensions."""
    return torch.transpose(x, dim0, dim1)

# ..................................................

log = scalar_safe(torch.log)
log10 = scalar_safe(torch.log10)
log1p = scalar_safe(torch.log1p)
exp = scalar_safe(torch.exp)
sqrt = scalar_safe(torch.sqrt)
sin = scalar_safe(torch.sin)
cos = scalar_safe(torch.cos)
tan = scalar_safe(torch.tan)
tanh = scalar_safe(torch.tanh)

# ..................................................

def axis_to_dim(f):
    def f_(x, axis=None, **kwargs):
        if axis is None:
            return f(x, **kwargs)
        else:
            return f(x, dim=axis, **kwargs)

    return f_


all = axis_to_dim(torch.all)
unique = axis_to_dim(torch.unique)
diff = axis_to_dim(torch.diff)
sum = axis_to_dim(torch.sum)
cumsum = axis_to_dim(torch.cumsum)
prod = axis_to_dim(torch.prod)
mean = axis_to_dim(torch.mean)

# ..................................................

def std(x, axis=None, ddof=0, keepdims=False):
    correction = int(ddof)
    if axis is None:
        return torch.std(x, correction=correction, keepdim=keepdims)
    return torch.std(x, dim=axis, correction=correction, keepdim=keepdims)


def var(x, axis=None, ddof=0, keepdims=False):
    correction = int(ddof)
    if axis is None:
        return torch.var(x, correction=correction, keepdim=keepdims)
    return torch.var(x, dim=axis, correction=correction, keepdim=keepdims)


def cov(
    m,
    y=None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights=None,
    aweights=None,
    dtype=None,
):
    x = asarray(m, dtype=_resolve_torch_dtype(dtype))
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if not rowvar:
        x = x.transpose(0, 1)
    if y is not None:
        y = asarray(y, dtype=_resolve_torch_dtype(dtype))
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if not rowvar:
            y = y.transpose(0, 1)
        x = concatenate((x, y), axis=0)

    correction = int(ddof) if ddof is not None else (0 if bias else 1)
    return torch.cov(x, correction=correction, fweights=fweights, aweights=aweights)

# ..................................................

def percentile(
    x,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
    interpolation=None,
):
    if interpolation is not None and interpolation != "linear":
        raise ValueError("Only 'linear' interpolation is supported.")
    if weights is not None:
        raise NotImplementedError("weights not supported in torch percentile")
    if method != "linear":
        raise NotImplementedError("only 'linear' method supported in this wrapper")

    return torch.quantile(x, q / 100.0, dim=axis, keepdim=keepdims, out=out)


def norm(x, axis=None, ord=2):
    return torch.norm(x, dim=axis, p=ord)


def min(x, axis=0, keepdims=False):
    m = torch.min(x, dim=axis)
    return m.values.unsqueeze(axis) if keepdims else m.values


def max(x, axis=0, keepdims=False):
    m = torch.max(x, dim=axis)
    return m.values.unsqueeze(axis) if keepdims else m.values


def maximum(x1, x2):
    if torch.is_tensor(x1) and torch.is_tensor(x2):
        return torch.maximum(x1, x2)
    return torch.maximum(asarray(x1), asarray(x2))


def minimum(x1, x2):
    if torch.is_tensor(x1) and torch.is_tensor(x2):
        return torch.minimum(x1, x2)
    return torch.minimum(asarray(x1), asarray(x2))


def clip(x, min=None, max=None, out=None):
    if not torch.is_tensor(x):
        x = asarray(x)
    if min is not None and not torch.is_tensor(min):
        min = asarray(min).to(dtype=x.dtype)
    if max is not None and not torch.is_tensor(max):
        max = asarray(max).to(dtype=x.dtype)
    return torch.clamp(x, min=min, max=max, out=out)


def sort(x, axis=-1):
    xsorted = torch.sort(x, dim=axis)
    return xsorted.values


def inftobigf(a, bigf=fmax / 1000.0):
    a = torch.where(torch.isinf(a), torch.full_like(a, bigf), a)
    return a

# ..................................................

def grad(f: Callable[[ArrayLike], ArrayLike]) -> Callable[[ArrayLike], ArrayLike]:
    def f_grad(x):
        x = asarray(x).detach().clone().requires_grad_(True)
        y = f(x)
        gradients = torch.autograd.grad(y, x, allow_unused=True)[0]
        return gradients

    return f_grad


def value_and_grad(f, x):
    # Returns (y, grady) with y = f(x)
    with torch.enable_grad():
        x_ = x.detach().requires_grad_(True)
        y = f(x_)
        if not torch.is_tensor(y):
            raise ValueError("f(x) must return a torch scalar tensor.")
        if y.ndim != 0:
            if y.numel() == 1:
                y = y.reshape(())
            else:
                raise ValueError("f(x) must return a scalar.")
        if not torch.isfinite(y):
            return y.detach(), torch.zeros_like(x_).detach()
        (g,) = torch.autograd.grad(y, x_, create_graph=False, allow_unused=True)
        if g is None:
            g = torch.zeros_like(x_)
    return y.detach(), g.detach()


# def value_and_grad(f, x):
#     # Returns (y, grady) with y = f(x)
#     with torch.enable_grad():
#         x_ = x.detach().requires_grad_(True)
#         y = f(x_)
#         if y.ndim != 0:
#             raise ValueError("f(x) must return a scalar.")
#         (g,) = torch.autograd.grad(y, x_, create_graph=False)
#     return y.detach(), g.detach()


class DifferentiableSelectionCriterion:
    """Wraps a selection criterion f(p, x, z) -> scalar, allowing gradient computation."""

    def __init__(self, f: CriterionCallable, x: ArrayLike, z: ArrayLike):
        self.f = f
        self.x = x
        self.z = z
        self._p_value = None
        self._f_value = None

    def __call__(self, p: ArrayLike) -> _torch_dtype:
        return self.evaluate(p)

    def evaluate(self, p: ArrayLike) -> _torch_dtype:
        return self.f(p, self.x, self.z)

    def evaluate_no_grad(self, p: ArrayLike) -> _torch_dtype:
        p = asarray(p)
        try:
            with torch.no_grad():
                f_value = self.f(p, self.x, self.z)
            return f_value
        except Exception as exc:
            if _is_linalg_exception(exc):
                return inf
            raise

    def evaluate_pre_grad(self, p: ArrayLike) -> _torch_dtype:
        self._p_value = asarray(p).detach().clone().requires_grad_(True)
        try:
            self._f_value = self.f(self._p_value, self.x, self.z)
            return self._f_value.item()
        except Exception as exc:
            if _is_linalg_exception(exc):
                self._f_value = torch.tensor(float("inf"), requires_grad=True)
                return self._f_value.item()
            raise

    def gradient(
        self, p: ArrayLike, retain: bool = False, allow_unused: bool = True
    ) -> ArrayLike:
        if self._f_value is None:
            raise ValueError("Call 'evaluate_pre_grad(p)' before 'gradient(p)'")

        if not torch.equal(asarray(p), self._p_value):
            raise ValueError(
                "The input 'p' in 'gradient' must be the same as in 'evaluate'"
            )

        gradients = torch.autograd.grad(
            self._f_value,
            self._p_value,
            retain_graph=retain,
            allow_unused=allow_unused,
        )[0]
        if gradients is None:
            raise RuntimeError("Gradient is None.")
        return gradients


class BatchDifferentiableSelectionCriterion:
    """
    Scalar selection criterion evaluated on mini-batches.

    Parameters
    ----------
    crit : callable
        `crit(param, x_batch, z_batch) -> scalar tensor` (only `param`
        needs `requires_grad`).
    loader : torch.utils.data.DataLoader
    reduction : {'mean', 'sum'}, default 'mean'
    batches_per_eval : int, default 0
        0  -> run over the *whole* loader each call (default behaviour).
        >0 -> run over exactly this many batches per call, cycling when
              the iterator is exhausted.
    """

    def __init__(
        self,
        crit: CriterionCallable,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        reduction: str = "mean",
        batches_per_eval: int = 0,
    ):
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        if batches_per_eval < 0:
            raise ValueError("batches_per_eval must be >= 0")
        if len(loader) == 0:
            raise ValueError("DataLoader is empty.")

        self.crit = crit
        self.loader = loader
        self.reduction = reduction
        self.bpe = int(batches_per_eval)

        self._batch_iter = iter(loader) if self.bpe > 0 else None
        self._gradient = None  # cached grad from last evaluate

    @staticmethod
    def _prepare_param(param: TensorLike, *, req_grad: bool = False) -> torch.Tensor:
        """Convert to tensor and set `requires_grad` (no dtype/device casting)."""
        param = asarray(param)
        param.requires_grad_(req_grad)
        return param

    def _batches(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield the next set of batches for one evaluation."""
        if self.bpe == 0:  # full epoch
            yield from self.loader
        else:  # fixed count, cycling
            for _ in range(self.bpe):
                try:
                    yield next(self._batch_iter)
                except StopIteration:
                    self._batch_iter = iter(self.loader)
                    yield next(self._batch_iter)

    def evaluate(self, param: TensorLike) -> torch.Tensor:
        total, n = 0.0, 0
        for xb, zb in self._batches():
            bs = xb.shape[0]
            loss = self.crit(param, xb, zb) * bs
            total += loss.item()
            n += bs
        if n == 0:
            raise ValueError("Loader is empty.")
        if self.reduction == "mean":
            total /= n
        return total

    def evaluate_no_grad(self, param: TensorLike) -> _torch_dtype:
        total, n = 0.0, 0
        with torch.no_grad():
            p = self._prepare_param(param, req_grad=False)
            for xb, zb in self._batches():
                bs = xb.shape[0]
                total += self.crit(p, xb, zb).item() * bs
                n += bs
        if n == 0:
            raise ValueError("Loader is empty.")
        return total / n if self.reduction == "mean" else total

    def evaluate_pre_grad(self, param: TensorLike) -> _torch_dtype:
        p = self._prepare_param(param, req_grad=True)
        grad = None  # accumulated gradient
        total, n = 0.0, 0
        for xb, zb in self._batches():
            bs = xb.shape[0]
            loss = self.crit(p, xb, zb) * bs
            g = torch.autograd.grad(loss, p, retain_graph=False)[0]
            if grad is None:
                grad = g
            else:
                grad += g
            total += loss.item()
            n += bs
        if n == 0:
            raise ValueError("Loader is empty.")
        if self.reduction == "mean":
            total /= n
            grad /= n
        self._gradient = grad.detach()
        return total

    # -----------------------------------------------------------------
    def gradient(self, _param: TensorLike) -> torch.Tensor:
        if self._gradient is None:
            raise RuntimeError("Call `evaluate` first.")
        return self._gradient


class SecondOrderDifferentiableFunction:
    """Helper class to compute second-order derivatives (Hessian) of scalar functions."""

    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]):
        """
        Parameters
        ----------
        f : callable
            Function f(x) returning a scalar given input tensor x.
        """
        self.f = f
        self._x = None
        self._y = None
        self._grad = None

    def evaluate(self, x: TensorLike) -> torch.Tensor:
        """Evaluate the function at x and store gradient graph."""
        x = asarray(x).detach().clone().requires_grad_(True)
        self._x = x
        self._y = self.f(self._x)

        if self._y.dim() != 0:
            raise ValueError("Function output must be a scalar.")

        return self._y

    def gradient(self, retain: bool = True) -> torch.Tensor:
        """Compute and return gradient of the function at stored x."""
        if self._y is None or self._x is None:
            raise RuntimeError("Call evaluate(x) before calling gradient().")

        (grad,) = torch.autograd.grad(
            self._y, self._x, create_graph=True, retain_graph=retain
        )
        self._grad = grad
        return grad.detach()

    def hessian(self) -> torch.Tensor:
        """Compute and return Hessian of the function at stored x."""
        if self._grad is None:
            raise RuntimeError("Call gradient() before calling hessian().")

        x = self._x
        grad = self._grad
        n = x.numel()
        hessian = torch.zeros((n, n), dtype=x.dtype, device=x.device)

        for idx in range(n):
            (grad2,) = torch.autograd.grad(
                grad[idx], x, retain_graph=True, allow_unused=True
            )
            if grad2 is None:
                raise RuntimeError(f"Second derivative for parameter {idx} is None.")
            hessian[idx] = grad2

        # Ensure symmetry
        hessian = 0.5 * (hessian + hessian.T)

        return hessian.detach()

# ..................................................

def custom_sqrt(x):
    arbitrary_value = 1.0
    mask = x == 0.0
    x_copy = torch.where(mask, arbitrary_value, x)
    res = torch.where(mask, 0.0, sqrt(x_copy))
    return res


def cdist(x, y, zero_diagonal=True):
    if x is y:
        # use view method: requires contiguous tensor
        x_norm = (x**2).sum(1).view(-1, 1)
        distances = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
    else:
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        distances = x_norm + y_norm - 2.0 * torch.mm(x, y.t())

    distances = custom_sqrt(distances.clamp(min=0.0))

    if zero_diagonal and x is y:
        mask = torch.eye(distances.size(0), dtype=torch.bool, device=x.device)
        distances = distances.masked_fill(mask, 0.0)

    return distances


def scaled_distance(loginvrho, x, y):
    invrho = exp(loginvrho)
    xs = invrho * x

    if x is y:
        d = cdist(xs, xs)
    else:
        ys = invrho * y
        d = cdist(xs, ys)

    return d


def scaled_distance_elementwise(loginvrho, x, y):
    if x is y or y is None:
        d = zeros((x.shape[0],))
    else:
        invrho = exp(loginvrho)
        d = sqrt(sum((invrho * (x - y)) ** 2, axis=1))
    return d

# ..................................................

def svd(A, full_matrices=True, hermitian=True):
    return torch.linalg.svd(A, full_matrices)


def solve(A, B, overwrite_a=True, overwrite_b=True, assume_a="gen", sym_pos=False):
    return torch.linalg.solve(A, B)


def solve_triangular(
    A,
    B,
    trans=0,
    lower=False,
    unit_diagonal=False,
    overwrite_b=False,
    check_finite=False,
):
    if trans in (0, "N", "n"):
        Aop = A
    elif trans in (1, "T", "t"):
        Aop = A.mT
    elif trans in (2, "C", "c"):
        Aop = A.mH
    else:
        raise ValueError(f"Invalid trans={trans!r}; expected 0/1/2 or 'N'/'T'/'C'.")
    x = torch.linalg.solve_triangular(
        Aop,
        B,
        upper=not lower,
        left=True,
        unitriangular=unit_diagonal,
    )
    return x


def cho_factor(A, lower=False, overwrite_a=False, check_finite=True):
    C = cholesky(A, upper=not lower)
    return (C, lower)


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    C, lower = c_and_lower
    b = asarray(b)
    return torch.cholesky_solve(b, C, upper=not lower)


def cholesky_solve(A, b):
    if b.dim() == 1:
        b = b.reshape(-1, 1)
    L = cholesky(A)
    y = torch.linalg.solve_triangular(L, b, upper=False)
    x = torch.linalg.solve_triangular(L.t(), y, upper=True)
    return x, L


def cholesky_inv(A):
    C = cholesky(A)
    return torch.cholesky_inverse(C)

# ..................................................

# Build a global Torch Generator
_torch_gen = torch.Generator()
_torch_gen.manual_seed(1234)


def set_seed(seed):
    """Set the global Torch generator seed."""
    global _torch_gen
    _torch_gen = torch.Generator()
    _torch_gen.manual_seed(seed)


def rand(*shape):
    return torch.rand(shape, generator=_torch_gen)


def randn(*shape):
    return torch.randn(shape, generator=_torch_gen)


def choice(a, size=None, replace=True, p=None):
    if size is None:
        size = 1

    if not torch.is_tensor(a):
        a = (
            torch.arange(a)
            if isinstance(a, int)
            else torch.tensor(a, dtype=_torch_dtype)
        )

    n = a.shape[0]

    if p is not None:
        p = torch.tensor(p, dtype=_torch_dtype, device=a.device)
        p = p / p.sum()
        indices = torch.multinomial(
            p, num_samples=size, replacement=replace, generator=_torch_gen
        )
    else:
        if replace:
            indices = torch.randint(
                0, n, (size,), generator=_torch_gen, device=a.device
            )
        else:
            perm = torch.randperm(n, generator=_torch_gen, device=a.device)
            indices = perm[:size]

    return a.index_select(0, indices)


def permutation(x):
    if isinstance(x, int):
        return torch.randperm(x, generator=_torch_gen)
    else:
        n = x.shape[0]
        perm = torch.randperm(n, generator=_torch_gen, device=x.device)
        return x.index_select(0, perm)


class normal:
    @staticmethod
    def cdf(x, loc=0.0, scale=1.0):
        x = asarray(x)
        d = Normal(loc, scale)
        return d.cdf(x)

    @staticmethod
    def logcdf(x, loc=0.0, scale=1.0):
        x = asarray(x)
        d = Normal(loc, scale)
        return log(d.cdf(x))

    @staticmethod
    def pdf(x, loc=0.0, scale=1.0):
        x = asarray(x)
        t = (x - loc) / scale
        return 1 / sqrt(2 * pi) * exp(-0.5 * t**2)

    @staticmethod
    def logpdf(x, loc=0.0, scale=1.0):
        d = Normal(loc, scale)
        return d.log_prob(asarray(x))


class multivariate_normal:
    @staticmethod
    def _as_torch_float(x, *, device=None) -> torch.Tensor:
        t = asarray(x, dtype=_torch_dtype)
        if device is not None and t.device != device:
            t = t.to(device=device)
        return t

    @staticmethod
    def _mean_vector(mean, d: int, device) -> torch.Tensor:
        m = asarray(mean)
        if torch.is_tensor(m) and m.numel() > 1:
            m = multivariate_normal._as_torch_float(m, device=device).reshape(-1)
            if m.numel() != d:
                raise ValueError("mean has incompatible length.")
            return m
        return torch.full(
            (d,),
            float(asarray(mean).reshape(()).item()),
            dtype=_torch_dtype,
            device=device,
        )

    @staticmethod
    def rvs(mean=0.0, cov=1.0, n=1):
        cov_t = multivariate_normal._as_torch_float(cov)

        # scalar variance (Python scalar, 0-d tensor, 1x1 tensor, or any 1-element tensor)
        if cov_t.numel() == 1:
            var = cov_t.reshape(())
            mean_s = multivariate_normal._as_torch_float(
                mean, device=var.device
            ).reshape(())
            dist = Normal(mean_s, var.sqrt())
            return dist.sample((n,))

        # matrix covariance
        if cov_t.ndim != 2 or cov_t.shape[0] != cov_t.shape[1]:
            raise ValueError("cov must be a scalar or a square 2D matrix.")
        d = cov_t.shape[0]
        mean_vec = multivariate_normal._mean_vector(mean, d, cov_t.device)
        dist = MultivariateNormal(mean_vec, covariance_matrix=cov_t)
        return dist.sample((n,))

    @staticmethod
    def logpdf(x, mean=0.0, cov=1.0):
        x_t = multivariate_normal._as_torch_float(x)
        cov_t = multivariate_normal._as_torch_float(cov, device=x_t.device)

        # scalar variance
        if cov_t.numel() == 1:
            var = cov_t.reshape(())
            mean_s = multivariate_normal._as_torch_float(
                mean, device=var.device
            ).reshape(())
            dist = Normal(mean_s, var.sqrt())
            x_u = x_t.squeeze(-1) if (x_t.ndim > 0 and x_t.shape[-1] == 1) else x_t
            return dist.log_prob(x_u)

        # matrix covariance
        if cov_t.ndim != 2 or cov_t.shape[0] != cov_t.shape[1]:
            raise ValueError("cov must be a scalar or a square 2D matrix.")
        d = cov_t.shape[0]
        mean_vec = multivariate_normal._mean_vector(mean, d, cov_t.device)

        if x_t.shape[-1] != d:
            raise ValueError("x has incompatible last dimension.")

        dist = MultivariateNormal(mean_vec, covariance_matrix=cov_t)
        return dist.log_prob(x_t)

    @staticmethod
    def cdf(x, mean=0.0, cov=1.0):
        # SciPy backend, CPU only.
        try:
            from scipy.stats import norm as _sp_norm
            from scipy.stats import multivariate_normal as _sp_mvn
        except Exception as exc:
            raise ImportError(
                "SciPy is required for multivariate_normal.cdf in the torch backend."
            ) from exc

        x_np = to_np(asarray(x)) if torch.is_tensor(x) else np.asarray(x)
        cov_np = to_np(asarray(cov)) if torch.is_tensor(cov) else np.asarray(cov)
        mean_np = to_np(asarray(mean)) if torch.is_tensor(mean) else np.asarray(mean)

        if np.isscalar(cov_np) or cov_np.size == 1:
            m0 = float(np.asarray(mean_np).reshape(()))
            s0 = float(np.sqrt(np.asarray(cov_np).reshape(())))
            return asarray(_sp_norm.cdf(x_np, loc=m0, scale=s0))

        if np.isscalar(mean_np) or np.asarray(mean_np).ndim == 0:
            d = cov_np.shape[0]
            mean_np = np.full((d,), float(np.asarray(mean_np).reshape(())))
        return asarray(_sp_mvn.cdf(x_np, mean=mean_np, cov=cov_np))
