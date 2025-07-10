"""num.py — Numerical backend for GPmp.

This module provides numerical operations for GPmp, with a unified API
based on an underlying backend.

The backend is selected automatically among:
    1. torch
    2. jax
    3. numpy (default fallback)

All basic array operations, linear algebra routines, random sampling,
and differentiable functions are defined here.

The active backend is stored in the environment variable 'GPMP_BACKEND'.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022–2025, CentraleSupélec
License: GPLv3 (see LICENSE)

"""

import os
import warnings
import logging
from importlib.util import find_spec

# Setup a logger
_logger = logging.getLogger("gpmp")
if not _logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)

# Detect backend
_gpmp_backend_ = os.environ.get("GPMP_BACKEND")


def set_backend_env_var(backend):
    global _gpmp_backend_
    os.environ["GPMP_BACKEND"] = backend
    _gpmp_backend_ = backend


if _gpmp_backend_ is None:
    if find_spec("torch") is not None:
        set_backend_env_var("torch")
    elif find_spec("jax") is not None:
        set_backend_env_var("jax")
    else:
        set_backend_env_var("numpy")

_logger.info(f"Using backend: {_gpmp_backend_}")


# -----------------------------------------------------
#
#                      NUMPY
#
# -----------------------------------------------------
if _gpmp_backend_ == "numpy":
    import numpy
    from numpy import array, empty
    from numpy.typing import NDArray

    ndarray = NDArray[numpy.float64]
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

    eps = finfo(float64).eps
    fmax = numpy.finfo(numpy.float64).max
    # ..................................................

    def set_elem_1d(x, index, v):
        x[index] = v
        return x

    def set_elem_2d(x, i, j, v):
        x[i, j] = v
        return x

    def set_row_2d(A, index, x):
        A[index, :] = x
        return A

    def set_col_2d(A, index, x):
        A[:, index] = x
        return A

    def set_col_3d(A, index, x):
        A[:, :, index] = x
        return A

    def index_select(x, dim, indices):
        if dim == 0:
            return x[indices]
        elif dim == 1:
            # manual slicing for dim=1
            return x[:, indices]
        else:
            # general fallback: move dimension to front
            x_moved = numpy.moveaxis(x, dim, 0)
            x_selected = x_moved[indices]
            return numpy.moveaxis(x_selected, 0, dim)

    # ..................................................

    def asarray(x, dtype=None):
        if isinstance(x, numpy.ndarray):
            return x
        elif isinstance(x, (int, float)):
            return numpy.array([x], dtype=dtype)
        else:
            return numpy.asarray(x, dtype=dtype)

    def asdouble(x):
        return x.astype(float64)

    def asint(x):
        return x.astype(int)

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

    class jax:
        @staticmethod
        def jit(f, *args, **kwargs):
            return f

    # ..................................................

    def grad(f):
        return None

    class DifferentiableSelectionCriterion:
        def __init__(self, crit, x, z):
            self.crit = crit
            self.x, self.z = x, z
            self.gradient = None

        def __call__(self, p):
            return self.evaluate(p)

        def evaluate(self, p):
            try:
                return self.crit(p, self.x, self.z)
            except Exception:
                return inf

        def evaluate_no_grad(self, p):
            return self.evaluate(p)

    class BatchDifferentiableSelectionCriterion:
        def __init__(self, crit, loader, reduction="mean"):
            """
            Batch differentiable function for parameter optimization.

            Parameters
            ----------
            crit : callable
                Function taking three arguments (p, xb, zb), where:
                  - p : parameter array
                  - xb : batch input
                  - zb : batch output/targets
                Returns a scalar loss.

            loader : iterable
                Iterable yielding batches (xb, zb).

            reduction : str, optional
                Reduction mode: 'mean' (default) or 'sum'.
            """
            self.crit = crit
            self.loader = loader
            self.reduction = reduction
            self.gradient = None

        def __call__(self, p):
            return self.evaluate_no_grad(p)

        def evaluate_no_grad(self, p):
            return self.evaluate(p)

        def evaluate(self, p):
            try:
                total_loss = 0.0
                n_samples = 0
                for xb, zb in self.loader:
                    batch_size = xb.shape[0]
                    total_loss += self.crit(p, xb, zb) * batch_size
                    n_samples += batch_size
                if n == 0:
                    raise ValueError("Loader is empty.")
                if self.reduction == "mean":
                    total_loss /= n_samples
                return total_loss
            except Exception:
                return inf

    # ..................................................

    def scaled_distance(loginvrho, x, y):
        invrho = exp(loginvrho)
        xs = invrho * x
        ys = invrho * y
        return cdist(xs, ys)

    def scaled_distance_elementwise(loginvrho, x, y):
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

    def set_seed(seed):
        """Set the global NumPy generator seed."""
        global _np_rng
        _np_rng = numpy.random.default_rng(seed=seed)

    def rand(*shape):
        return _np_rng.random(shape)

    def randn(*shape):
        return _np_rng.normal(loc=0, scale=1, size=shape)

    def choice(a, size=None, replace=True, p=None):
        return _np_rng.choice(a, size=size, replace=replace, p=p)

    def permutation(x):
        return _np_rng.permutation(x)

    class multivariate_normal:
        @staticmethod
        def rvs(mean=0.0, cov=1.0, n=1):
            # Check if cov is a scalar or 1x1 array, and use norm if so
            if isscalar(cov) or (isinstance(cov, numpy.ndarray) and cov.size == 1):
                return normal.rvs(mean, numpy.sqrt(cov), size=n)

            # For dxd covariance matrix, use multivariate_normal
            d = cov.shape[0]  # Dimensionality from the covariance matrix
            mean_array = numpy.full(d, mean)  # Expand mean to an array
            return scipy_mvnormal.rvs(mean=mean_array, cov=cov, size=n)

        @staticmethod
        def logpdf(x, mean=0.0, cov=1.0):
            # Check if cov is a scalar or 1x1 array, and use norm if so
            if numpy.isscalar(cov) or (
                isinstance(cov, numpy.ndarray) and cov.size == 1
            ):
                return normal.logpdf(x, mean, numpy.sqrt(cov))

            # For dxd covariance matrix, use multivariate_normal
            d = x.shape[-1] if x.ndim > 1 else 1  # Infer dimensionality from x
            mean_array = numpy.full(d, mean)  # Expand mean to an array
            return scipy_mvnormal.logpdf(x, mean=mean_array, cov=cov)

        @staticmethod
        def cdf(x, mean=0.0, cov=1.0):
            # Check if cov is a scalar or 1x1 array, and use norm for the univariate case
            if isscalar(cov) or (isinstance(cov, numpy.ndarray) and cov.size == 1):
                return normal.cdf(x, mean, sqrt(cov))

            # For dxd covariance matrix, use multivariate_normal
            if isinstance(mean, (float, int)):
                d = cov.shape[0]  # Dimensionality from the covariance matrix
                mean = full(d, mean)  # Expand mean to an array
            return scipy_mvnormal.cdf(x, mean=mean, cov=cov)


# -----------------------------------------------------
#
#                      TORCH
#
# -----------------------------------------------------
elif _gpmp_backend_ == "torch":
    import torch

    torch.set_default_dtype(torch.float64)
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
        cov,
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
    from scipy.stats import multivariate_normal as scipy_mvnormal
    from scipy.special import gammaln

    # ..................................................

    eps = finfo(float64).eps
    fmax = finfo(float64).max
    # ..................................................

    def copy(x):
        if isinstance(x, torch.Tensor):
            return x.clone().detach()
        else:
            return torch.clone(tensor(x))

    def array_equal(x, y):
        return torch.equal(x, y)

    def set_elem_1d(x, index, v):
        x[index] = v
        return x

    def set_elem_2d(x, i, j, v):
        x[i, j] = v
        return x

    def set_row_2d(A, index, x):
        A[index, :] = x
        return A

    def set_col_2d(A, index, x):
        A[:, index] = x
        return A

    def set_col_3d(A, index, x):
        A[:, :, index] = x
        return A

    def index_select(x, dim, indices):
        return x.index_select(dim, indices)

    def expand_dims(tensor, axis):
        return tensor.unsqueeze(axis)

    # ..................................................

    def array(x: list):
        return tensor(x)

    def asarray(x, dtype=None):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, (int, float)):
            return tensor([x], dtype=dtype)
        else:
            return torch.asarray(x, dtype=dtype)

    def asdouble(x):
        return x.to(torch.double)

    def asint(x):
        return x.to(torch.int)

    def to_np(x):
        if is_tensor(x):
            return x.numpy()
        else:
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
            if torch.is_tensor(x):
                return f(x)
            else:
                return f(torch.tensor(x))

        return f_

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
        * **list / 1-D tensor of boundary indices** (NumPy/JAX style)

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
            vec = torch.as_tensor(
                indices_or_sections, dtype=torch.long, device=x.device
            )
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
    std = axis_to_dim(torch.std)
    var = axis_to_dim(torch.var)
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
        if not torch.is_tensor(x1):
            x1 = tensor(x1)
        if not torch.is_tensor(x2):
            x2 = tensor(x2)
        return torch.maximum(x1, x2)

    def minimum(x1, x2):
        if not torch.is_tensor(x1):
            x1 = tensor(x1)
        if not torch.is_tensor(x2):
            x2 = tensor(x2)
        return torch.minimum(x1, x2)

    def clip(x, min=None, max=None, out=None):
        if not torch.is_tensor(x):
            x = tensor(x)
        return torch.clamp(x, min, max, out)

    def sort(x, axis=-1):
        xsorted = torch.sort(x, dim=axis)
        return xsorted.values

    def inftobigf(a, bigf=fmax / 1000.0):
        a = torch.where(torch.isinf(a), torch.full_like(a, bigf), a)
        return a

    # ..................................................

    def grad(f):
        def f_grad(x):
            if not torch.is_tensor(x):
                x = torch.tensor(x, requires_grad=True)
            else:
                x = x.detach().clone().requires_grad_(True)

            y = f(x)
            gradients = torch.autograd.grad(y, x, allow_unused=True)[0]
            return gradients

        return f_grad

    class jax:
        @staticmethod
        def jit(f, *args, **kwargs):
            return f

    class DifferentiableSelectionCriterion:
        """Wraps a selection criterion f(p, x, z) -> scalar, allowing gradient computation."""

        def __init__(self, f, x, z):
            self.f = f
            self.x = x
            self.z = z
            self._p_value = None
            self._f_value = None

        def __call__(self, p):
            return self.evaluate_no_grad(p)

        def evaluate_no_grad(self, p):
            if not torch.is_tensor(p):
                p = torch.tensor(p)
            try:
                with torch.no_grad():
                    f_value = self.f(p, self.x, self.z)
                return f_value.item()
            except Exception:
                return inf

        def evaluate(self, p):
            if not torch.is_tensor(p):
                self._p_value = torch.tensor(p, requires_grad=True)
            else:
                self._p_value = p.detach().clone().requires_grad_(True)

            try:
                self._f_value = self.f(self._p_value, self.x, self.z)
                return self._f_value.item()
            except Exception:
                # construct inf with None gradient
                self._f_value = gnp.tensor(float("inf"), requires_grad=True)
                return self._f_value.item()

        def gradient(self, p, retain=False, allow_unused=True):
            if self._f_value is None:
                raise ValueError("Call 'evaluate(p)' before 'gradient(p)'")

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
        def __init__(self, crit, loader, reduction="mean"):
            self.crit = crit
            self.loader = loader
            self.reduction = reduction
            self._gradient = None

        def _prepare_param(self, param, ref_tensor, requires_grad=False):
            if not isinstance(param, torch.Tensor):
                param = torch.tensor(
                    param, dtype=ref_tensor.dtype, device=ref_tensor.device
                )
            else:
                param = param.to(dtype=ref_tensor.dtype, device=ref_tensor.device)
            param.requires_grad_(requires_grad)
            return param

        def evaluate_no_grad(self, param):
            total, n = 0.0, 0
            with torch.no_grad():
                for xb, zb in self.loader:
                    p = self._prepare_param(param, xb)
                    total += self.crit(p, xb, zb).item() * xb.size(0)
                    n += xb.size(0)
            if n == 0:
                raise ValueError("Loader is empty.")
            return total / n if self.reduction == "mean" else total

        def evaluate(self, param):
            total, n = 0.0, 0
            first_batch = True
            for xb, zb in self.loader:
                p = self._prepare_param(param, xb, requires_grad=True)
                loss = self.crit(p, xb, zb)
                total += loss.item() * xb.size(0)
                n += xb.size(0)
                if first_batch:
                    first_batch = False
                    grad = torch.autograd.grad(loss, p, retain_graph=False)[0]
                else:
                    grad += torch.autograd.grad(loss, p, retain_graph=False)[0]
            if n == 0:
                raise ValueError("Loader is empty.")
            if self.reduction == "mean":
                total /= n
                grad /= n
            self._gradient = grad.detach()
            return torch.tensor(total, dtype=grad.dtype, device=grad.device)

        def gradient(self, param):
            if self._gradient is None:
                raise RuntimeError("Call 'evaluate' first.")
            return self._gradient

    class SecondOrderDifferentiableFunction:
        """Helper class to compute second-order derivatives (Hessian) of scalar functions."""

        def __init__(self, f):
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

        def evaluate(self, x):
            """Evaluate the function at x and store gradient graph."""
            if not torch.is_tensor(x):
                x = torch.tensor(x, requires_grad=True, dtype=torch.double)
            else:
                x = x.detach().clone().requires_grad_(True)

            self._x = x
            self._y = self.f(self._x)

            if self._y.dim() != 0:
                raise ValueError("Function output must be a scalar.")

            return self._y

        def gradient(self, retain=True):
            """Compute and return gradient of the function at stored x."""
            if self._y is None or self._x is None:
                raise RuntimeError("Call evaluate(x) before calling gradient().")

            (grad,) = torch.autograd.grad(
                self._y, self._x, create_graph=True, retain_graph=retain
            )
            self._grad = grad
            return grad.detach()

        def hessian(self):
            """Compute and return Hessian of the function at stored x."""
            if self._grad is None:
                raise RuntimeError("Call gradient() before calling hessian().")

            x = self._x
            grad = self._grad
            n = x.numel()
            hessian = torch.zeros((n, n), dtype=torch.double)

            for idx in range(n):
                (grad2,) = torch.autograd.grad(
                    grad[idx], x, retain_graph=True, allow_unused=True
                )
                if grad2 is None:
                    raise RuntimeError(
                        f"Second derivative for parameter {idx} is None."
                    )
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
            d = sqrt(sum(invrho * (x - y) ** 2, axis=1))
        return d

    # ..................................................

    def svd(A, full_matrices=True, hermitian=True):
        return torch.linalg.svd(A, full_matrices)

    def solve(A, B, overwrite_a=True, overwrite_b=True, assume_a="gen", sym_pos=False):
        return torch.linalg.solve(A, B)

    def cho_factor(A, lower=False, overwrite_a=False, check_finite=True):
        # torch.linalg does not have cho_factor(), use cholesky() instead.
        return cholesky(A, upper=not (lower))

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
                else torch.tensor(a, dtype=torch.float64)
            )

        n = a.shape[0]

        if p is not None:
            p = torch.tensor(p, dtype=torch.float64, device=a.device)
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
            d = Normal(loc, scale)
            return d.cdf(x)

        @staticmethod
        def logcdf(x, loc=0.0, scale=1.0):
            d = Normal(loc, scale)
            return log(d.cdf(x))

        @staticmethod
        def pdf(x, loc=0.0, scale=1.0):
            t = (x - loc) / scale
            return 1 / sqrt(2 * pi) * exp(-0.5 * t**2)

        @staticmethod
        def logpdf(x, loc=0.0, scale=1.0):
            d = Normal(loc, scale)
            return d.logpdf(x)

    class multivariate_normal:
        @staticmethod
        def rvs(mean=0.0, cov=1.0, n=1):
            # Check if cov is a scalar or 1x1 array, and use Normal if so
            if (
                torch.is_tensor(cov)
                and cov.ndim == 0
                or (cov.ndim == 2 and cov.shape[0] == 1 and cov.shape[1] == 1)
            ):
                distribution = Normal(torch.tensor(mean), cov.sqrt())
                return distribution.sample((n,))

            # For dxd covariance matrix, use MultivariateNormal
            d = cov.shape[0]  # Dimensionality from the covariance matrix
            mean_tensor = torch.full((d,), mean)  # Expand mean to a tensor
            distribution = MultivariateNormal(mean_tensor, covariance_matrix=cov)
            return distribution.sample((n,))

        @staticmethod
        def logpdf(x, mean=0.0, cov=1.0):
            # Check if cov is a scalar or 1x1 array, and use Normal if so
            if (
                torch.is_tensor(cov)
                and cov.ndim == 0
                or (cov.ndim == 2 and cov.shape[0] == 1 and cov.shape[1] == 1)
            ):
                distribution = Normal(torch.tensor(mean), cov.sqrt())
                return distribution.log_prob(x.squeeze(-1))

            # For dxd covariance matrix, use MultivariateNormal
            d = x.shape[-1]  # Infer dimensionality from x
            mean_tensor = torch.full((d,), mean)  # Expand mean to a tensor
            distribution = MultivariateNormal(mean_tensor, covariance_matrix=cov)
            return distribution.log_prob(x)

        @staticmethod
        def cdf(x, mean=0.0, cov=1.0):
            # Convert inputs to NumPy arrays if they are PyTorch tensors
            if torch.is_tensor(x):
                x = x.numpy()
            if torch.is_tensor(mean):
                mean = mean.numpy()
            if torch.is_tensor(cov):
                cov = cov.numpy()

            # Check if cov is a scalar or 1x1 array, and use norm for the univariate case
            if (
                isscalar(cov)
                or (isinstance(cov, numpy.ndarray) and cov.size == 1)
                or (torch.is_tensor(cov) and cov.size == 1)
            ):
                return Normal.cdf(x, mean, np.sqrt(cov))

            # For dxd covariance matrix, use multivariate_normal
            if isscalar(mean):
                d = cov.shape[0]  # Dimensionality from the covariance matrix
                mean = full(d, mean)  # Expand mean to an array
            return scipy_mvnormal.cdf(x, mean, cov)


# ------------------------------------------------------
#
#                        JAX
#
# ------------------------------------------------------
elif _gpmp_backend_ == "jax":
    import os
    import numpy

    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")  # stay on the GPU
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE", "false"
    )  # tiny RAM footprint

    import jax

    # set multithreaded/multicore parallelism
    # see https://github.com/google/jax/issues/8345
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=512"

    # set double precision for floats
    jax.config.update("jax_enable_x64", True)
    from jax.numpy import array, empty

    ndarray = jax.numpy.ndarray

    from jax.numpy import (
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
    from jax.numpy.linalg import norm
    from jax.numpy.linalg import cond, cholesky, qr, svd, inv, slogdet
    from jax.numpy import pi, inf
    from jax.numpy import finfo, float64
    from jax.scipy.special import gammaln
    from jax.scipy.linalg import solve, solve_triangular, cho_factor, cho_solve
    from jax.scipy.stats import norm as normal
    from scipy.stats import multivariate_normal as scipy_mvnormal

    # ..................................................

    eps = finfo(float64).eps
    fmax = finfo(float64).max
    # ..................................................

    def copy(x):
        return array(x, copy=True)

    def is_boolean_mask(mask):
        return isinstance(mask, jax.numpy.ndarray) and jax.numpy.issubdtype(
            mask.dtype, jax.numpy.bool_
        )

    def set_elem_1d(x, index, v):
        return x.at[index].set(v)

    def set_elem_2d(x, i, j, v):
        return x.at[i, j].set(v)

    def set_row_2d(A, index, x):
        if isinstance(index, (int, jax.numpy.integer)):
            return A.at[index, :].set(x)
        elif isinstance(index, jax.numpy.ndarray):
            if jax.numpy.issubdtype(index.dtype, jax.numpy.bool_):
                rows = jax.numpy.where(index)[0]
                return A.at[rows].set(x)
            elif jax.numpy.issubdtype(index.dtype, jax.numpy.integer):
                return A.at[index].set(x)
        raise TypeError(
            "Unsupported index type: must be int, integer array, or boolean mask"
        )

    def set_col_2d(A, index, x):
        if isinstance(index, (int, jax.numpy.integer)):
            return A.at[:, index].set(x)
        elif isinstance(index, jax.numpy.ndarray):
            if jax.numpy.issubdtype(index.dtype, jax.numpy.bool_):
                cols = jax.numpy.where(index)[0]
                return A.at[:, cols].set(x)
            elif jax.numpy.issubdtype(index.dtype, jax.numpy.integer):
                return A.at[:, index].set(x)
        raise TypeError(
            "Unsupported index type: must be int, integer array, or boolean mask"
        )

    def set_col_3d(A, index, x):
        return A.at[:, :, index].set(x)

    def index_select(x, dim, indices):
        if dim == 0:
            return jax.numpy.take(x, indices, axis=0)
        elif dim == 1:
            return jax.numpy.take(x, indices, axis=1)
        else:
            # General fallback for dim > 1
            x_moved = jax.numpy.moveaxis(x, dim, 0)
            x_selected = jax.numpy.take(x_moved, indices, axis=0)
            return jax.numpy.moveaxis(x_selected, 0, dim)

    # ..................................................

    def asarray(x, dtype=None):
        if isinstance(x, jax.numpy.ndarray):
            return x
        if isinstance(x, (int, float)):
            return jax.numpy.array([x])
        else:
            return jax.numpy.asarray(x, dtype=dtype)

    def asdouble(x):
        return x.astype(float64)

    def asint(x):
        return x.astype(int)

    def to_np(x):
        if isinstance(x, jax.numpy.ndarray):
            return numpy.array(x)
        else:
            return x

    def to_scalar(x):
        return x.item()

    def isarray(x):
        return isinstance(x, jax.numpy.ndarray)

    def inftobigf(a, bigf=fmax / 1000.0):
        a = where(isinf(a), full_like(a, bigf), a)
        return a

    # ..................................................

    class DifferentiableSelectionCriterion:
        def __init__(self, crit, x, z):
            self.crit = jax.jit(lambda p: crit(p, x, z))  # jax.jit(f)
            self.crit_grad = jax.jit(jax.grad(self.crit))  # jax.jit(jax.grad(self.f))

        def __call__(self, p):
            return self.evaluate_no_grad(p)

        def evaluate_no_grad(self, p):
            return self.evaluate(p)

        def evaluate(self, p):
            if not isinstance(p, jax.numpy.ndarray):
                p = jax.numpy.array(p)
            try:
                return self.crit(p)
            except Exception:
                return inf

        def gradient(self, p):
            if not isinstance(p, jax.numpy.ndarray):
                p = jax.numpy.array(p)
            try:
                return self.crit_grad(p)
            except Exception:
                return None

    class BatchDifferentiableSelectionCriterion:
        def __init__(self, f_single_batch):
            """
            Parameters
            ----------
            f_single_batch : callable
                A function f_single_batch(x, batch) -> scalar loss for a single batch.
                It must be JAX differentiable.
            """
            self.f_single_batch = f_single_batch  # single batch function
            self.f_grad_single_batch = jax.grad(
                self.f_single_batch
            )  # grad of the batch function
            self.f_value = None
            self.x_value = None
            self.loader_value = None

        def __call__(self, x, loader):
            return self.evaluate_no_grad(x, loader)

        def evaluate(self, x, loader):
            if not isinstance(x, jax.Array):
                x = jax.numpy.asarray(x)

            self.x_value = x
            self.loader_value = loader

            loss_acc = 0.0
            for batch in loader:
                loss_b = self.f_single_batch(x, batch)
                loss_acc += loss_b

            self.f_value = loss_acc / len(loader)
            return self.f_value

        def evaluate_no_grad(self, x, loader):
            if not isinstance(x, jax.Array):
                x = jax.numpy.asarray(x)

            loss_acc = 0.0
            for batch in loader:
                loss_b = self.f_single_batch(x, batch)
                loss_acc += loss_b

            return loss_acc / len(loader)

        def gradient(self, x):
            if self.f_value is None:
                raise ValueError("Call 'evaluate(x, loader)' before 'gradient(x)'")

            if not jax.numpy.array_equal(x, self.x_value):
                raise ValueError(
                    "The input 'x' in 'gradient' must be the same as in 'evaluate'"
                )

            grad_acc = jax.numpy.zeros_like(self.x_value)

            for batch in self.loader_value:
                grad_b = self.f_grad_single_batch(self.x_value, batch)
                grad_acc += grad_b

            grad_acc = grad_acc / len(self.loader_value)
            return grad_acc

    # ..................................................

    def cdist(x, y):
        if y is None:
            y = x
        y2 = sum(y**2, axis=1)
        # Debug: check if x is y
        # print("&x = {}, &y = {}".format(hex(id(x)), hex(id(y))))
        if x is y:
            d = sqrt(reshape(y2, [-1, 1]) + y2 - 2 * inner(x, y))
        else:
            x2 = reshape(sum(x**2, axis=1), [-1, 1])
            d = sqrt(x2 + y2 - 2 * inner(x, y))
        return d

    @jax.custom_vjp
    def scaled_distance_(loginvrho, x, y):
        invrho = exp(loginvrho)
        hs2 = (invrho * (x - y)) ** 2
        d = sqrt(sum(hs2))
        return d

    def scaled_distance__fwd(loginvrho, x, y):
        invrho = exp(loginvrho)
        hs2 = (invrho * (x - y)) ** 2
        d = sqrt(sum(hs2))
        intermediates = (d, hs2)
        return d, intermediates

    def scaled_distance__bwd(intermediates, g):
        d, hs2 = intermediates
        grad_loginvrho = g * hs2 / (d + eps)
        grad_x = None  # Not computed, since we only need the gradient with respect to loginvrho
        grad_y = None
        return (grad_loginvrho, grad_x, grad_y)

    scaled_distance_.defvjp(scaled_distance__fwd, scaled_distance__bwd)

    def scaled_distance(loginvrho, x, y):
        f = jax.vmap(
            lambda x1: jax.vmap(
                lambda y1: jax.jit(scaled_distance_)(loginvrho, x1, y1)
            )(y)
        )(x)
        return f

    def scaled_distance_elementwise(loginvrho, x, y):
        f = jax.vmap(jax.jit(scaled_distance), in_axes=(None, 0, 0), out_axes=0)(
            loginvrho, x, y
        )
        return f

    # ..................................................

    def logdet(A):
        sign, logabsdet = slogdet(A)
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

    # One global key:
    _jax_key = jax.random.PRNGKey(1234)

    def _update_key():
        global _jax_key
        _jax_key, subkey = jax.random.split(_jax_key)
        return subkey

    def set_seed(seed):
        """Set the global JAX key seed."""
        global _jax_key
        _jax_key = jax.random.PRNGKey(seed)

    def rand(*shape):
        subkey = _update_key()
        return jax.random.uniform(subkey, shape=shape)

    def randn(*shape):
        subkey = _update_key()
        return jax.random.normal(subkey, shape=shape)

    def choice(a, size=None, replace=True, p=None):
        subkey = _update_key()

        if size is None:
            size = 1

        if isinstance(a, int):
            a_ar = jax.numpy.arange(a)
        else:
            a_ar = jax.numpy.asarray(a)

        n = a_ar.shape[0]

        if p is not None:
            p_ar = jax.numpy.asarray(p, dtype=jax.numpy.float32)
            p_ar = p_ar / p_ar.sum()
            if replace:
                indices = jax.random.categorical(
                    subkey, jax.numpy.log(p_ar), shape=(size,)
                )
            else:
                raise NotImplementedError(
                    "choice w/o replacement + p not supported in JAX."
                )
        else:
            if replace:
                indices = jax.random.randint(subkey, shape=(size,), minval=0, maxval=n)
            else:
                perm = jax.random.permutation(subkey, n)
                if size == n:
                    indices = perm
                else:
                    indices = jax.lax.dynamic_slice(perm, (0,), (size,))

        return a_ar[indices]

    def permutation(x):
        subkey = _update_key()
        return jax.random.permutation(subkey, x)

    class multivariate_normal:
        @staticmethod
        def rvs(mean=0.0, cov=1.0, n=1):
            if isscalar(cov) or cov.ndim == 0:
                cov = cov * eye(1)
            if isscalar(mean):
                mean = full((cov.shape[0],), mean)

            subkey = _update_key()
            return jax.random.multivariate_normal(
                subkey, mean=mean, cov=cov, shape=(n,)
            )

        @staticmethod
        def logpdf(x, mean=0.0, cov=1.0, allow_singular=False):
            mean_vector = full((x.shape[-1],), mean)
            return jax.scipy.stats.multivariate_normal.logpdf(
                x, mean=mean_vector, cov=cov, allow_singular=allow_singular
            )

        @staticmethod
        def cdf(x, mean=0.0, cov=1.0):
            x = numpy.asarray(x)
            mean = numpy.asarray(mean)
            cov = numpy.asarray(cov)

            if isscalar(cov) or cov.size == 1:
                return normal.cdf(x, mean, numpy.sqrt(cov))

            if isscalar(mean):
                d = cov.shape[0]
                mean = full(d, mean)

            return scipy_mvnormal.cdf(x, mean, cov)


# ------------------------------------------------------------------
#
# No more backends
#
# ------------------------------------------------------------------

else:
    raise RuntimeError(
        "Please set the GPMP_BACKEND environment variable to 'numpy', 'torch' or 'jax'."
    )

# ------------------------------------------------------------------
#
# Backend independent functions
#
# ------------------------------------------------------------------


def derivative_finite_diff(f, x, h):
    """
    5-point central difference derivative of f w.r.t. scalar x.
    f(x) must return a NumPy (or similar) array/matrix/tensor.
    """
    f_x_p2 = f(x + 2 * h)
    f_x_p1 = f(x + h)
    f_x_m1 = f(x - h)
    f_x_m2 = f(x - 2 * h)
    return (-f_x_p2 + 8 * f_x_p1 - 8 * f_x_m1 + f_x_m2) / (12.0 * h)


def try_with_postmortem(func, *args, **kwargs):
    """
    Executes `func(*args, **kwargs)` with a try/except block.
    If an exception is raised, it prints the traceback and drops into pdb post-mortem.

    Parameters
    ----------
    func : callable
        The function to execute.
    *args, **kwargs
        Arguments passed to the function.
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        extype, value, tb = __import__("sys").exc_info()
        __import__("traceback").print_exc()
        __import__("pdb").post_mortem(tb)


# ----------------------------------------------------------------------
#                              TODO (roadmap)
# ----------------------------------------------------------------------
# * Maintain a state dictionary to hold: dtype, device, gpmp cache, gln... see also stk
# * Heal Jax slow perfomances
# * DifferentiableSelectionCriterion: avoid silent fail, do not catch
#   all Exceptions and return inf. (That hides bugs, typos, device errors...).
# * BatchDifferentiableSelectionCriterion: assume crit returns mean
# * BatchDifferentiableSelectionCriterion: dtype and device consistency
#    xb0, _ = next(iter(self.loader))
#    param = torch.tensor(param, dtype=xb0.dtype, device=xb0.device, requires_grad=True)
# * Fix dtypes and device not specified
# ----------------------------------------------------------------------
