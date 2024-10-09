"""
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2024, CentraleSupelec
License: GPLv3 (see LICENSE)
--------------------------------------------------------------

Description:
    This module sets and prints the backend to be used by the GPmp (Gaussian
    Process micro package) framework.

    The backend is set by checking for the existence of certain libraries in
    the system and, if available, sets an environment variable "GPMP_BACKEND" 
    to the corresponding library's name.

    The priority order for backends is: torch -> jax -> numpy. If neither 
    torch nor jax are found in the system, numpy is set as the default backend.

"""

import os
import numpy
from importlib import util as importlib_util

_gpmp_backend_ = os.environ.get("GPMP_BACKEND")


def set_backend_env_var(backend):
    global _gpmp_backend_
    os.environ["GPMP_BACKEND"] = backend
    _gpmp_backend_ = backend


# Automatically set the backend if not already set in the environment.
if _gpmp_backend_ is None:
    if importlib_util.find_spec("torch") is not None:
        set_backend_env_var("torch")
    elif importlib_util.find_spec("jax") is not None:
        set_backend_env_var("jax")
    else:
        set_backend_env_var("numpy")

print(f"Using backend: {_gpmp_backend_}")


# -----------------------------------------------------
#
#                      NUMPY
#
# -----------------------------------------------------
if _gpmp_backend_ == "numpy":
    from numpy import array, empty

    from numpy import (
        copy,
        reshape,
        where,
        any,
        isscalar,
        isnan,
        isinf,
        isfinite,
        unique,
        hstack,
        vstack,
        stack,
        tile,
        concatenate,
        expand_dims,
        empty,
        empty_like,
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
        abs,
        sqrt,
        exp,
        log,
        log10,
        sum,
        mean,
        std,
        cov,
        sort,
        min,
        max,
        argmin,
        argmax,
        minimum,
        maximum,
        einsum,
        matmul,
        inner,
        all,
        logical_not,
        logical_and,
        logical_or,
    )
    from numpy.linalg import norm, cond, cholesky, qr, svd, inv
    from numpy.random import rand, randn, choice
    from numpy import pi, inf
    from numpy import finfo, float64
    from scipy.special import gammaln
    from scipy.linalg import solve, solve_triangular, cho_factor, cho_solve
    from scipy.spatial.distance import cdist
    from scipy.stats import norm as normal
    from scipy.stats import multivariate_normal as scipy_mvnormal

    eps = finfo(float64).eps
    fmax = numpy.finfo(numpy.float64).max

    def set_elem1(x, index, v):
        x[index] = v
        return x

    def set_row2(A, index, x):
        A[index, :] = x
        return A

    def set_col2(A, index, x):
        A[:, index] = x
        return A

    def set_col3(A, index, x):
        A[:, :, index] = x
        return A

    def asarray(x):
        if isinstance(x, numpy.ndarray):
            return x
        elif isinstance(x, (int, float)):
            return numpy.array([x])
        else:
            return numpy.asarray(x)

    def asdouble(x):
        return x.astype(float64)

    def to_np(x):
        return x

    def to_scalar(x):
        return x.item()

    def isarray(x):
        return isinstance(x, numpy.ndarray)

    def inftobigf(a, bigf=fmax / 1000.0):
        a = where(numpy.isinf(a), numpy.full_like(a, bigf), a)
        return a

    def grad(f):
        return None

    class jax:
        @staticmethod
        def jit(f, *args, **kwargs):
            return f

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
            d = sqrt(sum(invrho * (xs - ys) ** 2, axis=1))
        return d

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

    from torch import (
        reshape,
        where,
        any,
        isnan,
        isinf,
        isfinite,
        hstack,
        vstack,
        stack,
        tile,
        concatenate,
        empty,
        empty_like,
        zeros,
        ones,
        full,
        eye,
        diag,
        arange,
        linspace,
        logspace,
        meshgrid,
        abs,
        cov,
        argmax,
        argmin,
        einsum,
        matmul,
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

    eps = finfo(float64).eps
    fmax = finfo(float64).max

    def copy(x):
        return torch.clone(tensor(x))

    def set_elem1(x, index, v):
        x[index] = v
        return x

    def set_row2(A, index, x):
        A[index, :] = x
        return A

    def set_col2(A, index, x):
        A[:, index] = x
        return A

    def set_col3(A, index, x):
        A[:, :, index] = x
        return A

    def expand_dims(tensor, axis):
        return tensor.unsqueeze(axis)

    def array(x: list):
        return tensor(x)

    def asarray(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, (int, float)):
            return tensor([x])
        else:
            return torch.asarray(x)

    def asdouble(x):
        return x.to(torch.double)

    def to_np(x):
        return x.numpy()

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

    log = scalar_safe(torch.log)
    log10 = scalar_safe(torch.log10)
    exp = scalar_safe(torch.exp)
    sqrt = scalar_safe(torch.sqrt)

    def axis_to_dim(f):
        def f_(x, axis=None, **kwargs):
            if axis is None:
                return f(x, **kwargs)
            else:
                return f(x, dim=axis, **kwargs)

        return f_

    all = axis_to_dim(torch.all)
    unique = axis_to_dim(torch.unique)
    sum = axis_to_dim(torch.sum)
    mean = axis_to_dim(torch.mean)
    std = axis_to_dim(torch.std)
    var = axis_to_dim(torch.var)

    def norm(x, axis=None, ord=2):
        return torch.norm(x, dim=axis, p=ord)

    def min(x, axis=0):
        m = torch.min(x, dim=axis)
        return m.values

    def max(x, axis=0):
        m = torch.max(x, dim=axis)
        return m.values

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

    def sort(x, axis=-1):
        xsorted = torch.sort(x, dim=axis)
        return xsorted.values

    def inftobigf(a, bigf=fmax / 1000.0):
        a = torch.where(torch.isinf(a), torch.full_like(a, bigf), a)
        return a

    def svd(A, full_matrices=True, hermitian=True):
        return torch.linalg.svd(A, full_matrices)

    def solve(A, B, overwrite_a=True, overwrite_b=True, assume_a="gen", sym_pos=False):
        return torch.linalg.solve(A, B)

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
            d = sqrt(sum(invrho * (xs - ys) ** 2, axis=1))
        return d

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
    import jax

    # set multithreaded/multicore parallelism
    # see https://github.com/google/jax/issues/8345
    # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=512"

    # set double precision for floats
    jax.config.update("jax_enable_x64", True)

    from jax.numpy import array, empty

    from jax.numpy import (
        reshape,
        where,
        any,
        isscalar,
        isnan,
        isinf,
        isfinite,
        unique,
        hstack,
        vstack,
        stack,
        tile,
        concatenate,
        expand_dims,
        empty,
        empty_like,
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
        abs,
        sqrt,
        exp,
        log,
        log10,
        sum,
        mean,
        std,
        cov,
        sort,
        min,
        max,
        argmin,
        argmax,
        minimum,
        maximum,
        einsum,
        matmul,
        inner,
        all,
        logical_not,
        logical_and,
        logical_or,
    )
    from jax.numpy.linalg import norm
    from jax.numpy.linalg import cond, cholesky, qr, svd, inv
    from jax.numpy import pi, inf
    from jax.numpy import finfo, float64
    from jax.scipy.special import gammaln
    from jax.scipy.linalg import solve, solve_triangular, cho_factor, cho_solve
    from jax.scipy.stats import norm as normal
    from scipy.stats import multivariate_normal as scipy_mvnormal

    eps = finfo(float64).eps
    fmax = finfo(float64).max

    def copy(x):
        return array(x, copy=True)

    @jax.jit
    def set_elem1(x, index, v):
        return x.at[index].set(v)

    @jax.jit
    def set_row2(A, index, x):
        return A.at[index, :].set(x)

    @jax.jit
    def set_col2(A, index, x):
        return A.at[:, index].set(x)

    @jax.jit
    def set_col3(A, index, x):
        return A.at[:, :, index].set(x)

    def asarray(x):
        if isinstance(x, jax.numpy.ndarray):
            return x
        if isinstance(x, (int, float)):
            return jax.numpy.array([x])
        else:
            return jax.numpy.asarray(x)

    def asdouble(x):
        return x.astype(float64)

    def to_np(x):
        return numpy.array(x)

    def to_scalar(x):
        return x.item()

    def isarray(x):
        return isinstance(x, jax.numpy.ndarray)

    def inftobigf(a, bigf=fmax / 1000.0):
        a = where(isinf(a), full_like(a, bigf), a)
        return a

    from jax import grad

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
        xs = invrho * x
        ys = invrho * y
        hs2 = (xs - ys) ** 2
        d = sqrt(sum(hs2))
        return d

    def scaled_distance__fwd(loginvrho, x, y):
        invrho = exp(loginvrho)
        xs = invrho * x
        ys = invrho * y
        hs2 = (xs - ys) ** 2
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

    def seed(s):
        return jax.random.PRNGKey(s)

    key = seed(0)

    def rand(*args):
        return jax.random.uniform(key, shape=args)

    def randn(*args):
        return jax.random.normal(key, shape=args)

    class multivariate_normal:
        @staticmethod
        def rvs(mean=0.0, cov=1.0, n=1):
            if isscalar(cov) or cov.ndim == 0:
                cov = cov * eye(1)
            if isscalar(mean):
                mean = array([mean])

            d = cov.shape[0]  # Dimensionality from the covariance matrix
            mean_vector = full((d,), mean)  # Expand mean to a vector

            return jax.random.multivariate_normal(
                key, mean=mean_vector, cov=cov, shape=(n,)
            )

        @staticmethod
        def logpdf(x, mean=0.0, cov=1.0, allow_singular=False):
            mean_vector = full(
                (x.shape[-1],), mean
            )  # Expand mean to match x dimensions
            return jax.scipy.stats.multivariate_normal.logpdf(
                x, mean=mean_vector, cov=cov, allow_singular=allow_singular
            )

        @staticmethod
        def cdf(x, mean=0.0, cov=1.0):
            # Convert JAX arrays to NumPy arrays for compatibility with SciPy
            if isinstance(x, jax.numpy.ndarray):
                x = numpy.array(x)
            if isinstance(mean, jax.numpy.ndarray):
                mean = numpy.array(mean)
            if isinstance(cov, jax.numpy.ndarray):
                cov = numpy.array(cov)

            # Check if cov is a scalar or 1x1 array, and use norm for the univariate case
            if isscalar(cov) or (isinstance(cov, jax.numpy.ndarray) and cov.size == 1):
                return normal.cdf(x, mean, sqrt(cov))

            # For dxd covariance matrix, use multivariate_normal
            if isscalar(mean):
                d = cov.shape[0]  # Dimensionality from the covariance matrix
                mean = full(d, mean)  # Expand mean to an array

            return scipy_mvnormal.cdf(x, mean, cov)


# ------------------------------------------------------------------

else:
    raise RuntimeError(
        "Please set the GPMP_BACKEND environment variable to 'numpy', 'torch' or 'jax'."
    )
