"""
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)
--------------------------------------------------------------

Description:
    This module sets and prints the backend to be used by the GPMP (Generic
    Python Matrix Processing) framework.

    The backend is set by checking for the existence of certain libraries in
    the system and, if available, sets an environment variable "GPMP_BACKEND" 
    to the corresponding library's name.

    The priority order for backends is: torch -> jax -> numpy. If neither 
    torch nor jax are found in the system, numpy is set as the default backend.

"""
import os
import importlib
import numpy


_gpmp_backend_ = os.environ.get("GPMP_BACKEND")


def set_backend_env_var(backend):
    global _gpmp_backend_
    os.environ["GPMP_BACKEND"] = backend
    _gpmp_backend_ = backend

# Automatically set the backend if not already set in the environment.
if _gpmp_backend_ is None:
    if importlib.util.find_spec("torch") is not None:
        set_backend_env_var("torch")
    elif importlib.util.find_spec("jax") is not None:
        set_backend_env_var("jax")
    else:
        set_backend_env_var("numpy")

print(f"Using backend: {_gpmp_backend_}")


# -----------------------------------------------------
#
#                      NUMPY
#
# -----------------------------------------------------
if _gpmp_backend_ == 'numpy':
    from numpy import array, empty

    from numpy import (
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
        logical_not,
        logical_and,
        logical_or
    )
    from numpy.linalg import norm, cond, qr, svd, inv
    from numpy.random import randn, choice
    from numpy import pi, inf
    from numpy import finfo, float64
    from scipy.special import gammaln
    from scipy.linalg import solve, cholesky, cho_factor, cho_solve
    from scipy.spatial.distance import cdist
    from scipy.stats import norm as normal
    from scipy.stats import multivariate_normal

    eps = finfo(float64).eps
    fmax = numpy.finfo(numpy.float64).max

    
    def asarray(x):
        if isinstance(x, numpy.ndarray):
            return x
        elif isinstance(x, (int, float)):
            return numpy.array([x])
        else:
            return numpy.asarray(x)

    def to_np(x):
        return x

    def to_scalar(x):
        return x.item()

    def isarray(x):
        return isinstance(x, numpy.ndarray)

    def inftobigf(a, bigf=fmax/1000.):
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
        C, lower = cho_factor(A)
        return cho_solve((C, lower), b), C


# -----------------------------------------------------
#
#                      TORCH
#
# -----------------------------------------------------
elif _gpmp_backend_ == 'torch':
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
        unique,
        hstack,
        vstack,
        stack,
        tile,
        concatenate,
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
        argmax,
        argmin,
        einsum,
        matmul,
        inner,
        logical_not,
        logical_and,
        logical_or
    )
    from torch.linalg import cond, qr, inv
    from torch import randn
    from torch import cdist
    from torch import pi, inf
    from torch import finfo, float64
    from scipy.special import gammaln
    
    eps = finfo(float64).eps
    fmax = finfo(float64).max

    def array(x: list):
        return tensor(x)

    def asarray(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, (int, float)):
            return tensor([x])
        else:
            return torch.asarray(x)

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

    def inftobigf(a, bigf=fmax/1000.):
        a = torch.where(torch.isinf(a), torch.full_like(a, bigf), a)
        return a

    class normal:
        @staticmethod
        def cdf(x, loc=0.0, scale=1.0):
            d = torch.distributions.normal.Normal(loc, scale)
            return d.cdf(x)

        @staticmethod
        def pdf(x, loc=0.0, scale=1.0):
            t = (x - loc) / scale
            return 1 / sqrt(2*pi) * exp(-0.5 * t**2)

    from scipy.stats import multivariate_normal
        
    def cholesky(A, lower=True, overwrite_a=True):
        return torch.linalg.cholesky(A, upper=not (lower))

    def svd(A, full_matrices=True, hermitian=True):
        return torch.linalg.svd(A, full_matrices)

    def solve(A, B, overwrite_a=True, overwrite_b=True, assume_a='gen', sym_pos=False):
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

    def cholesky_solve(A, b):
        C = torch.linalg.cholesky(A)
        return torch.cholesky_solve(b.reshape(-1, 1), C, upper=False), C

    def cholesky_inv(A):
        C = torch.linalg.cholesky(A)
        return torch.cholesky_inverse(C)


# ------------------------------------------------------
#
#                        JAX
#
# ------------------------------------------------------
elif _gpmp_backend_ == 'jax':
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
        logical_not,
        logical_and,
        logical_or
    )
    from jax.numpy.linalg import norm
    from jax.numpy.linalg import cond, qr, svd, inv
    from jax.numpy import pi, inf
    from jax.numpy import finfo, float64
    from jax.scipy.special import gammaln
    from jax.scipy.linalg import solve, cholesky, cho_factor, cho_solve
    from jax.scipy.stats import norm as normal
    from scipy.stats import multivariate_normal
    
    eps = finfo(float64).eps
    fmax = finfo(float64).max

    def asarray(x):
        if isinstance(x, jax.numpy.ndarray):
            return x
        if isinstance(x, (int, float)):
            return jax.numpy.array([x])
        else:
            return jax.numpy.asarray(x)

    def to_np(x):
        return numpy.array(x)

    def to_scalar(x):
        return x.item()

    def isarray(x):
        return isinstance(x, jax.numpy.ndarray)

    def inftobigf(a, bigf=fmax/1000.):
        a = where(isinf(a), full_like(a, bigf), a)
        return a
    
    def seed(s):
        return jax.random.PRNGKey(s)

    key = seed(0)

    def randn(*args):
        return jax.random.normal(key, shape=args)
    
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
        hs2 = (xs - ys)**2
        d = sqrt(sum(hs2))
        return d

    def scaled_distance__fwd(loginvrho, x, y):
        invrho = exp(loginvrho)
        xs = invrho * x
        ys = invrho * y
        hs2 = (xs - ys)**2
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
        f = jax.vmap(lambda x1: jax.vmap(lambda y1: jax.jit(scaled_distance_)(loginvrho, x1, y1))(y))(x)
        return f

    def scaled_distance_elementwise(loginvrho, x, y):
        f = jax.vmap(jax.jit(scaled_distance), in_axes=(None, 0, 0), out_axes=0)(loginvrho, x, y)
        return f

    def cholesky_inv(A):
        # FIXME: slow!
        # n = A.shape[0]
        # C, lower = cho_factor(A)
        # Ainv = cho_solve((C, lower), eye(n))
        return inv(A)

    def cholesky_solve(A, b):
        C, lower = cho_factor(A)
        return cho_solve((C, lower), b), C


# ------------------------------------------------------------------

else:
    raise RuntimeError("Please set the GPMP_BACKEND environment variable to 'numpy', 'torch' or 'jax'.")
