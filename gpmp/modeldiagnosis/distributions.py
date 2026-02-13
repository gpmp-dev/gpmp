# gpmp/modeldiagnosis/distributions.py
"""Scalar distributions for diagnostic computations.

This module defines small helpers to build one-dimensional distributions from
unnormalized scalar log-densities. It is intended for model diagnosis and relies
on SciPy numerical integration and root finding.

Defined objects
---------------
Unnormalized1DDistribution
    One-dimensional distribution defined by an unnormalized scalar log-pdf.

Notes
-----
The implementation is scalar-oriented and uses ``scipy.integrate.quad`` and
``scipy.optimize.brentq``. It is therefore CPU-only. Quantiles require finite
bounds.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

from scipy.integrate import quad
from scipy.optimize import brentq

import gpmp.num as gnp

LogPDF = Callable[[float], float]
Bounds = Tuple[float, float]


class Unnormalized1DDistribution:
    """
    One-dimensional distribution defined by an unnormalized scalar log-pdf.

    Parameters
    ----------
    log_pdf : callable
        Function ``log_pdf(x: float) -> float``.
    bounds : tuple of float
        Integration bounds ``(a, b)`` with ``a < b``. May be infinite for
        normalization/integration.
    quad_opts : dict, optional
        Extra keyword arguments passed to ``scipy.integrate.quad``.

    Attributes
    ----------
    log_pdf : callable
        Stored log-pdf callable.
    bounds : tuple of float
        Stored bounds ``(a, b)``.
    Z : float
        Normalization constant.

    Notes
    -----
    Quantiles require finite bounds.
    """

    def __init__(
        self,
        log_pdf: LogPDF,
        bounds: Bounds,
        *,
        quad_opts: Optional[dict] = None,
    ):
        a, b = bounds
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            raise TypeError("bounds must be a pair of numbers (a, b).")
        if not (a < b):
            raise ValueError("bounds must satisfy a < b.")

        self.log_pdf = log_pdf
        self.bounds = (float(a), float(b))
        self._quad_opts = {} if quad_opts is None else dict(quad_opts)

        self.Z, _ = quad(
            self._f_scalar, self.bounds[0], self.bounds[1], **self._quad_opts
        )
        if not math.isfinite(self.Z) or self.Z <= 0.0:
            raise ValueError("Normalization failed (Z is not positive and finite).")

    @staticmethod
    def _safe_exp(u: float) -> float:
        if u > 709.0:
            return float("inf")
        if u < -745.0:
            return 0.0
        return math.exp(u)

    def _f_scalar(self, x: float) -> float:
        lp = float(self.log_pdf(float(x)))
        return self._safe_exp(lp)

    def f(self, x: Sequence[float]) -> gnp.ndarray:
        """
        Evaluate the unnormalized density on a 1D grid.

        Parameters
        ----------
        x : sequence of float
            Evaluation points.

        Returns
        -------
        gpmp.num.ndarray
            Unnormalized density values.
        """
        return gnp.asarray([self._f_scalar(float(t)) for t in x])

    def pdf(self, x: Sequence[float]) -> gnp.ndarray:
        """
        Evaluate the normalized density on a 1D grid.

        Parameters
        ----------
        x : sequence of float
            Evaluation points.

        Returns
        -------
        gpmp.num.ndarray
            Density values.
        """
        return self.f(x) / self.Z

    def cdf(self, x: float) -> float:
        """
        Evaluate the CDF at a scalar point.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            CDF value.
        """
        a, b = self.bounds
        x = float(x)
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        integral, _ = quad(self._f_scalar, a, x, **self._quad_opts)
        return float(integral / self.Z)

    def mean(self) -> float:
        """
        Compute the mean.

        Returns
        -------
        float
            Mean.
        """
        a, b = self.bounds
        integrand = lambda t: float(t) * self._f_scalar(float(t))
        mu, _ = quad(integrand, a, b, **self._quad_opts)
        return float(mu / self.Z)

    def var(self) -> float:
        """
        Compute the variance.

        Returns
        -------
        float
            Variance.
        """
        a, b = self.bounds
        mu = self.mean()
        integrand = lambda t: (float(t) ** 2) * self._f_scalar(float(t))
        m2, _ = quad(integrand, a, b, **self._quad_opts)
        return float(m2 / self.Z - mu * mu)

    def quantile(self, p: float, *, xtol: float = 1e-6) -> float:
        """
        Compute the quantile at level ``p``.

        Parameters
        ----------
        p : float
            Probability level in ``(0, 1)``.
        xtol : float, optional
            Absolute tolerance for ``brentq``.

        Returns
        -------
        float
            Quantile value.

        Raises
        ------
        ValueError
            If ``p`` is outside ``(0, 1)`` or bounds are not finite.
        """
        p = float(p)
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0, 1).")

        a, b = self.bounds
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("quantile requires finite bounds.")

        return float(brentq(lambda t: self.cdf(float(t)) - p, a, b, xtol=float(xtol)))


__all__ = ["Unnormalized1DDistribution"]
