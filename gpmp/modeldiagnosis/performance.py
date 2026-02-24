# gpmp/modeldiagnosis/performance.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Predictive performance metrics for GP models.

This module defines:
- compute_performance: compute LOO and optional test-set metrics.
- perf: print a readable summary using gpmp.misc.dataframe.DataFrame.

Metrics
-------
For targets z:
- TSS = sum (z - mean(z))^2

For errors e:
- PRESS (LOO) = sum e_loo^2
- RSS (test)  = sum (z_test - mean_pred)^2

Derived:
- Q2 (LOO) = 1 - PRESS/TSS
- R2 (test) = 1 - RSS/TSS
- RMSE = sqrt(SSE / n)
- std(z)
- RMSE/std(z) = RMSE / std(z)
- log10(SSE/TSS)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import gpmp.num as gnp
from gpmp.misc.dataframe import DataFrame

# ---------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------


def _as_1d(x: Any) -> Any:
    return gnp.asarray(x).reshape(-1)


def compute_performance(
    model: Any,
    xi: Any,
    zi: Any,
    loo: bool = True,
    loo_res: Optional[Tuple[Any, Any, Any]] = None,
    xtzt: Optional[Tuple[Any, Any]] = None,
    zpmzpv: Optional[Tuple[Any, Any]] = None,
    compute_pit: bool = False,
) -> Dict[str, Any]:
    """
    Compute LOO and optional test-set performance metrics.

    Parameters
    ----------
    model : object
        Must provide:
        - loo(xi, zi) -> (zloom, zloov, eloo)
        - predict(xi, zi, xt) -> (zpm, zpv)
    xi : array-like
        Observation inputs, shape (n, d).
    zi : array-like
        Observation targets.
    loo : bool, optional
        If True, compute LOO metrics.
    loo_res : tuple, optional
        Precomputed (zloom, zloov, eloo).
    xtzt : tuple, optional
        (xt, zt) for test-set metrics.
    zpmzpv : tuple, optional
        Precomputed (zpm, zpv) for the test set.
    compute_pit : bool, optional
        If True, include PIT arrays when available.

    Returns
    -------
    dict
        LOO keys (if loo is True)
        - loo_n, loo_std, loo_tss, loo_press
        - loo_press_over_tss, loo_log10_press_over_tss
        - loo_rmse, loo_rmse_over_std, loo_Q2
        - loo_pit (optional)

        Test keys (if xtzt is not None)
        - test_n, test_std, test_tss, test_rss
        - test_rss_over_tss, test_log10_rss_over_tss
        - test_rmse, test_rmse_over_std, test_R2
        - test_pit (optional)
    """
    xi = gnp.asarray(xi)
    zi_arr = gnp.asarray(zi)
    zi_vec = _as_1d(zi_arr)

    out: Dict[str, Any] = {}

    if loo:
        if loo_res is None:
            zloom, zloov, eloo = model.loo(xi, zi_arr)
        else:
            zloom, zloov, eloo = loo_res

        eloo_vec = _as_1d(eloo)
        n = int(zi_vec.shape[0])

        tss = gnp.norm(zi_vec - gnp.mean(zi_vec), ord=2) ** 2
        press = gnp.norm(eloo_vec, ord=2) ** 2

        press_over_tss = press / tss
        rmse = gnp.sqrt(press / float(max(n, 1)))
        std = gnp.std(zi_vec)

        out["loo_n"] = n
        out["loo_std"] = std
        out["loo_tss"] = tss
        out["loo_press"] = press
        out["loo_press_over_tss"] = press_over_tss
        out["loo_log10_press_over_tss"] = gnp.log10(press_over_tss)
        out["loo_rmse"] = rmse
        out["loo_rmse_over_std"] = rmse / std
        out["loo_Q2"] = 1 - press / tss

        if compute_pit:
            zloov_arr = gnp.asarray(zloov)
            scale = gnp.sqrt(gnp.clip(zloov_arr, 0.0, gnp.inf))
            out["loo_pit"] = gnp.normal.cdf(zi_arr, loc=zloom, scale=scale)

    if xtzt is not None:
        xt, zt = xtzt
        xt = gnp.asarray(xt)
        zt_arr = gnp.asarray(zt)
        zt_vec = _as_1d(zt_arr)

        if zpmzpv is None:
            zpm, zpv = model.predict(xi, zi_arr, xt)
        else:
            zpm, zpv = zpmzpv
            zpm = gnp.asarray(zpm)
            zpv = gnp.asarray(zpv)

        zpm_vec = _as_1d(zpm)
        n = int(zt_vec.shape[0])

        tss = gnp.norm(zt_vec - gnp.mean(zt_vec), ord=2) ** 2
        rss = gnp.norm(zt_vec - zpm_vec, ord=2) ** 2

        rss_over_tss = rss / tss
        rmse = gnp.sqrt(rss / float(max(n, 1)))
        std = gnp.std(zt_vec)

        out["test_n"] = n
        out["test_std"] = std
        out["test_tss"] = tss
        out["test_rss"] = rss
        out["test_rss_over_tss"] = rss_over_tss
        out["test_log10_rss_over_tss"] = gnp.log10(rss_over_tss)
        out["test_rmse"] = rmse
        out["test_rmse_over_std"] = rmse / std
        out["test_R2"] = 1 - rss / tss

        if compute_pit:
            scale = gnp.sqrt(gnp.clip(zpv, 0.0, gnp.inf))
            out["test_pit"] = gnp.normal.cdf(zt_arr, loc=zpm, scale=scale)

    return out


# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------


def _section_dataframe(title: str, n: int, rows: Dict[str, Any]) -> None:
    rownames = list(rows.keys())
    data = np.asarray([rows[k] for k in rownames], dtype=object).reshape(-1, 1)
    df = DataFrame(data, ["value"], rownames)
    print(f"{title} (n={n:d})")
    print(df)


def perf(
    model: Any,
    xi: Any,
    zi: Any,
    loo: bool = True,
    loo_res: Optional[Tuple[Any, Any, Any]] = None,
    xtzt: Optional[Tuple[Any, Any]] = None,
    zpmzpv: Optional[Tuple[Any, Any]] = None,
) -> None:
    """
    Print compute_performance() results (PIT omitted) as DataFrames.

    Parameters
    ----------
    model, xi, zi : object, array-like
        See compute_performance.
    loo : bool, optional
        If True, include LOO metrics.
    loo_res : tuple, optional
        Precomputed (zloom, zloov, eloo).
    xtzt : tuple, optional
        (xt, zt) for test-set metrics.
    zpmzpv : tuple, optional
        Precomputed (zpm, zpv) for the test set.
    """
    p = compute_performance(
        model,
        xi,
        zi,
        loo=loo,
        loo_res=loo_res,
        xtzt=xtzt,
        zpmzpv=zpmzpv,
        compute_pit=False,
    )

    print("[Prediction performances]")

    if loo and "loo_press" in p:
        n = int(p["loo_n"])
        rows = {
            "std(z)": p["loo_std"],
            "tss": p["loo_tss"],
            "press": p["loo_press"],
            "press/tss": p["loo_press_over_tss"],
            "log10(press/tss)": p["loo_log10_press_over_tss"],
            "rmse": p["loo_rmse"],
            "rmse/std(z)": p["loo_rmse_over_std"],
            "Q2": p["loo_Q2"],
        }
        _section_dataframe("  LOO", n, rows)

    if xtzt is not None and "test_rss" in p:
        n = int(p["test_n"])
        rows = {
            "std(z)": p["test_std"],
            "tss": p["test_tss"],
            "rss": p["test_rss"],
            "rss/tss": p["test_rss_over_tss"],
            "log10(rss/tss)": p["test_log10_rss_over_tss"],
            "rmse": p["test_rmse"],
            "rmse/std(z)": p["test_rmse_over_std"],
            "R2": p["test_R2"],
        }
        _section_dataframe("  Test", n, rows)


__all__ = ["compute_performance", "perf"]
