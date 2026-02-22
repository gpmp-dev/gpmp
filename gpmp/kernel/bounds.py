# gpmp/kernel/bounds.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Empirical parameter bounds for GP kernel hyperparameters.
"""
import gpmp.num as gnp

def _minimum_nonzero_gap_distance_1d(xj):
    """Smallest positive spacing among points in 1D (inf if none)."""
    xj = xj.reshape(-1)
    if xj.shape[0] < 2:
        return gnp.inf
    xs = gnp.sort(xj)
    diffs = gnp.diff(xs)
    diffs = diffs[diffs > 0.0]
    return gnp.min(diffs) if diffs.shape[0] > 0 else gnp.inf

def empirical_bounds_factory(
    xi, zi, *, mean_paramlength=0, var_lower_factor=2.0, var_upper_factor=10.0, length_lower_factor=2.0
):
    """Build bounds for params [mean..., log(sigma2), -log(rho_1), ..., -log(rho_d)]."""
    xi = gnp.asarray(xi)
    zi = gnp.asarray(zi).reshape(-1)
    _n, d = xi.shape
    neg_inf, pos_inf = -gnp.inf, gnp.inf
    bounds = []
    for _ in range(int(mean_paramlength)):
        bounds.append((neg_inf, pos_inf))
    emp_var = gnp.var(zi)
    bounds.append((gnp.log(var_lower_factor * emp_var), gnp.log(var_upper_factor * emp_var)))
    for j in range(d):
        min_gap = _minimum_nonzero_gap_distance_1d(xi[:, j])
        if gnp.isfinite(min_gap):
            rho_lower = length_lower_factor * min_gap
            bounds.append((neg_inf, -gnp.log(rho_lower)))
        else:
            bounds.append((neg_inf, pos_inf))
    return gnp.asarray(bounds, dtype=float)
