# gpmp/modeldiagnosis/report.py
"""
Model diagnosis report assembly and display.

Defines
-------
modeldiagnosis_init
    Build a diagnosis dictionary from a model and an optimization info object.
    Optionally reconstruct a Param object and project optimizer bounds onto it.
model_diagnosis_disp
    Print a compact report: parameter selection summary, Param table, and basic
    data description (xi and zi).
diag
    Convenience wrapper: build the report then display it.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2026, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import gpmp.num as gnp
from gpmp.misc.param import (
    param_from_covparam_anisotropic,
    param_from_covparam_anisotropic_noisy,
)

from .utils import describe_array, pretty_print_dictionnary


def modeldiagnosis_init(
    model: Any,
    info: Any,
    *,
    model_type: str = "linear_mean_matern_anisotropic",
    param_obj: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build a diagnosis dictionary from a model and selection/optimization info.

    Parameters
    ----------
    model : object
        Must expose attributes:
        - covparam
        - meanparam (optional)
    info : object
        Must expose attributes:
        - success
        - best_value_returned
        - nfev
        - total_time
        - selection_criterion(params)   (for initial_val)
        - initial_params
        - fun
        May expose:
        - bounds : array-like, shape (n_total_params, 2)
          Bounds in the optimizer parameter space, ordered as [meanparam, covparam].
    model_type : str, optional
        Used only when param_obj is not provided. Supported values:
        - "linear_mean_matern_anisotropic"
        - "linear_mean_matern_anisotropic_noisy"
    param_obj : Param, optional
        If provided, this Param is used directly. If info.bounds is present, bounds
        are still projected onto the covariance part of this Param when possible.

    Returns
    -------
    md : dict
        Keys:
        - "optim_info" : info
        - "param_selection" : dict
        - "parameters" : dict (param_obj.to_simple_dict())
        - "param_obj" : Param
        - "loo" : dict (reserved)
        - "data" : dict (reserved)
    """
    md: Dict[str, Any] = {
        "optim_info": info,
        "param_selection": {},
        "parameters": {},
        "param_obj": None,
        "loo": {},
        "data": {},
    }

    md["param_selection"] = {
        "cvg_reached": info.success,
        "optimal_val": info.best_value_returned,
        "n_evals": info.nfev,
        "time": info.total_time,
        "initial_val": float(info.selection_criterion(info.initial_params)),
        "final_val": info.fun,
    }

    def _apply_cov_bounds_to_param(pobj: Any, cov_bounds: Any) -> Any:
        """
        Project bounds onto the covariance part of a Param.

        Parameters
        ----------
        pobj : Param
            Expected to have .paths and .bounds aligned with .to_simple_dict().
            The covariance entries are identified by path[0] == "covparam".
        cov_bounds : array-like, shape (k, 2)
            Bounds for covparam entries in optimizer space. Entries with both sides
            infinite are stored as None.

        Returns
        -------
        pobj : Param
            The same object, possibly modified in-place.
        """
        cov_bounds = np.asarray(cov_bounds, dtype=float)

        cov_inds = [
            j for j, pth in enumerate(pobj.paths) if pth and pth[0] == "covparam"
        ]
        if len(cov_inds) != cov_bounds.shape[0]:
            return pobj

        for dst_idx, (lo, hi) in zip(cov_inds, cov_bounds):
            if np.isinf(lo) and np.isinf(hi):
                pobj.bounds[dst_idx] = None
            else:
                pobj.bounds[dst_idx] = (float(lo), float(hi))
        return pobj

    if param_obj is None:
        covparam = gnp.asarray(model.covparam)
        builders = {
            "linear_mean_matern_anisotropic": param_from_covparam_anisotropic,
            "linear_mean_matern_anisotropic_noisy": param_from_covparam_anisotropic_noisy,
        }
        builder = builders.get(model_type)
        if builder is None:
            raise ValueError(f"Unknown model type: {model_type}")
        param_obj = builder(covparam, None, None, name_prefix="")

    bounds_arr = getattr(info, "bounds", None)
    if bounds_arr is not None:
        if getattr(model, "meanparam", None) is None:
            mpl = 0
        else:
            mpl = int(gnp.asarray(model.meanparam).reshape(-1).shape[0])

        cov_len = int(gnp.asarray(model.covparam).reshape(-1).shape[0])

        bounds_arr = gnp.asarray(bounds_arr)
        if (
            bounds_arr.ndim == 2
            and bounds_arr.shape[1] == 2
            and bounds_arr.shape[0] >= mpl + cov_len
        ):
            cov_bounds = bounds_arr[mpl : mpl + cov_len]
            param_obj = _apply_cov_bounds_to_param(param_obj, cov_bounds)

    md["parameters"] = param_obj.to_simple_dict()
    md["param_obj"] = param_obj
    return md


def model_diagnosis_disp(
    md: Dict[str, Any], xi: Any, zi: Any, *, model_type: str = "linear_mean_matern_anisotropic"
) -> None:
    """
    Print a diagnosis report.

    Parameters
    ----------
    md : dict
        Output of modeldiagnosis_init. Must contain "param_obj" and "parameters".
    xi : array-like
        Inputs, shape (n, d).
    zi : array-like
        Targets, shape (n,) or (n, p).
    model_type : str, optional
        Unused. Kept for backward compatibility with the former monolithic module.
    """
    _ = model_type  # intentionally unused

    xi = gnp.asarray(xi)
    zi = gnp.asarray(zi)

    print("[Model diagnosis]")
    print("  * Parameter selection")
    pretty_print_dictionnary(md["param_selection"])

    print("  * Parameters")
    print("\n".join("    " + line for line in str(md["param_obj"]).splitlines()))

    print("  * Data")
    print("    {:>0}: {:d}".format("count", int(zi.shape[0])))
    print("    -----")

    param_values = np.array(list(md["parameters"].values()), dtype=float)

    if getattr(zi, "ndim", 1) == 1:
        rownames_zi = ["zi"]
    else:
        rownames_zi = [f"zi_{j}" for j in range(int(zi.shape[1]))]
    df_zi = describe_array(zi, rownames_zi, 1.0 / param_values[0])

    n, d = int(xi.shape[0]), int(xi.shape[1])
    rownames_xi = [f"xi_{j}" for j in range(d)]
    df_xi = describe_array(xi, rownames_xi, 1.0 / param_values[-d:])

    print(df_zi.concat(df_xi))


def diag(
    model: Any,
    info_select_parameters: Any,
    xi: Any,
    zi: Any,
    *,
    model_type: str = "linear_mean_matern_anisotropic",
    param_obj: Optional[Any] = None,
) -> None:
    """
    Build and display a model diagnosis report.

    Parameters
    ----------
    model : object
        GP model.
    info_select_parameters : object
        Selection/optimization info passed to modeldiagnosis_init.
    xi, zi : array-like
        Training data.
    model_type : str, optional
        Passed to modeldiagnosis_init when param_obj is not provided.
    param_obj : Param, optional
        If provided, use this Param directly.
    """
    md = modeldiagnosis_init(
        model,
        info_select_parameters,
        model_type=model_type,
        param_obj=param_obj,
    )
    model_diagnosis_disp(md, xi, zi, model_type=model_type)


__all__ = ["modeldiagnosis_init", "model_diagnosis_disp", "diag"]
