# gpmp/misc/param.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Param: structured parameter management

This class provides a flexible and extensible structure to manage
parameters in hierarchical Gaussian process models or other statistical
models.

Features:
- Parameters can be grouped hierarchically using a tree-like path structure
- Each parameter has an associated normalization type (log, log_inv, none)
- Full support for indexing, slicing, concatenation
- Bounds can be attached to parameters for diagnostics or constraint handling
- get_by_name and get_by_path support view-based access (modifies internal state)

Note: Bounds are currently informative only. Future versions may allow:
- Automatic clipping to bounds when setting values
- Hard error if a value outside bounds is set
- Selectable enforcement mode per parameter
"""

from enum import Enum
from typing import List, Union, Optional, Dict, Any, Tuple
import gpmp.num as gnp
from gpmp.misc.dataframe import ftos


class Normalization(Enum):
    LOG = "log"
    LOG_INV = "log_inv"
    NONE = "none"


class Param:
    def __init__(
        self,
        values: Optional[Union[List[float], gnp.ndarray]] = None,
        paths: Optional[List[List[str]]] = None,
        normalizations: Optional[List[Union[str, Normalization]]] = None,
        names: Optional[List[str]] = None,
        bounds: Optional[List[Optional[Tuple[float, float]]]] = None,
        name_prefix: str = "param_",
        dim: Optional[int] = None,
    ):
        if values is None and dim is None:
            self.values = gnp.zeros(0)
        elif values is None and dim is not None:
            self.values = gnp.zeros(dim)
        else:
            self.values = gnp.asarray(values)

        self.dim: int = len(self.values)
        self.paths: List[List[str]] = (
            paths if paths is not None else [["param"] for _ in range(self.dim)]
        )
        self.names: List[str] = (
            names
            if names is not None
            else [f"{name_prefix}{i}" for i in range(self.dim)]
        )
        self.normalizations: List[Normalization] = self._parse_normalizations(
            normalizations
        )
        self.bounds: List[Optional[Tuple[float, float]]] = (
            bounds if bounds is not None else [None] * self.dim
        )

        self._check_consistency()

    def _parse_normalizations(
        self, normalizations: Optional[List[Union[str, Normalization]]]
    ) -> List[Normalization]:
        if normalizations is None:
            return [Normalization.NONE] * self.dim
        parsed = []
        for norm in normalizations:
            if isinstance(norm, Normalization):
                parsed.append(norm)
            elif isinstance(norm, str):
                norm = norm.lower()
                if norm == "log":
                    parsed.append(Normalization.LOG)
                elif norm == "log_inv":
                    parsed.append(Normalization.LOG_INV)
                elif norm in ("none", "None"):
                    parsed.append(Normalization.NONE)
                else:
                    raise ValueError(f"Unknown normalization: {norm}")
            else:
                raise TypeError("Normalization must be a str or Normalization enum.")
        return parsed

    def _check_consistency(self) -> None:
        if not (
            len(self.paths)
            == len(self.names)
            == len(self.normalizations)
            == len(self.bounds)
            == self.dim
        ):
            raise ValueError(
                "All parameter fields must have the same length as the number of parameters."
            )

    @property
    def values(self) -> gnp.ndarray:
        return self._values

    @values.setter
    def values(self, new_values: Union[List[float], gnp.ndarray]) -> None:
        self._values = gnp.asarray(new_values)
        self.dim = len(self._values)

    @property
    def denormalized_values(self) -> gnp.ndarray:
        return gnp.array(
            [
                self._denormalize(v, norm)
                for v, norm in zip(self._values, self.normalizations)
            ]
        )

    @denormalized_values.setter
    def denormalized_values(self, new_values: Union[List[float], gnp.ndarray]) -> None:
        new_values = gnp.asarray(new_values, dtype=float)
        if len(new_values) != self.dim:
            raise ValueError("Mismatch in size for denormalized values.")
        self._values = gnp.array(
            [
                self._normalize(v, norm)
                for v, norm in zip(new_values, self.normalizations)
            ]
        )

    def _normalize(self, value: float, normalization: Normalization) -> float:
        if normalization == Normalization.LOG:
            return gnp.log(value)
        elif normalization == Normalization.LOG_INV:
            return -gnp.log(value)
        return value

    def _denormalize(self, value: float, normalization: Normalization) -> float:
        if normalization == Normalization.LOG:
            return gnp.exp(value)
        elif normalization == Normalization.LOG_INV:
            return gnp.exp(-value)
        return value

    def get_paths(self, prefix: Optional[List[str]] = None) -> List[List[str]]:
        """Return all unique paths or paths matching a given prefix."""
        if prefix is None:
            return list({tuple(p) for p in self.paths})
        return [p for p in self.paths if p[: len(prefix)] == prefix]

    def select_by_path_prefix(
        self, prefix: List[str], return_view: bool = False
    ) -> gnp.ndarray:
        """Return parameter values for all paths matching the prefix."""
        return self.get_by_path(prefix, prefix_match=True, return_view=return_view)

    def indices_by_path_prefix(self, prefix: List[str]) -> List[int]:
        """Return indices of parameters whose path matches the prefix."""
        return [i for i, p in enumerate(self.paths) if p[: len(prefix)] == prefix]

    def names_by_path_prefix(self, prefix: List[str]) -> List[str]:
        """Return parameter names whose path matches the prefix."""
        return [self.names[i] for i in self.indices_by_path_prefix(prefix)]

    def get_by_name(
        self, name: str, return_view: bool = False
    ) -> Union[float, gnp.ndarray]:
        idx = self.names.index(name)
        return self._values[idx : idx + 1] if return_view else self._values[idx]

    def set_by_name(self, name: str, new_value: float) -> None:
        idx = self.names.index(name)
        self._values[idx] = new_value

    def get_by_path(
        self, path: List[str], prefix_match: bool = False, return_view: bool = False
    ) -> gnp.ndarray:
        if prefix_match:
            indices = [i for i, p in enumerate(self.paths) if p[: len(path)] == path]
        else:
            indices = [i for i, p in enumerate(self.paths) if p == path]
        indices_array = gnp.asarray(indices, dtype=int)

        if return_view:
            if not gnp.all(gnp.diff(indices_array) == 1):
                raise ValueError(
                    "Requested path does not map to a contiguous block — cannot return view."
                )
            return self._values[indices_array[0] : indices_array[-1] + 1]
        else:
            return gnp.copy(self._values[indices_array])

    def set_by_path(
        self,
        path: List[str],
        new_values: Union[List[float], gnp.ndarray],
        prefix_match: bool = False,
    ) -> None:
        if prefix_match:
            indices = [i for i, p in enumerate(self.paths) if p[: len(path)] == path]
        else:
            indices = [i for i, p in enumerate(self.paths) if p == path]
        if len(indices) != len(new_values):
            raise ValueError(f"Expected {len(indices)} values, got {len(new_values)}.")
        for idx, val in zip(indices, new_values):
            self._values[idx] = val

    def set_from_unnormalized(self, **kwargs: float) -> None:
        for name, val in kwargs.items():
            idx = self.names.index(name)
            self._values[idx] = self._normalize(val, self.normalizations[idx])
            
    def check_bounds(self) -> List[bool]:
        return [
            True if b is None else (b[0] <= v <= b[1])
            for v, b in zip(self.denormalized_values, self.bounds)
        ]

    def __getitem__(self, index: Union[int, slice]) -> "Param":
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            index = list(range(self.dim))[index]
        return Param(
            values=self._values[index],
            paths=[self.paths[i] for i in index],
            normalizations=[self.normalizations[i] for i in index],
            names=[self.names[i] for i in index],
            bounds=[self.bounds[i] for i in index],
        )

    def __add__(self, other: "Param") -> "Param":
        return Param.concat(self, other)

    @staticmethod
    def concat(*params: "Param") -> "Param":
        values = gnp.concatenate([p.values for p in params])
        paths = sum((p.paths for p in params), [])
        names = sum((p.names for p in params), [])
        normalizations = sum((p.normalizations for p in params), [])
        bounds = sum((p.bounds for p in params), [])
        return Param(values, paths, normalizations, names, bounds)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            self.names[i]: {
                "value": self._values[i],
                "path": self.paths[i],
                "normalization": self.normalizations[i].value,
                "denormalized": self.denormalized_values[i],
                "bounds": self.bounds[i],
            }
            for i in range(self.dim)
        }

    def to_simple_dict(self) -> dict:
        return {name: val for name, val in zip(self.names, self.denormalized_values)}

    def __repr__(self) -> str:
        # Prepare the raw values for all fields
        raw_data = []
        for i in range(self.dim):
            name = self.names[i] + ":"
            path = "->".join(self.paths[i])
            norm = self.normalizations[i].value
            bounds = (
                f"[{self.bounds[i][0]:.4g}, {self.bounds[i][1]:.4g}]"
                if self.bounds[i]
                else "(-inf, inf)"
            )
            value = ftos(self._values[i]) # f"{self._values[i]:.4f}"
            denorm = ftos(self.denormalized_values[i]) #  f"{self.denormalized_values[i]:.4f}"
            raw_data.append((name, path, norm, bounds, value, denorm))

        # Compute max width for each column
        headers = ("Name:", "Path", "Norm", "Bounds", "Value", "Denorm")
        columns = list(zip(*raw_data))  # Transpose
        widths = [
            max(len(h), max(len(val) for val in col))
            for h, col in zip(headers, columns)
        ]

        # Format the header
        header = "    ".join(h.rjust(w) for h, w in zip(headers, widths))
        lines = [header]

        # Format each row
        for row in raw_data:
            line = "    ".join(val.rjust(w) for val, w in zip(row, widths))
            lines.append(line)

        return "\n".join(lines)


def make_anisotropic_param(
    d: Optional[int] = None,
    values: Optional[Union[List[float], gnp.ndarray]] = None,
    logsigma2_bounds: Optional[Tuple[float, float]] = None,
    loginvrho_bounds: Optional[Tuple[float, float]] = None,
    name_prefix: str = "",
) -> Param:
    """
    Build a Param object for anisotropic covariance [sigma2, rho_1, ..., rho_d].

    If `values` is provided, its length must be d + 1.
    If not, d must be specified and default values [0.0, -1.0, ..., -1.0] are used.

    Returns a Param object with [log, log_inv, ..., log_inv] normalization.
    """
    if values is not None:
        values = gnp.asarray(values)
        d = len(values) - 1
    elif d is not None:
        values = gnp.array([0.0] + [-1.0] * d)
    else:
        raise ValueError("Must provide either `values` or `d`.")

    names = [f"{name_prefix}sigma2"] + [f"{name_prefix}rho_{i}" for i in range(d)]
    paths = [["covparam", "variance"]] + [["covparam", "lengthscale"]] * d
    normalizations = [Normalization.LOG] + [Normalization.LOG_INV] * d
    bounds = [logsigma2_bounds] + [loginvrho_bounds] * d

    return Param(
        values=values,
        names=names,
        paths=paths,
        normalizations=normalizations,
        bounds=bounds,
    )


def param_from_covparam_anisotropic(
    covparam: Union[List[float], gnp.ndarray],
    logsigma2_bounds: Optional[Tuple[float, float]] = None,
    loginvrho_bounds: Optional[Tuple[float, float]] = None,
    name_prefix: str = "",
) -> Param:
    covparam = gnp.asarray(covparam)
    d = len(covparam) - 1
    values = covparam
    names = [f"{name_prefix}sigma2"] + [f"{name_prefix}rho_{i}" for i in range(d)]
    paths = [["covparam", "variance"]] + [["covparam", "lengthscale"]] * d
    normalizations = [Normalization.LOG] + [Normalization.LOG_INV] * d
    bounds = [logsigma2_bounds] + [loginvrho_bounds] * d
    return Param(
        values=values,
        paths=paths,
        normalizations=normalizations,
        names=names,
        bounds=bounds,
    )


def param_from_covparam_anisotropic_noisy(
    covparam: Union[List[float], gnp.ndarray],
    logsigma2_bounds: Optional[Tuple[float, float]] = None,
    logsigma2_noise_bounds: Optional[Tuple[float, float]] = None,
    loginvrho_bounds: Optional[Tuple[float, float]] = None,
    name_prefix: str = "",
) -> Param:
    d = len(covparam) - 2
    values = covparam
    names = [f"{name_prefix}sigma2"] + [f"{name_prefix}sigma2_noise"] + [f"{name_prefix}rho_{i}" for i in range(d)]
    paths = [["covparam", "variance"]] + [["covparam", "variance"]] + [["covparam", "lengthscale"]] * d
    normalizations = [Normalization.LOG] + [Normalization.LOG] + [Normalization.LOG_INV] * d
    bounds = [logsigma2_bounds] + [logsigma2_noise_bounds] + [loginvrho_bounds] * d
    return Param(
        values=values,
        paths=paths,
        normalizations=normalizations,
        names=names,
        bounds=bounds,
    )
