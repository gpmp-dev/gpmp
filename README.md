<div align="left">
  <img src="https://raw.githubusercontent.com/gpmp-dev/gpmp/main/docs/source/images/logo_gpmp.svg" width="120" alt="GPmp logo"/>
</div>

# GPmp: Gaussian Process micro package

[![Python](https://img.shields.io/pypi/pyversions/gpmp.svg)](https://pypi.org/project/gpmp/)
[![PyPI](https://img.shields.io/pypi/v/gpmp.svg)](https://pypi.org/project/gpmp/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE.txt)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://gpmp-dev.github.io/gpmp/)
[![Docs build](https://github.com/gpmp-dev/gpmp/actions/workflows/docs.yml/badge.svg)](https://github.com/gpmp-dev/gpmp/actions/workflows/docs.yml)

[Website and documentation](https://gpmp-dev.github.io/gpmp/)
| [Examples](https://gpmp-dev.github.io/gpmp/examples/index.html)
| [API reference](https://gpmp-dev.github.io/gpmp/gpmp.html)
| [PyPI](https://pypi.org/project/gpmp/)
| [gpmp-contrib](https://github.com/gpmp-dev/gpmp-contrib)

GPmp provides building blocks for kriging / Gaussian-process (GP)
interpolation and regression: covariance modeling, covariance-parameter
selection, diagnostics, conditional simulation, and posterior sampling of
covariance parameters.

The package is meant for GP-based algorithms and research software. Its API is
small and explicit: users provide the mean and covariance functions, choose or
define selection criteria, inspect diagnostics, and keep numerical backend
objects visible through `gpmp.num`. The backend can be either NumPy or PyTorch.

## When to use GPmp

Use `gpmp` when you need an explicit GP core for kriging, parameter selection,
diagnostics, posterior sampling, conditional simulation, plotting, data
utilities, or integration inside another algorithm.

Use `gpmp-contrib` when you want complete computer-experiment procedures,
Bayesian optimization algorithms, excursion-set estimation, set inversion, and
related sequential-design utilities.

GPmp currently focuses on exact GP computations. For very large data sets,
large-scale approximate inference is future work.

## Core features

- **Exact GP interpolation and regression:** zero, parameterized, and
  linear-predictor means, including intrinsic and universal kriging settings.
- **Covariance functions:** Matérn covariance helpers, anisotropic scaling,
  distance matrices, and user-defined covariance functions.
- **Parameter selection:** ML, REML, REMAP, user-defined criteria, bounds, and
  SciPy optimizer access.
- **Diagnostics:** leave-one-out quantities, prediction-performance summaries,
  parameter reports, and selection-criterion cross sections.
- **Posterior covariance-parameter sampling:** adaptive Metropolis-Hastings,
  NUTS, and tempered Sequential Monte Carlo.
- **Conditional simulation:** conditional sample paths for GP models.
- **Data utilities:** random designs, splits, cross-validation helpers, and
  dataloaders for batched criterion evaluation.
- **Plotting helpers:** plotting utilities for GP predictions,
  diagnostics, and examples.

## Numerical backends

GPmp supports two numerical backends:

- **NumPy:** default backend when PyTorch is unavailable, and often fast for many
  small-to-medium exact GP computations.
- **PyTorch:** enables automatic differentiation and is useful when gradient
  information is needed, especially in higher-dimensional parameter settings.

Backend selection order at import time:

1. If `GPMP_BACKEND` is set to `torch` or `numpy`, use it.
2. Otherwise, use PyTorch if available, then NumPy.

```bash
export GPMP_BACKEND=torch
export GPMP_DTYPE=float64
```

## Package split

```text
gpmp
  core model
  kernels
  parameter objects
  parameter selection
  diagnostics
  posterior samplers
  plotting helpers

gpmp-contrib
  model containers
  computer experiments
  Bayesian optimization algorithms
  expected improvement
  excursion-set estimation
  set inversion
  reGP utilities
```

## Install

```bash
pip install gpmp
python -c "print(__import__('gpmp').__version__)"
```

The verification command prints the installed GPmp version.

For local development:

```bash
git clone https://github.com/gpmp-dev/gpmp.git
cd gpmp
pip install -e .
```

Use `pip install -e ".[dev]"` for development tools, or
`pip install -e ".[docs]"` for documentation tools.

Install the companion package when you need complete computer-experiment
procedures:

```bash
pip install gpmp-contrib
```

GPmp depends on NumPy, SciPy, and Matplotlib. PyTorch is optional. Install it
when automatic differentiation is needed. For GPU-enabled PyTorch, use the
official PyTorch installation selector.

## Documentation

The documentation is available at <https://gpmp-dev.github.io/gpmp/>.

To build it locally:

```bash
cd docs
make html
```

## Public API

The intended public API is organized around:

- `gpmp.core`
- `gpmp.kernel`
- `gpmp.parameter`
- `gpmp.modeldiagnosis`
- `gpmp.mcmc`
- `gpmp.plot`
- `gpmp.num`

## How to cite

If you use GPmp in research, please cite:

```bibtex
@software{gpmp2026,
  author       = {Emmanuel Vazquez},
  title        = {GPmp: Gaussian Process micro package},
  year         = {2026},
  url          = {https://github.com/gpmp-dev/gpmp},
  note         = {Version 0.9.37},
}
```

Update the version number when citing another release.

## Minimal example

The basic sequence is: define mean and covariance functions, build a model,
select covariance parameters, predict, and diagnose the result.

GPmp keeps model construction explicit: users provide the mean function and
the covariance function.

```python
import gpmp as gp
import gpmp.num as gnp


def mean(x, param):
    return gnp.ones((x.shape[0], 1))


def covariance(x, y, covparam, pairwise=False):
    p = 3  # Matern regularity p + 1/2
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


xt = gp.misc.designs.regulargrid(1, 200, [[-1.0], [1.0]])
zt = gp.misc.testfunctions.twobumps(xt)
xi = gp.misc.designs.ldrandunif(1, 6, [[-1.0], [1.0]])
zi = gp.misc.testfunctions.twobumps(xi)

model = gp.Model(mean, covariance)
model, info = gp.kernel.select_parameters_with_reml(model, xi, zi, info=True)
zpm, zpv = model.predict(xi, zi, xt)

gp.modeldiagnosis.diag(model, info, xi, zi)
```

The final line prints parameter-selection diagnostics. The rendered
documentation page shows the corresponding interpolation plot:
[1D interpolation example](https://gpmp-dev.github.io/gpmp/examples/interpolation_1d.html).

## Authors

See [AUTHORS.md](AUTHORS.md) for details.

## License

GPmp is free software released under the GNU General Public License v3.0. See
[LICENSE](LICENSE.txt) for details.
