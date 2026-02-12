# GPmp: the Gaussian Process micro package

GPmp is a lightweight toolkit for Gaussian process (GP) modeling. It
provides the essential components for GP-based algorithms with a focus
on performance and customization.

## Features

- **GP interpolation & regression:** Supports known or unknown mean
    functions (including intrinsic kriging).
- **Likelihood computation:** standard Gaussian and restricted likelihood.
- **Efficient cross-validation:** fast leave-one-out predictions.
- **Conditional sampling:** generate conditional sample paths.

It is up to the user to write the mean and covariance functions for
setting a GP model.

However, for the purpose of the example, GPmp provides functions for:
 * anisotropic scaling
 * distance matrix
 * Matérn kernels with half-integer regularities
 * parameter selection procedure using maximum likelihood, restricted maximum
   likelihood, or user-defined criteria
 * model diagnosis
 * vizualization helper
 * ...

## Backends

GPmp supports two numerical backends:
- **PyTorch:** Dynamic computation with auto-differentiation.
- **NumPy:** Basic computation (default if neither PyTorch nor JAX are
    found).

Backend selection order at import time:
1. If GPMP_BACKEND is set to torch or numpy, use it.
2. Otherwise: PyTorch if available, else NumPy.

Example:
export GPMP_BACKEND=torch
# optional:
export GPMP_DTYPE=float64   # or float32

## Installation

Clone the repository:
```bash
git clone https://github.com/gpmp-dev/gpmp.git
```
Install in development mode:
```bash
pip install -e .
```

## Dependencies

Core:
- NumPy

Recommended:
- PyTorch

Install PyTorch (CPU-only):
pip install torch

Verify:
python -c "import torch; print(torch.__version__)"

For GPU-enabled PyTorch, install the build matching the local CUDA setup.
Use the official PyTorch install selector to obtain the exact pip command.

## Quick start

See the examples/ directory.

Typical steps:
1. Provide mean and covariance functions.
2. Build a GP model.
3. Select parameters (ML / REML / custom criterion) and validate model.
4. Predict / cross-validate / sample conditionally / visualize.

## Documentation

GPmp’s documentation is built with
[Sphinx](https://www.sphinx-doc.org/en/master/) using the PyData
theme. To generate the HTML documentation:

```bash
cd doc
make html
```

## How to Cite

If you use GPmp in your research, please cite it as follows:

```bibtex
@software{gpmp2026,
  author       = {Emmanuel Vazquez},
  title        = {GPmp: the Gaussian Process micro package},
  year         = {2026},
  url          = {https://github.com/gpmp-dev/gpmp},
  note         = {Version x.y},
}
```

*Please update the version number as appropriate.*

## Future Work

- Expand documentation and tutorials.

## Authors

See [AUTHORS.md](AUTHORS.md) for details.

## License

GPmp is free software released under the GNU General Public License v3.0.
See [LICENSE](LICENSE.txt) for more details.
