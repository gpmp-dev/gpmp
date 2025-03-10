# GPmp: the Gaussian Process micro package

GPmp is a lightweight toolkit for Gaussian process (GP) modeling. It
provides the essential components for GP-based algorithms with a focus
on speed and customization.

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
 * vizualization of results
 * ...

## Backends

GPmp supports three numerical backends:
- **PyTorch:** Dynamic computation with auto-differentiation.
- **JAX:** Auto-differentiation with JIT compilation.
- **NumPy:** Basic computation (default if neither PyTorch nor JAX are
    found).

On startup, GPmp automatically selects the backend in this order:
PyTorch → JAX → NumPy. You can override this by setting the
`GPMP_BACKEND` environment variable before launching GPmp.

## Installation

Clone the repository:
```bash
git clone https://github.com/gpmp-dev/gpmp.git
```
Install in development mode:
```bash
pip install -e .
```

### Backend Setup

- **PyTorch:** Recommended for best performance.
- **JAX:** 
  - For CPU-only usage, install directly from PyPI:
    ```bash
    pip install jax
    ```
  - For NVIDIA GPU support (e.g., with CUDA 12), run:
    ```bash
    pip install -U "jax[cuda12]"
    ```
  Detailed, platform-specific instructions are available in the
  [official JAX documentation](https://github.com/google/jax#installation).

## Usage

Check the examples in the repository for a quick start. Customize your
own mean and covariance functions as needed.

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
@software{gpmp2025,
  author       = {Emmanuel Vazquez},
  title        = {GPmp: the Gaussian Process micro package},
  year         = {2025},
  url          = {https://github.com/gpmp-dev/gpmp},
  note         = {Version x.y},
}
```

*Please update the version number as appropriate.*

## Future Work

- Expand documentation and tutorials.
- Enhance diagnostic tools and model visualization.

## Authors

See [AUTHORS.md](AUTHORS.md) for details.

## License

GPmp is free software released under the GNU General Public License v3.0.
See [LICENSE](LICENSE.txt) for more details.
