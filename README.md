# GPmp: the Gaussian process micro package

## What is GPmp?

GPmp provides simple building blocks for GP-based algorithms.  It is meant to
be fast and easily customizable.

The user can choose between three backends to perform numerical
computations: plain numpy, [JAX](https://jax.readthedocs.io/) or
[PyTorch](). When JAX or PyTorch is used, GPmp relies on
auto-differentiation to compute gradients. GPmp also uses JIT
compilation features provided by JAX.

As core features, GPmp implements:
 * GP interpolation and regression with known or unknown mean / intrinsinc kriging
 * The standard Gaussian likelihood and the restricted likelihood of a model
 * Leave-one-out predictions using fast cross-validation formulas
 * Conditional sample paths

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

Note also that for the purpose of simplicity, the gpmp does not check
the validity of arguments. This is left to the responsibility of the
user / calling code.

## Installation

Clone the git repository:
```
git clone https://github.com/gpmp-dev/gpmp.git
```
(or download a zip version)

Install in dev mode:
```
pip install -e .
```

Remarks:

1. It is recommended to run GPmp using PyTorch or JAX as a backend instead of Numpy.

2. Upon initialization, GPmp automatically detects if PyTorch is
   installed and sets the GPMP_BACKEND environment variable to 'torch'
   by default. If PyTorch is not found, it then searches for JAX and
   sets GPMP_BACKEND to 'jax'. If neither PyTorch nor JAX are
   detected, GPmp defaults to using Numpy, setting the GPMP_BACKEND
   environment variable to 'numpy'. Users have the option to manually
   override the chosen backend by setting the GPMP_BACKEND variable
   before launching GPmp.
   
2. JAX, which requires jaxlib, which is
   supported on Linux and macOS platforms, and on Windows via the
   Windows Subsystem for Linux. There is some initial native Windows
   support, but it is still somewhat immature (see below).

2. To install JAX with both CPU and NVidia GPU support, CUDA and CuDNN
   must be installed first. See
   https://github.com/google/jax#installation and
   https://jax.readthedocs.io/en/latest/changelog.html for details.

   For instance, to install jaxlib  with Cuda 11 and cudnn 8.2 or newer:
```bash   
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
   
3. To install jaxlib on Windows, one can follow instructions at
   https://github.com/cloudhan/jax-windows-builder.  Select and
   download a version of jaxlib from
   https://whls.blob.core.windows.net/unstable/index.html according to
   your Python version. Then install jax manually.

```powershell
pip install <jaxlib_whl>
pip install jax
```

## Usage

Please refer to the examples that illustrate the different features of GPmp.

## Documentation with Sphinx
The automatic documentation is created using [Sphinx](https://www.sphinx-doc.org/en/master/) and following the python docstrings recommendations from [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)

The html theme is taken from
>pip install pydata_sphinx_theme

In order to generate the intereactive html:
>cd ./doc

>make html

## To do

* write documentation

## Authors

 See AUTHORS.md

## Copyright

 Copyright (C) 2022-2024 CentraleSupelec

## License

 GPmp is free software: you can redistribute it and/or modify it
 under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 GPmp is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 License for more details.

 You should have received a copy of the GNU General Public License
 along with gpmp. If not, see http://www.gnu.org/licenses/.
