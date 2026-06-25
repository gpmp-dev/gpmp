GPmp Documentation
==================

``GPmp`` provides building blocks for kriging / Gaussian-process (GP)
interpolation and regression: covariance modeling, covariance-parameter
selection, diagnostics, conditional simulation, and posterior sampling of
covariance parameters.

The package is meant for GP-based algorithms and research software. Its API is
small and explicit: users provide the mean and covariance functions, choose or
define selection criteria, inspect diagnostics, and keep numerical backend
objects visible through ``gpmp.num``. The backend can be either NumPy or
PyTorch.

Features
--------

* GP interpolation and regression with known or unknown mean functions.
* Maximum likelihood, restricted maximum likelihood, REMAP, and custom
  parameter-selection criteria.
* Posterior parameter sampling with MH, NUTS, and SMC helpers.
* Leave-one-out diagnostics, model reports, and selection-criterion plots.
* Conditional sample paths and utilities for Matérn covariance models.

Positioning
-----------

Users provide the mean and covariance functions. GPmp supplies covariance
kernels, parameter initialization and selection routines, numerical backends,
diagnostics, and posterior samplers.

* `GPyTorch <https://docs.gpytorch.ai/>`_,
  `GPflow <https://gpflow.github.io/GPflow/>`_, and
  `GPJax <https://docs.jaxgaussianprocesses.com/>`_ provide APIs for
  scalable, variational, deep, or multi-output GP models.
* `SMT <https://smt.readthedocs.io/>`_ provides engineering surrogate
  modeling tools, with sampling methods, mixed variables, and several
  surrogate model families.
* `scikit-learn <https://scikit-learn.org/stable/modules/gaussian_process.html>`_
  provides a stable estimator API for standard GP regression and
  classification.

GPmp exposes parameter selection, diagnostics, and exact Gaussian-process
computations in code that can be adapted for GP-based algorithms.

Installation from source
------------------------

Editable installation requires a local clone of the repository. Clone GPmp,
enter the repository root, then run ``pip install -e .``:

.. code-block:: shell

   git clone https://github.com/gpmp-dev/gpmp.git
   cd gpmp
   pip install -e .

Documentation contents
----------------------

* :doc:`tutorial` builds and diagnoses a Hartmann4 interpolation model.
* :doc:`examples/index` contains rendered scripts for interpolation,
  regression, parameter selection, posterior sampling, sample paths, and
  dataloader-based selection.
* :doc:`gpmp` documents the public API, including backend objects, core models,
  kernels, parameter helpers, diagnostics, samplers, plotting helpers, designs,
  and test functions.
* :doc:`references` lists the literature cited by the tutorial and examples.

Related package
---------------

`gpmp-contrib <https://github.com/gpmp-dev/gpmp-contrib>`_ extends GPmp with
computer-experiment objects, model containers, sequential strategies,
optimization criteria, set-estimation tools, and reGP utilities.

.. toctree::
   :hidden:
   :maxdepth: 3

   self
   tutorial
   examples/index
   gpmp
   references

How to Cite
-----------

.. code-block:: bibtex

   @software{gpmp2026,
     author       = {Emmanuel Vazquez},
     title        = {GPmp: the Gaussian Process micro package},
     year         = {2026},
     url          = {https://github.com/gpmp-dev/gpmp},
     note         = {Version 0.9.37},
   }

Authors
-------

See `AUTHORS.md <https://github.com/gpmp-dev/gpmp/blob/main/AUTHORS.md>`_.

License
-------

GPmp is free software released under the GNU General Public License v3.0.
See `LICENSE.txt <https://github.com/gpmp-dev/gpmp/blob/main/LICENSE.txt>`_
for details.
