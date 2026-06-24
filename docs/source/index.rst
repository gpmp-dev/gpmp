GPmp Documentation
==================

``GPmp`` is a lightweight toolkit for Gaussian process modeling. It
provides compact building blocks for GP interpolation, regression,
parameter selection, diagnostics, posterior parameter sampling, and
conditional simulation.

The package favors explicit modeling choices: users provide the mean and
covariance functions, while GPmp supplies common covariance kernels,
parameter initialization and selection routines, numerical backends, and
diagnostic helpers.

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

GPmp targets kriging and computer-experiment code where the mean function,
covariance function, covariance parameters, selection criterion, diagnostics,
and numerical backend objects should remain directly inspectable.

* `GPyTorch <https://docs.gpytorch.ai/>`_,
  `GPflow <https://gpflow.github.io/GPflow/>`_, and
  `GPJax <https://docs.jaxgaussianprocesses.com/>`_ provide broader
  automatic-differentiation ecosystems for scalable, variational, deep, or
  multi-output GP models.
* `SMT <https://smt.readthedocs.io/>`_ focuses on engineering surrogate
  modeling, with sampling methods, mixed variables, and several surrogate
  model families.
* `scikit-learn <https://scikit-learn.org/stable/modules/gaussian_process.html>`_
  provides a stable estimator API for standard GP regression and
  classification.

GPmp's role is narrower: explicit parameter selection and diagnostics for exact
GP interpolation and regression, with compact code that can be adapted for
research experiments.

Installation from source
------------------------

Editable installation requires a local clone of the repository. Clone GPmp,
enter the repository root, then run ``pip install -e .``:

.. code-block:: shell

   git clone https://github.com/gpmp-dev/gpmp.git
   cd gpmp
   pip install -e .

Documentation map
-----------------

Start with :doc:`tutorial` for a complete Hartmann4 example. Use
:doc:`examples/index` for task-specific scripts and :doc:`gpmp` for the API
reference.

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
