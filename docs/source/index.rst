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

Numerical arrays are backend-native objects exposed through ``gpmp.num``.
Use ``import gpmp.num as gnp`` in custom mean, covariance, and criterion
functions when code should work with both supported backends.

Features
--------

* GP interpolation and regression with known or unknown mean functions.
* Maximum likelihood, restricted maximum likelihood, REMAP, and custom
  parameter-selection criteria.
* Posterior parameter sampling with MH, NUTS, and SMC helpers.
* Leave-one-out diagnostics, model reports, and selection-criterion plots.
* Conditional sample paths and utilities for Matérn covariance models.

Backends
--------

GPmp supports two numerical backends:

* ``numpy``: the default fallback backend and often efficient for
  small-to-medium workloads.
* ``torch``: provides automatic differentiation and is useful when
  gradient-based parameter selection or high-dimensional parameter
  problems are part of the workflow.

Backend selection happens at import time. Set ``GPMP_BACKEND`` to
``numpy`` or ``torch`` before importing GPmp to choose explicitly. If the
variable is not set, GPmp uses PyTorch when available and otherwise falls
back to NumPy.

Installation
------------

Clone the repository and install it in development mode:

.. code-block:: shell

   git clone https://github.com/gpmp-dev/gpmp.git
   cd gpmp
   pip install -e .

Install PyTorch separately when the ``torch`` backend is desired.

Usage
-----

See the tutorial and examples for complete workflows.

.. toctree::
   :hidden:
   :maxdepth: 3

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
     note         = {Version 0.9.35},
   }

Authors
-------

See ``AUTHORS.md``.

License
-------

GPmp is free software released under the GNU General Public License v3.0.
See ``LICENSE.txt`` for details.
