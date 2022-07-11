.. image:: images/logo.png
  :width: 300

========================
GPmp documentation
========================

Welcome to the GPmp API: the Gaussian process micro package.

GPmp intends to provide building blocks for GP-based algorithms. It uses auto-differentiation and JIT compilation features provided by `JAX <https://jax.readthedocs.io/>`_. It should be fast and easily
customizable.

For the purpose of simplicity, the gpmp does not check the validity of arguments. This is left to the responsibility of the user / calling
code.

Implemented Methods
===================

*   GP interpolation and regression with unknown mean / intrisinc kriging

*   The standard Gaussian likelihood and the restricted likelihood of a model

*   Leave-one-out predictions using fast cross-validation formulas

*   Conditional sample paths

It is up to the user to write the mean and covariance functions for setting a GP model. However, GPmp provides building blocks for:

*   Anisotropic scaling

*   Distance matrix

*   Matérn kernels with half-integers regularties

*   Parameter selection using maximum likelihood / restricted maximum likelihood / or user-defined criteria

Installation
============

.. panels::
    :card: + install-card
    :column: col-12 p-3

    Dev mode
    ^^^^^^^^^^^^^^^^^^^^^^^^
    Dowload the git repository (or download a zip version) and install in dev mode
    ++++

    .. code-block:: bash

        git clone https://github.com/emvazquez/gpmp.git

        pip install -e .
    
    ---
    :column: col-12 p-3

    Install in user mode
    ^^^^^^^^^^^^^^^^^^^^
    ++++

    .. code-block:: bash

        pip install gpmp

Content
====================
.. toctree::
    :maxdepth: 2

    core
    kernel
    misc/index
    examples/index