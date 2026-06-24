gpmp.kernel covariance helpers
==============================

This page documents covariance functions and covariance-parameter initial
guesses. These are low-level functions: they work with backend-native arrays
and plain covariance-parameter vectors, independently from the optional
:mod:`gpmp.parameter` display helpers.

Covariance functions
--------------------

``exponential_kernel``, ``matern32_kernel``, and ``maternp_kernel`` evaluate
correlation kernels as functions of a scaled distance. ``maternp_covariance``
combines the Matern kernel with the GPmp covariance-parameter convention
``[log(sigma2), -log(rho_0), ...]`` and is the usual covariance function for
anisotropic Matern models.

exponential_kernel
~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.exponential_kernel
   :no-index:

matern32_kernel
~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.matern32_kernel

maternp_kernel
~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.maternp_kernel

maternp_covariance
~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.maternp_covariance

Initial guesses
---------------

Initial-guess helpers construct a covariance-parameter vector from observation
points and values. They are often used as ``covparam0`` for selection
procedures, or as a prior anchor for REMAP procedures.

anisotropic_parameters_initial_guess_zero_mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.anisotropic_parameters_initial_guess_zero_mean

anisotropic_parameters_initial_guess
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.anisotropic_parameters_initial_guess
