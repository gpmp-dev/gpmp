Custom covariance
=================

This example shows how to define a covariance function outside GPmp's built-in
Matern helpers and use it in a one-dimensional interpolation workflow.

What this example does
----------------------

The script defines a user covariance callable with the same signature expected
by :class:`gpmp.core.Model`: ``covariance(x, y, covparam, pairwise=False)``.
It then builds a GP model from that covariance, selects covariance parameters,
and predicts on a dense one-dimensional grid.

How to read the figure
----------------------

The plot shows the reference function, observations, posterior mean, and
uncertainty envelope obtained with the custom covariance. If the covariance is
valid and the selected parameters are reasonable, the posterior mean should
interpolate the noise-free observations and the uncertainty should reflect
local data density.

API points
----------

* Custom covariance functions should use :mod:`gpmp.num` operations when they
  must work with both NumPy and torch backends.
* The covariance callable must accept the ``pairwise`` argument used internally
  by GPmp prediction and likelihood routines.
* Once the model is built, parameter selection and prediction use the same API
  as for built-in covariance functions.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example05_1d_custom_kernel as ex

   ex.main()

Script: ``examples/gpmp_example05_1d_custom_kernel.py``

.. literalinclude:: ../../../examples/gpmp_example05_1d_custom_kernel.py
   :language: python
   :linenos:
