Custom covariance
=================

This example shows how to define a covariance function outside GPmp's built-in
Matern helpers and use it in a one-dimensional interpolation example.

What this example does
----------------------

The script defines a user covariance callable with the same signature expected
by :class:`gpmp.core.Model`: ``covariance(x, y, covparam, pairwise=False)``.
It then builds a GP model from that covariance, selects covariance parameters,
and predicts on a dense one-dimensional grid.

Outputs
-------

The displayed quantities are the reference function, observations, posterior
mean, and uncertainty envelope obtained with the custom covariance. For a
positive definite covariance and converged parameter selection, the posterior
mean interpolates the noise-free observations and the uncertainty envelope
widens where observations are sparse.

API points
----------

* Custom covariance functions should use :mod:`gpmp.num` operations when they
  must work with both NumPy and torch backends.
* The covariance callable must accept the ``pairwise`` argument used internally
  by GPmp prediction and likelihood routines.
* Once the model is built, parameter selection and prediction use the same API
  as for built-in covariance functions.

Covariance construction
-----------------------

``gpmp.core.Model`` calls the covariance function in three situations:

``k(x_i, x_i)``
    Covariance matrix between observation points.  In the example, this is
    handled by ``kernel_ii_or_tt``.

``k(x_i, x_t)``
    Cross-covariance matrix between observation points and prediction points.
    In the example, this is handled by ``kernel_it``.

``k(x_t, x_t)``
    Prior covariance at prediction points.  GPmp uses only its diagonal when
    computing posterior variances, unless a full posterior covariance matrix is
    requested.  In the example, this is also handled by ``kernel_ii_or_tt``.

The suffix ``ii_or_tt`` means "same-set covariance": the two arguments are the
same point set, either observations ``x_i`` or prediction points ``x_t``.  The
suffix ``it`` means "cross covariance" between observations and prediction
points.

The ``pairwise`` flag controls the returned shape.  With ``pairwise=False``,
the covariance function returns a full matrix.  With ``pairwise=True``, it
returns the elementwise covariance vector
``[k(x_0, y_0), ..., k(x_{n-1}, y_{n-1})]``.  When ``y`` is ``None``, this is the
diagonal of ``k(x, x)``.

The wrapper ``kernel`` dispatches to the same-set or cross-set function.  This
is the callable passed to ``gpmp.core.Model``.  The small nugget added in
``kernel_ii_or_tt`` is numerical jitter on same-set covariance matrices.  It is
not an observation-noise model.

The example uses a one-dimensional Matern covariance with

.. math::

   k_\theta(x, x')
   =
   \sigma^2 c_2\left(|x-x'|/\rho\right),
   \qquad
   \theta = \left(\log(\sigma^2), -\log(\rho)\right).

The custom part of the example is not the mathematical covariance itself, but
the way the covariance callable is written and dispatched.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example05_1d_custom_kernel as ex

   ex.main()

Script: ``examples/gpmp_example05_1d_custom_kernel.py``

.. literalinclude:: ../../../examples/gpmp_example05_1d_custom_kernel.py
   :language: python
   :linenos:
