Plotting Matern covariances
===========================

This example plots Matern covariance kernels with half-integer smoothness
parameters. It is the smallest example in the documentation and shows the shape
of the kernels used by the interpolation and regression examples.

What this example does
----------------------

The script builds a one-dimensional grid of scaled distances ``h`` and evaluates
``gp.kernel.maternp_kernel(p, abs(h))`` for several integer values of ``p``.
For half-integer Matern kernels, ``nu = p + 1/2``. Increasing ``p`` produces
smoother sample paths and a covariance function that is flatter near the
origin.

Mathematical description
------------------------

The function ``maternp_kernel`` returns the correlation part of a stationary
Matern covariance. For :math:`\nu = p + 1/2`, GPmp evaluates

.. math::

   c_p(h)
   =
   \exp(-2\sqrt{\nu}\,h)
   \frac{\Gamma(p+1)}{\Gamma(2p+1)}
   \sum_{j=0}^{p}
   \frac{(p+j)!}{j!(p-j)!}
   \left(4\sqrt{\nu}\,h\right)^{p-j}.

The full anisotropic covariance used in later examples has the form

.. math::

   k_\theta(x, x')
   =
   \sigma^2 c_p\left(
       \left\|
       \left(
       (x_0-x'_0)/\rho_0,\ldots,
       (x_{d-1}-x'_{d-1})/\rho_{d-1}
       \right)
       \right\|_2
   \right).

Outputs
-------

All curves are normalized to one at ``h = 0``. The curve with ``p = 0`` is the
exponential kernel and decays sharply away from the origin. Larger ``p`` values
produce stronger local smoothness and a slower initial decay. No observations or
parameter selection are involved.

API points
----------

* ``gp.kernel.maternp_kernel`` evaluates the correlation kernel as a function
  of scaled distance.
* ``gp.plot.Figure`` is a small Matplotlib wrapper used throughout the
  examples.
* For full covariance matrices with variance and lengthscales, use
  ``gp.kernel.maternp_covariance`` instead.

.. jupyter-execute::
   :hide-code:

   import os
   import sys
   _this_dir = os.path.dirname(os.path.abspath(''))
   _main_dir = os.path.abspath(os.path.join(_this_dir))
   sys.path.append(os.path.join(_main_dir))

.. jupyter-execute:: ../../../examples/gpmp_example01_materncov.py
