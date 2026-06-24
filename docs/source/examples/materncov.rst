Plotting Matern covariances
===========================

This example plots Matern covariance kernels with half-integer smoothness
parameters. It is the smallest example in the documentation and is useful for
understanding the shape of the kernels used by the interpolation and regression
examples.

What this example does
----------------------

The script builds a one-dimensional grid of scaled distances ``h`` and evaluates
``gp.kernel.maternp_kernel(p, abs(h))`` for several integer values of ``p``.
For half-integer Matern kernels, ``nu = p + 1/2``. Increasing ``p`` produces
smoother sample paths and a covariance function that is flatter near the
origin.

How to read the figure
----------------------

All curves are normalized to one at ``h = 0``. The curve with ``p = 0`` is the
exponential kernel and decays sharply away from the origin. Larger ``p`` values
produce stronger local smoothness and a slower initial decay. This plot is only
a kernel-shape comparison: no observations or parameter selection are involved.

API points
----------

* ``gp.kernel.maternp_kernel`` evaluates the correlation kernel as a function
  of scaled distance.
* ``gp.plot.Figure`` is a lightweight Matplotlib wrapper used throughout the
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
