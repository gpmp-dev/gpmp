Higher-dimensional interpolation
================================

This example applies anisotropic Matern GP interpolation to a test function in
dimension greater than two. Since direct contour plots are no longer available,
the preview focuses on leave-one-out (LOO) diagnostics.

What this example does
----------------------

The script chooses a benchmark function, evaluates it at observation points,
selects covariance parameters with REMAP, and computes LOO predictions. LOO
prediction removes one observation at a time, predicts it from all other
observations, and compares the prediction with the removed value.

How to read the figure
----------------------

The horizontal axis shows observed values and the vertical axis shows LOO
predictions. Points close to the diagonal indicate small LOO errors. Vertical
bars show nominal predictive intervals. Large deviations from the diagonal or
many observations outside the intervals indicate a poor covariance fit, missing
structure, or under-estimated uncertainty.

API points
----------

* ``model.loo(xi, zi)`` returns LOO means, variances, and errors.
* ``gp.plot.plot_loo`` provides a compact diagnostic for high-dimensional
  problems where spatial plots are impossible.
* REMAP parameter selection can be used when pure likelihood criteria produce
  poorly identified lengthscales.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example04_nd as ex

   problem_name, f, dim, box, ni, xi, nt, xt = ex.choose_test_case(1)
   zi = f(xi)
   model = ex.gp.Model(ex.constant_mean, ex.kernel)
   model, info = ex.gp.kernel.select_parameters_with_remap(model, xi, zi, info=True)
   zloom, zloov, eloo = model.loo(xi, zi)
   ex.gp.plot.plot_loo(zi, zloom, zloov)

Script: ``examples/gpmp_example04_nd.py``

.. literalinclude:: ../../../examples/gpmp_example04_nd.py
   :language: python
   :linenos:
