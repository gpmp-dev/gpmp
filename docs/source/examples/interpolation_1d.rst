1D interpolation
================

This example builds a one-dimensional noise-free Gaussian process model and
selects covariance parameters by restricted maximum likelihood (REML). It is the
recommended first complete workflow because it shows the basic GPmp sequence:
construct a model, select ``covparam``, predict, and plot the result.

What this example does
----------------------

The script creates observation points ``xi`` and observations ``zi`` from a
known reference function. It defines a constant-mean GP with an anisotropic
Matern covariance and calls ``gp.kernel.select_parameters_with_reml``. The
selected covariance parameters are stored in ``model.covparam`` and are then
used by ``model.predict`` on a dense grid ``xt``.

How to read the figure
----------------------

The figure compares four objects: the reference function, the observations, the
posterior mean, and the posterior uncertainty envelope. Because the observations
are treated as noise-free, the posterior mean should interpolate the observed
values. The uncertainty is small near observations and larger away from them.

API points
----------

* Use :class:`gpmp.core.Model` or ``gp.Model`` to assemble a mean function and
  a covariance function.
* Use ``select_parameters_with_reml`` when covariance parameters should be
  selected by REML.
* Use ``model.predict(xi, zi, xt)`` to compute posterior mean and variance at
  prediction points.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example02_1d_interpolation as ex

   xt, zt, xi, zi = ex.generate_data()
   model = ex.gp.Model(ex.constant_mean, ex.kernel)
   model, info = ex.gp.kernel.select_parameters_with_reml(model, xi, zi, info=True)
   zpm, zpv = model.predict(xi, zi, xt)
   ex.visualize_results(xt, zt, xi, zi, zpm, zpv)

Script: ``examples/gpmp_example02_1d_interpolation.py``

.. literalinclude:: ../../../examples/gpmp_example02_1d_interpolation.py
   :language: python
   :linenos:
