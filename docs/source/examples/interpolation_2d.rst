2D interpolation
================

This example builds an anisotropic Matern GP on a two-dimensional test function.
It illustrates the same modeling sequence as the 1D interpolation example, but
with a two-dimensional input space and contour plots for spatial diagnostics.

What this example does
----------------------

The script selects a two-dimensional test function, draws observation points,
constructs a GP model, and selects covariance parameters with a REMAP criterion.
The covariance vector follows the GPmp convention
``[log(sigma2), -log(rho_0), -log(rho_1)]``. The selected model is evaluated on
a regular grid to compare the reference function and the GP approximation.

How to read the figure
----------------------

The four panels show the reference function, GP posterior mean, absolute
prediction error, and posterior standard deviation. Observation points are
shown as red dots. The error and posterior standard deviation panels should be
read together: a well-calibrated model should assign larger uncertainty in
regions with sparse observations or difficult extrapolation.

API points
----------

* ``gp.misc.designs.ldrandunif`` creates a space-filling observation design.
* ``gp.kernel.make_selection_criterion_with_gradient`` can wrap a custom
  criterion with a gradient for SciPy optimization.
* ``gp.kernel.autoselect_parameters`` performs the actual numerical
  optimization when lower-level control is desired.

.. jupyter-execute::
   :hide-code:

   import numpy as np
   import matplotlib.pyplot as plt
   from examples import gpmp_example03_2d as ex

   f, dim, box, ni = ex.select_test_function(1)
   nt = [80, 80]
   xt = ex.gp.misc.designs.regulargrid(dim, nt, box)
   zt = f(xt)
   xi = ex.gp.misc.designs.ldrandunif(dim, ni, box)
   zi = f(xi)
   model = ex.create_model()
   covparam0 = ex.gp.kernel.anisotropic_parameters_initial_guess(model, xi, zi)
   nlrl, nlrl_pregrad, nlrl_nograd, dnlrl = (
       ex.gp.kernel.make_selection_criterion_with_gradient(
           model, ex.gp.kernel.neg_log_restricted_posterior_power_laws_prior, xi, zi
       )
   )
   covparam, info = ex.gp.kernel.autoselect_parameters(
       covparam0, nlrl_pregrad, dnlrl, info=True
   )
   model.covparam = ex.gnp.asarray(covparam)
   zpm, zpv = model.predict(xi, zi, xt)

   cmap = plt.get_cmap("PiYG")
   fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
   data = [zt, zpm, np.abs(zpm - zt), np.sqrt(zpv)]
   titles = ["reference", "GP approximation", "absolute error", "posterior std"]
   for ax, z, title in zip(axes.flat, data, titles):
       cs = ax.contourf(
           xt[:, 0].reshape(nt),
           xt[:, 1].reshape(nt),
           z.reshape(nt),
           levels=30,
           cmap=cmap,
       )
       ax.plot(xi[:, 0], xi[:, 1], "ro", markersize=3)
       ax.set_title(title)
       ax.set_xlabel("$x_1$")
       ax.set_ylabel("$x_2$")
       fig.colorbar(cs, ax=ax, shrink=0.8)
   fig.tight_layout()

Script: ``examples/gpmp_example03_2d.py``

.. literalinclude:: ../../../examples/gpmp_example03_2d.py
   :language: python
   :linenos:
