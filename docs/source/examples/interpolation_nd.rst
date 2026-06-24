Higher-dimensional interpolation
================================

This example applies anisotropic Matern GP interpolation to a test function in
dimension greater than two. Since direct contour plots are no longer available,
the preview uses prediction scatter plots, leave-one-out (LOO) diagnostics, and
one-dimensional cross sections.

What this example does
----------------------

The script chooses a benchmark function, evaluates it at observation points,
selects covariance parameters with REMAP, predicts independent test points, and
computes LOO predictions. LOO prediction removes one observation at a time,
predicts it from all other observations, and compares the prediction with the
removed value.

Mathematical object
-------------------

The model remains

.. math::

   Z \sim \mathcal{GP}(m, k_\theta),
   \qquad
   Z_i = Z(x_i),

and ``zi`` stores realized values :math:`z_i`. The input dimension is too large
for a direct contour plot. The first diagnostic therefore compares realized
reference values :math:`z_t` with the posterior mean
:math:`\mathbb{E}[Z_t\mid Z_i=z_i]` on independent test points.

For the LOO diagnostic, each observation is predicted after removing it from
the conditioning set:

.. math::

   \widehat z_{i,-i}
   =
   \mathbb{E}\left[
       Z_i
       \mid
       Z_j=z_j,\; j\ne i
   \right].

The cross-section plots fix all coordinates except one and display
:math:`x_j \mapsto \mathbb{E}[Z(x)\mid Z_i=z_i]` and its pointwise
uncertainty.

Outputs
-------

The first figure compares GP posterior means with reference values on
independent test points. Points close to the diagonal indicate small test
errors.

The second figure compares LOO predictions with observed values. Vertical bars
show nominal predictive intervals. Large deviations from the diagonal or many
observations outside the intervals indicate poorly selected covariance
parameters, missing structure, or predictive intervals that are too narrow.

The last figure shows prediction cross sections. Each panel fixes all
coordinates except one and plots the posterior mean and intervals along that
coordinate. Black points are projected observations. The red point is the
observation used as the cross-section anchor.

API points
----------

* ``model.loo(xi, zi)`` returns LOO means, variances, and errors.
* ``gp.plot.plot_loo`` provides a compact diagnostic for high-dimensional
  problems where spatial plots are impossible.
* ``gp.plot.crosssections`` plots one-dimensional slices through selected
  observation points.
* REMAP parameter selection can be used when pure likelihood criteria produce
  poorly identified lengthscales.

.. jupyter-execute::
   :hide-code:

   import numpy as np
   import matplotlib.pyplot as plt
   from examples import gpmp_example04_nd as ex

   problem_name, f, dim, box, ni, xi, nt, xt = ex.choose_test_case(1)
   zi = f(xi)
   zt = f(xt)
   model = ex.gp.Model(ex.constant_mean, ex.kernel)
   model, info = ex.gp.kernel.select_parameters_with_remap(model, xi, zi, info=True)
   zpm, zpv = model.predict(xi, zi, xt)

   zt_np = ex.gnp.to_np(zt).reshape(-1)
   zpm_np = ex.gnp.to_np(zpm).reshape(-1)
   zmin = min(float(np.min(zt_np)), float(np.min(zpm_np)))
   zmax = max(float(np.max(zt_np)), float(np.max(zpm_np)))

   fig, ax = plt.subplots(figsize=(5.4, 4.4))
   ax.plot(zt_np, zpm_np, "ko", markersize=3)
   ax.plot([zmin, zmax], [zmin, zmax], "--", linewidth=1)
   ax.set_xlabel("reference values")
   ax.set_ylabel("posterior mean")
   ax.set_title(f"{problem_name}: test predictions")
   ax.grid(True, "major", linestyle=(0, (1, 5)), linewidth=0.5)

   zloom, zloov, eloo = model.loo(xi, zi)
   ex.gp.plot.plot_loo(zi, zloom, zloov)

   _ = ex.gp.plot.crosssections(
       model,
       xi,
       zi,
       box,
       ind_i=[0, 10],
       ind_dim=list(range(dim)),
       nt=120,
   )

Script: ``examples/gpmp_example04_nd.py``

.. literalinclude:: ../../../examples/gpmp_example04_nd.py
   :language: python
   :linenos:
