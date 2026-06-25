2D interpolation
================

This example builds an anisotropic Matern GP on a two-dimensional test function.
It illustrates the same modeling sequence as the 1D interpolation example, but
with a two-dimensional input space and contour plots for spatial diagnostics.

What this example does
----------------------

The script selects a two-dimensional test function, draws observation points,
constructs a GP model, and selects covariance parameters with a REMAP criterion.
The covariance vector follows the convention
``[log(sigma2), -log(rho_0), -log(rho_1)]``. The selected model is evaluated on
a regular grid to compare the reference function and the GP approximation.

Mathematical description
------------------------

The model is the same noise-free conditional GP as in the 1D interpolation
example, but with two-dimensional points
:math:`x=(x_0,x_1)`. The anisotropic Matern kernel uses the scaled distance

.. math::

   h(x,x')
   =
   \left[
   \left(\frac{x_0-x'_0}{\rho_0}\right)^2
   +
   \left(\frac{x_1-x'_1}{\rho_1}\right)^2
   \right]^{1/2}.

The two lengthscales control smoothing along the two coordinate axes. Small
:math:`\rho_j` allows faster variation along coordinate :math:`j`. Large
:math:`\rho_j` makes the posterior vary more slowly along that coordinate.

Outputs
-------

The four panels contain the reference function, GP posterior mean, absolute
prediction error, and posterior standard deviation. Red dots mark observation
points. The error and posterior standard deviation panels should be read
together: a well-calibrated model should assign larger uncertainty in regions
with sparse observations or difficult extrapolation.

API points
----------

* ``gp.misc.designs.ldrandunif`` creates a space-filling observation design.
* ``gp.kernel.select_parameters_with_remap`` selects covariance parameters with
  the default REMAP criterion.
* ``model.predict`` returns posterior means and variances at the grid points.
* Convert backend arrays with ``gnp.to_np`` before sending them to Matplotlib
  functions that expect NumPy arrays.

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
   model, info = ex.gp.kernel.select_parameters_with_remap(model, xi, zi, info=True)
   zpm, zpv = model.predict(xi, zi, xt)

   xt_np = ex.gnp.to_np(xt)
   xi_np = ex.gnp.to_np(xi)
   zt_np = ex.gnp.to_np(zt).reshape(nt)
   zpm_np = ex.gnp.to_np(zpm).reshape(nt)
   zsd_np = np.sqrt(np.maximum(ex.gnp.to_np(zpv).reshape(nt), 0.0))

   fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
   data = [zt_np, zpm_np, np.abs(zpm_np - zt_np), zsd_np]
   titles = ["reference", "GP approximation", "absolute error", "posterior std"]
   cmaps = ["PiYG", "PiYG", "magma_r", "viridis"]
   for ax, z, title, cmap in zip(axes.flat, data, titles, cmaps):
       cs = ax.contourf(
           xt_np[:, 0].reshape(nt),
           xt_np[:, 1].reshape(nt),
           z,
           levels=30,
           cmap=cmap,
       )
       ax.plot(xi_np[:, 0], xi_np[:, 1], "ro", markersize=3)
       ax.set_title(title)
       ax.set_xlabel("$x_0$")
       ax.set_ylabel("$x_1$")
       fig.colorbar(cs, ax=ax, shrink=0.8)
   fig.tight_layout()

Script: ``examples/gpmp_example03_2d.py``

.. literalinclude:: ../../../examples/gpmp_example03_2d.py
   :language: python
   :linenos:
