1D interpolation
================

This example builds a one-dimensional noise-free Gaussian process model and
selects covariance parameters by restricted maximum likelihood (REML). It is the
recommended first complete example because it shows the basic GPmp sequence:
construct a model, select ``covparam``, predict, and plot the result.

What this example does
----------------------

The script creates observation points ``xi`` and observations ``zi`` from a
known reference function. It defines a constant-mean GP with an anisotropic
Matern covariance and calls ``gp.kernel.select_parameters_with_reml``. The
selected covariance parameters are stored in ``model.covparam`` and are then
used by ``model.predict`` on a dense grid ``xt``.

Mathematical description
------------------------

The noise-free observation model is

.. math::

   Z_i = Z(x_i),
   \qquad
   Z(x) = p(x)^\top \beta + Z_0(x),
   \qquad
   Z_0 \sim \mathcal{GP}(0, k_\theta),

where the trend coefficients :math:`\beta` are unknown. In this example,
``constant_mean`` gives a constant trend basis, so :math:`p(x)=1`.

After selecting :math:`\theta`, ``model.predict(xi, zi, xt)`` computes the
universal, or intrinsic, kriging predictor :cite:p:`stein1999kriging,chiles1999geostatistics`.
Let :math:`P_i` be the trend matrix at the observation points, :math:`P_t` the
trend matrix at prediction points, :math:`K_{ii}` the observation covariance
matrix, :math:`K_{it}` the covariance block between observations and
prediction points, and :math:`K_{tt}` the covariance block at prediction
points. The kriging weights :math:`\Lambda` and Lagrange multipliers
:math:`M` solve

.. math::

   \begin{bmatrix}
   K_{ii} & P_i \\
   P_i^\top & 0
   \end{bmatrix}
   \begin{bmatrix}
   \Lambda \\
   M
   \end{bmatrix}
   =
   \begin{bmatrix}
   K_{it} \\
   P_t^\top
   \end{bmatrix}.

The posterior mean and covariance are then

.. math::

   \mathrm{E}(Z_t \mid Z_i=z_i) = \Lambda^\top z_i,

and

.. math::

   \mathrm{cov}(Z_t \mid Z_i=z_i)
   =
   K_{tt}
   -
   \begin{bmatrix}
   \Lambda \\
   M
   \end{bmatrix}^\top
   \begin{bmatrix}
   K_{it} \\
   P_t^\top
   \end{bmatrix}.

The plotted uncertainty envelope is built from the diagonal of the conditional
covariance.

Outputs
-------

The displayed quantities are the reference function, the observations, the
posterior mean, and the posterior uncertainty envelope. Because the observations
are treated as noise-free, the posterior mean interpolates the observed values.
The uncertainty is small near observations and larger away from them.

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
