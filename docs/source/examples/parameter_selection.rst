ML / REML / REMAP parameter selection
=====================================

This example compares the main covariance-parameter selection criteria used in
GPmp: maximum likelihood (ML), restricted maximum likelihood (REML), and
restricted maximum a posteriori (REMAP). The scripts use the same
one-dimensional interpolation setup so that only the selection criterion
changes.

What this example does
----------------------

The rendered preview uses REMAP, selects covariance parameters, and plots the
posterior prediction. The full scripts below expose the corresponding REMAP,
REML, and ML procedures. All three procedures optimize a scalar criterion over
``covparam`` but differ in how they handle mean parameters and prior
regularization.

Mathematical criteria
---------------------

Let :math:`P_i` denote the matrix of mean basis functions evaluated at the
observation points, and let :math:`K_{ii}(\theta)` be the covariance matrix.
For a linear mean :math:`m_\beta(x)=h(x)^\top\beta`, ML profiles out
:math:`\beta`. The profiled estimator is

.. math::

   \widehat\beta_\theta
   =
   \left(P_i^\top K_{ii}^{-1}P_i\right)^{-1}
   P_i^\top K_{ii}^{-1}z_i.

Up to constants independent of :math:`\theta`, the ML criterion is

.. math::

   J_{\mathrm{ML}}(\theta)
   =
   \frac12 \log |K_{ii}|
   +
   \frac12 (z_i-P_i\widehat\beta_\theta)^\top
   K_{ii}^{-1}
   (z_i-P_i\widehat\beta_\theta).

REML is computed from contrasts that remove the linear mean. Let
:math:`W_i\in\mathbb{R}^{n\times(n-q)}` have full column rank and satisfy
:math:`W_i^\top P_i=0`. GPmp builds :math:`W_i` from a complete QR
factorization of :math:`P_i`. With

.. math::

   G_\theta = W_i^\top K_{ii}(\theta) W_i,

the REML criterion implemented in GPmp is, up to constants independent of
:math:`\theta`,

.. math::

   J_{\mathrm{REML}}(\theta)
   =
   \frac12 \log |G_\theta|
   +
   \frac12 (W_i^\top z_i)^\top
   G_\theta^{-1}
   (W_i^\top z_i).

REMAP adds prior regularization on covariance parameters:

.. math::

   J_{\mathrm{REMAP}}(\theta)
   =
   J_{\mathrm{REML}}(\theta) - \log \pi(\theta).

How to interpret the comparison
-------------------------------

ML optimizes the likelihood after estimating mean parameters. REML accounts for
the degrees of freedom used by the mean and is often preferable when the mean is
unknown. See :cite:p:`stein1999kriging`. REMAP adds prior terms to the
restricted likelihood. This may stabilize poorly identified variance or
lengthscale parameters. Different criteria can lead to different posterior
uncertainty, even when the posterior mean looks similar.

API points
----------

* ``select_parameters_with_reml`` is the standard restricted-likelihood helper.
* ``select_parameters_with_remap`` and the more explicit REMAP variants add
  prior regularization.
* The returned ``info`` object records the selected ``covparam``, optimization
  status, objective history, and criterion callables.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example20_1d_interpolation_variation_remap as ex

   xt, zt, xi, zi = ex.generate_data()
   model = ex.gp.core.Model(ex.constant_mean, ex.kernel)
   model, info = ex.gp.kernel.select_parameters_with_remap(model, xi, zi, info=True)
   zpm, zpv = model.predict(xi, zi, xt)
   ex.visualize_results(xt, zt, xi, zi, zpm, zpv)

REMAP
-----

Script: ``examples/gpmp_example20_1d_interpolation_variation_remap.py``

.. literalinclude:: ../../../examples/gpmp_example20_1d_interpolation_variation_remap.py
   :language: python
   :linenos:

REML
----

Script: ``examples/gpmp_example21_1d_interpolation_variation_reml.py``

.. literalinclude:: ../../../examples/gpmp_example21_1d_interpolation_variation_reml.py
   :language: python
   :linenos:

ML
--

Script: ``examples/gpmp_example22_1d_interpolation_variation_ml.py``

.. literalinclude:: ../../../examples/gpmp_example22_1d_interpolation_variation_ml.py
   :language: python
   :linenos:
