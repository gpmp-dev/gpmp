ML / REML / REMAP parameter selection
=====================================

This page compares the main covariance-parameter selection criteria used in
GPmp: maximum likelihood (ML), restricted maximum likelihood (REML), and
restricted maximum a posteriori (REMAP). The scripts use the same
one-dimensional interpolation setup so that only the selection criterion
changes.

What this example does
----------------------

The rendered preview uses REMAP, selects covariance parameters, and plots the
posterior prediction. The full scripts below expose the corresponding REMAP,
REML, and ML workflows. All three procedures optimize a scalar criterion over
``covparam`` but differ in how they handle mean parameters and prior
regularization.

How to interpret the comparison
-------------------------------

ML optimizes the likelihood after estimating mean parameters. REML accounts for
the degrees of freedom used by the mean and is often preferable when the mean is
unknown. REMAP adds prior terms to the restricted likelihood; this can stabilize
poorly identified variance or lengthscale parameters. Different criteria can
lead to different posterior uncertainty, even when the posterior mean looks
similar.

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
