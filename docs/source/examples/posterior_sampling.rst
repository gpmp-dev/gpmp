Posterior parameter sampling
============================

This example demonstrates REMAP-based parameter selection followed by posterior
sampling of GP covariance parameters. It is intended for workflows where point
estimates of ``covparam`` are not enough and uncertainty over covariance
parameters should be explored explicitly.

What this example does
----------------------

The rendered preview performs REMAP parameter selection and plots the posterior
prediction. The full script continues by sampling the posterior distribution of
covariance parameters with MCMC methods. For documentation build time, the
rendered preview stops before the expensive sampling loop.

How to interpret the workflow
-----------------------------

REMAP provides a stable starting point and prior definition. Posterior sampling
then explores nearby and competing covariance-parameter values according to the
selection criterion interpreted as a negative log density. The resulting chains
or particles can be used to assess whether the selected covariance parameters
are well identified.

API points
----------

* ``select_parameters_with_remap_gaussian_logsigma2_and_logrho_prior`` selects
  a regularized covariance vector and defines criterion callables in ``info``.
* Posterior samplers in :mod:`gpmp.mcmc` use ``info.selection_criterion_nograd``
  as a negative log-density when possible.
* Use sampling diagnostics before interpreting posterior parameter samples.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example23_1d_interpolation_posterior_sampling as ex

   xt, zt, xi, zi = ex.generate_data()
   model = ex.gp.core.Model(ex.constant_mean, ex.kernel)
   model, info = (
       ex.gp.kernel.select_parameters_with_remap_gaussian_logsigma2_and_logrho_prior(
           model, xi, zi, info=True
       )
   )
   zpm, zpv = model.predict(xi, zi, xt)
   ex.visualize_results(xt, zt, xi, zi, zpm, zpv)

Script: ``examples/gpmp_example23_1d_interpolation_posterior_sampling.py``

.. literalinclude:: ../../../examples/gpmp_example23_1d_interpolation_posterior_sampling.py
   :language: python
   :linenos:
