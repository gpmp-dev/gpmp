Noisy conditional sample paths
==============================

This example generates conditional sample paths in a regression setting with
heteroscedastic observation noise. It extends the previous sample-path example
by distinguishing noisy measurements from the latent process.

What this example does
----------------------

The script defines input-dependent observation noise, computes the latent
posterior distribution, and draws conditional paths compatible with noisy
observations. The covariance includes the observation-noise term where needed,
while predictions and paths target the latent process.

How to read the figures
-----------------------

The first figure shows the latent posterior distribution. The following figures
show conditional paths and posterior simulations. Because the observations are
noisy, conditional paths are not required to pass exactly through observation
values. Instead, they remain statistically compatible with the noise model.

API points
----------

* Heteroscedastic noise is represented through the covariance construction.
* Conditional simulation remains a GP conditioning problem once the covariance
  has encoded the correct observation-noise structure.
* Compare this page with :doc:`sample_paths` to see the practical difference
  between noise-free and noisy conditioning.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example11_sample_paths_noisy_obs as ex

   ex.main()

Script: ``examples/gpmp_example11_sample_paths_noisy_obs.py``

.. literalinclude:: ../../../examples/gpmp_example11_sample_paths_noisy_obs.py
   :language: python
   :linenos:
