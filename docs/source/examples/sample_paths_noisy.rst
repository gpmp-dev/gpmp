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

Mathematical description
------------------------

The noisy observation random variables are

.. math::

   Z_i^{\mathrm{obs}} = Z(x_i) + \varepsilon_i,
   \qquad
   \varepsilon_i \sim \mathcal{N}(0,\tau_i^2),

with :math:`\tau_i^2` depending on the observation location. The array ``zi``
stores realized noisy values :math:`z_i^{\mathrm{obs}}`. The conditional
distribution of the latent process at prediction points is computed with
the observation covariance. Let :math:`K_{ii}` be the covariance matrix with
entries :math:`k_\theta(x_i^a,x_i^b)`. Then

.. math::

   K_{ii}^{\mathrm{obs}}
   =
   K_{ii} + \operatorname{diag}(\tau_1^2,\ldots,\tau_n^2).

The conditional paths target :math:`Z_t=Z(x_t)`, not
:math:`Z_t^{\mathrm{obs}}`. They therefore need not pass through the noisy
realized observations.

Outputs
-------

The first output is the latent posterior distribution. The following outputs are
conditional paths and posterior simulations. Because the observations are noisy,
conditional paths are not required to pass exactly through observation values.
Instead, they remain statistically compatible with the noise model.

API points
----------

* Heteroscedastic noise is represented through the covariance construction.
* Conditional simulation remains a GP conditioning problem once the covariance
  has encoded the correct observation-noise structure.
* Compare with :doc:`sample_paths` to see the practical difference between
  noise-free and noisy conditioning.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example11_sample_paths_noisy_obs as ex

   ex.main()

Script: ``examples/gpmp_example11_sample_paths_noisy_obs.py``

.. literalinclude:: ../../../examples/gpmp_example11_sample_paths_noisy_obs.py
   :language: python
   :linenos:
