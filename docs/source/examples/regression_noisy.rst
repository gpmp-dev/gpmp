Noisy observations
==================

This example models noisy observations and predicts the latent, noise-free
process. It shows how a covariance function can distinguish observation inputs
from latent prediction targets and add a noise term only where appropriate.

What this example does
----------------------

The script generates latent function values and noisy observations. The model
uses an input flag to identify observation points, so the covariance can add the
observation-noise variance on the diagonal for observations while leaving
latent prediction points noise-free.

Mathematical description
------------------------

The latent process is still :math:`Z \sim \mathcal{GP}(m, k_\theta)`, but the
observed random variable is noisy:

.. math::

   Z_i^{\mathrm{obs}} = Z(x_i) + \varepsilon_i,
   \qquad
   \varepsilon_i \sim \mathcal{N}(0, \tau_i^2).

The covariance used for the observations is therefore

.. math::

   \operatorname{cov}(Z_i^{\mathrm{obs}}, Z_j^{\mathrm{obs}})
   =
   k_\theta(x_i, x_j) + \tau_i^2 \mathbf{1}_{i=j}.

Prediction targets use the latent random variable :math:`Z_t=Z(x_t)`, so the
cross-covariance between observations and targets is
:math:`k_\theta(x_i,x_t)`, without an observation-noise term.

In the script, ``zi`` stores realizations of
:math:`Z_i^{\mathrm{obs}}`.

Outputs
-------

Observation values are noisy and therefore need not lie exactly on the posterior
mean. The posterior distribution represents the latent process, not the noisy
data-generating variable. Compared with the noise-free examples, uncertainty
remains nonzero at observation locations because the observations are imperfect
measurements.

API points
----------

* Noisy-observation models are usually implemented by writing a covariance
  function that adds a nugget or input-dependent noise term.
* ``model.predict`` still returns posterior mean and variance for the target
  inputs supplied by the user.
* The distinction between noisy observations and latent targets is a modeling
  convention encoded in the covariance function, not a separate GPmp model
  class.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example06_1d_regression as ex

   xt, zt, xi, zi, zpm, zpv = ex.main()
   ex.visualize(xt, zt, xi, zi, zpm, zpv)

Script: ``examples/gpmp_example06_1d_regression.py``

.. literalinclude:: ../../../examples/gpmp_example06_1d_regression.py
   :language: python
   :linenos:
