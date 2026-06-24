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

How to read the figure
----------------------

The plotted observation values are noisy and therefore need not lie exactly on
the posterior mean. The posterior distribution represents the latent process,
not the noisy data-generating variable. Compared with the noise-free examples,
uncertainty remains nonzero at observation locations because the observations
are imperfect measurements.

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
