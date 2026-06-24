Conditional sample paths
========================

This example generates Gaussian process sample paths conditional on
noise-free observations. Conditional paths support simulation studies,
uncertainty visualization, and algorithms that require full random functions
instead of only posterior means and variances.

What this example does
----------------------

The script builds a one-dimensional GP, conditions it on observations, and draws
sample paths from the posterior process. The conditional paths are generated so
that they are consistent with the observations and with the selected covariance
model.

Mathematical object
-------------------

The script first draws prior paths from
:math:`Z \sim \mathcal{GP}(m, k_\theta)`. It then transforms those paths into
conditional paths. In the noise-free case, the target distribution is

.. math::

   Z_t\mid Z_i=z_i
   \sim
   \mathcal{N}\left(
       m_t + k_{ti} k_{ii}^{-1}(z_i-m_i),
       k_{tt} - k_{ti} k_{ii}^{-1} k_{it}
   \right).

The conditioning step changes each prior path so that the values at the
observation points match the realized data :math:`z_i`.

Outputs
-------

The sample paths pass through the noise-free observations. Away from the
observations, paths spread according to posterior uncertainty. The posterior
mean and uncertainty envelope summarize the conditional distribution, while the
sample paths show possible function realizations.

API points
----------

* Conditional sample paths use the same covariance model as prediction.
* Use sample paths when an algorithm needs random functions, beyond pointwise
  marginal uncertainty.
* The noise-free setting forces conditional paths to match observations exactly.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example10_sample_paths as ex

   ex.main()

Script: ``examples/gpmp_example10_sample_paths.py``

.. literalinclude:: ../../../examples/gpmp_example10_sample_paths.py
   :language: python
   :linenos:
