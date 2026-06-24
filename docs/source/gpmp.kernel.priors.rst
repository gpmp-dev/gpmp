gpmp.kernel priors
==================

Prior terms are used by REMAP selection procedures. Prior functions return
log-prior values. Posterior objective wrappers combine these terms with the
negative restricted likelihood and return negative log posterior values to
minimize.

Gaussian prior on log variance
------------------------------

``log_prior_gaussian_logsigma2`` defines a Gaussian prior on
``log(sigma2)``. In the REMAP helpers, the center is usually anchored at a
reference covariance vector, while the scale is calibrated by the ``gamma`` and
``sigma2_coverage`` hyperparameters.

log_prior_gaussian_logsigma2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.log_prior_gaussian_logsigma2

Log lengthscale barrier and linear penalty
------------------------------------------

The log-lengthscale prior is designed to prevent degenerate noise-model-like
solutions where lengthscales become too small, while also penalizing overly
large lengthscales. It is expressed on ``logrho = log(rho)``. GPmp stores
``-log(rho)`` in ``covparam``.

neglog_f_logrho
~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.neglog_f_logrho

log_prior_logrho_barrier_linear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.log_prior_logrho_barrier_linear

compute_logrho_min_from_xi
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.compute_logrho_min_from_xi

Bounds helpers
--------------

empirical_bounds_factory
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.empirical_bounds_factory
