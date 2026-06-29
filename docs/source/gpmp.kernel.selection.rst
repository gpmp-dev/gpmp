gpmp.kernel parameter selection
===============================

Likelihood objectives and parameter-selection wrappers operate on
:class:`gpmp.core.Model` objects and backend-native arrays. Selection
procedures modify ``model.covparam`` and return ``(model, info_ret)``.

Likelihood objectives
---------------------

The following functions return negative log-likelihood or negative restricted
log-likelihood values. They are objective functions to minimize over
covariance parameters.

negative_log_likelihood_zero_mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.negative_log_likelihood_zero_mean

negative_log_likelihood
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.negative_log_likelihood

negative_log_restricted_likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.negative_log_restricted_likelihood

Generic selection helpers
-------------------------

These helpers build criterion callables, connect them to SciPy optimizers, and
provide generic selection/update procedures for custom criteria.

make_selection_criterion_with_gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.make_selection_criterion_with_gradient

autoselect_parameters
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.autoselect_parameters

select_parameters_with_criterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.select_parameters_with_criterion

update_parameters_with_criterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.update_parameters_with_criterion

Specific selection procedures
-----------------------------

These are the main public procedures for common criteria. ``select_*`` uses an
explicit initial covariance vector when provided. ``update_*`` starts from the
current ``model.covparam`` when it exists.

select_parameters_with_ml_constant_mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.select_parameters_with_ml_constant_mean

update_parameters_with_ml_constant_mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.update_parameters_with_ml_constant_mean

select_parameters_with_reml
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.select_parameters_with_reml

update_parameters_with_reml
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.update_parameters_with_reml

select_parameters_with_remap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.select_parameters_with_remap

update_parameters_with_remap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.update_parameters_with_remap

select_parameters_with_remap_gaussian_logsigma2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.select_parameters_with_remap_gaussian_logsigma2

update_parameters_with_remap_gaussian_logsigma2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.update_parameters_with_remap_gaussian_logsigma2

select_parameters_with_remap_gaussian_logsigma2_and_logrho_prior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.select_parameters_with_remap_gaussian_logsigma2_and_logrho_prior

update_parameters_with_remap_gaussian_logsigma2_and_logrho_prior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gpmp.kernel.update_parameters_with_remap_gaussian_logsigma2_and_logrho_prior
