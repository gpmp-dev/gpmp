gpmp.mcmc module
================

The ``mcmc`` module provides posterior covariance-parameter sampling helpers
and lower-level samplers. Samples, particles, and log-probability traces are
backend-native :mod:`gpmp.num` objects unless a function explicitly documents
conversion. For GP parameter posterior exploration, prefer the
``sample_from_selection_criterion_*`` functions: they consume an optimization
``info`` object returned by :mod:`gpmp.kernel` selection functions, or a
standalone negative log-posterior callable.

Target convention
-----------------

The posterior helpers treat the selection criterion ``J(theta)`` as a negative
log-density. The target log-density is ``-J(theta)`` for NUTS and
``-J(theta) / temperature`` for tempered MH/SMC. ``sampling_box`` imposes hard
bounds on the target support. ``init_box`` is only an initialization box.

Initialization convention
-------------------------

For MH and NUTS, pass ``param_initial_states`` explicitly or set
``random_init=True`` with ``init_box``. For SMC, ``init_box`` is mandatory.
The parameter dimension is inferred from ``info.covparam``, initial states, or
the supplied box.

Experimental samplers are omitted from the public API page until their call
signature and return objects are fixed.

Posterior parameter sampling
----------------------------

.. autofunction:: gpmp.mcmc.sample_from_selection_criterion_mh

.. autofunction:: gpmp.mcmc.sample_from_selection_criterion_nuts

.. autofunction:: gpmp.mcmc.sample_from_selection_criterion_smc

.. autofunction:: gpmp.mcmc.get_log_target_values

Metropolis-Hastings
-------------------

GPmp's Metropolis-Hastings implementation is a multi-chain Gaussian
random-walk sampler. The default public wrapper
``sample_from_selection_criterion_mh`` uses Haario empirical-covariance
adaptation with ``adaptation_interval=50`` and target acceptance rate 0.3.
The ``n_pool`` option controls how chains are grouped when empirical proposal
covariances are estimated. The public wrapper keeps adaptation active after
burn-in. Use the lower-level ``MHOptions`` object if a frozen post-burn-in
proposal is required. The lower-level sampler can also use a Robbins-Monro
scale adaptation by setting ``adaptation_method="RM"`` in ``MHOptions``.

The sampler stores three traces after ``scheduler`` has run: ``x`` for chain
states, ``accept`` for acceptance indicators, and ``log_target_values`` for
cached target log-densities. Use ``get_log_target_values`` to retrieve the
stored log-density trace without recomputing the criterion.

.. autoclass:: gpmp.mcmc.MHOptions

.. autoclass:: gpmp.mcmc.MetropolisHastings

.. autofunction:: gpmp.mcmc.sample_multivariate_normal_with_jitter

NUTS
----

.. autoclass:: gpmp.mcmc.NUTSOptions

.. autofunction:: gpmp.mcmc.nuts_sample

.. autofunction:: gpmp.mcmc.nuts_transition

.. autofunction:: gpmp.mcmc.plot_nuts_diagnostics

Sequential Monte Carlo
----------------------

.. autoclass:: gpmp.mcmc.ParticlesSetConfig

.. autoclass:: gpmp.mcmc.SMCConfig

.. autofunction:: gpmp.mcmc.run_smc_sampling

.. autofunction:: gpmp.mcmc.log_indicator_density

.. autofunction:: gpmp.mcmc.run_subset_simulation
