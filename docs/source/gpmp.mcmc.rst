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

Experimental samplers are intentionally not documented here until their API is
stable.

Posterior parameter sampling
----------------------------

.. autofunction:: gpmp.mcmc.sample_from_selection_criterion_mh

.. autofunction:: gpmp.mcmc.sample_from_selection_criterion_nuts

.. autofunction:: gpmp.mcmc.sample_from_selection_criterion_smc

.. autofunction:: gpmp.mcmc.get_log_target_values

Metropolis-Hastings
-------------------

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
