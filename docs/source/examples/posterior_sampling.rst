Posterior parameter sampling
============================

This example demonstrates REMAP-based parameter selection followed by posterior
sampling of GP covariance parameters. It is intended for cases where point
estimates of ``covparam`` are not enough and uncertainty over covariance
parameters should be explored explicitly. Here "posterior" means the
unnormalized density obtained by exponentiating the selected criterion. With a
REMAP criterion, this is the restricted-likelihood posterior induced by the
chosen prior.

What this example does
----------------------

The rendered preview performs REMAP parameter selection and plots the posterior
prediction. The full script continues by sampling covariance parameters with
the posterior-sampling helpers in :mod:`gpmp.mcmc`. The available public
wrappers cover adaptive Metropolis-Hastings, NUTS, and tempered Sequential
Monte Carlo (SMC).

Mathematical target
-------------------

The samplers work on the covariance-parameter vector
:math:`\theta=\mathrm{covparam}`. With the REMAP criterion used in the script,
the target is defined by an unnormalized density on :math:`\theta`. GPmp uses
the selection criterion as a negative log density:

.. math::

   J(\theta)
   =
   J_{\mathrm{REMAP}}(\theta)
   =
   -\log \widetilde\pi(\theta\mid z_i),

where :math:`\widetilde\pi` denotes the target density up to a normalizing
constant. Equivalently,

.. math::

   \log \widetilde\pi(\theta\mid z_i)
   =
   -J_{\mathrm{REMAP}}(\theta).

The normalizing constant is not needed by MH, NUTS, or SMC. When a
``sampling_box`` is supplied, GPmp truncates the target support: outside the
box the log density is :math:`-\infty`. The ``init_box`` controls
initialization only. For SMC it is mandatory because the initial particle cloud
must be drawn from a user-specified region.

The sampled object is :math:`\theta`, not a conditional sample path of
:math:`Z`. For each sampled :math:`\theta`, one may compute a GP conditional
distribution for :math:`Z_t=Z(x_t)`. This separates uncertainty on covariance
parameters from conditional GP uncertainty at fixed covariance parameters.

Sampling methods
----------------

Metropolis-Hastings
    The public MH helper uses the lower-level
    :class:`gpmp.mcmc.MetropolisHastings` sampler. It runs several chains in
    parallel. For chain :math:`c`, the default proposal is a Gaussian
    random-walk proposal

    .. math::

       \theta' = \theta_{c,t} + \eta_{c,t},
       \qquad
       \eta_{c,t}\sim\mathcal N(0,\Sigma_c).

    With the default symmetric proposal, the log acceptance ratio is

    .. math::

       \log a
       =
       \log \widetilde\pi(\theta'\mid z_i)
       -
       \log \widetilde\pi(\theta_{c,t}\mid z_i).

    If ``symmetric=False`` is used in the lower-level sampler, GPmp adds the
    usual reverse-proposal correction from the Metropolis-Hastings ratio
    :cite:p:`metropolis1953equation,hastings1970monte`.

    The wrapper only needs criterion values. It builds

    .. math::

       \log \widetilde\pi_T(\theta\mid z_i)
       =
       -J(\theta) / T,

    where ``temperature=T``. Values :math:`T>1` flatten the target and may help
    exploration. Samples at :math:`T\ne 1` are samples from the tempered
    target, not from the nominal posterior.

    The default adaptation method in the public wrapper is Haario adaptation
    :cite:p:`haario2001adaptive`. Every ``adaptation_interval`` iterations
    (50 by default in the wrapper), GPmp computes a block acceptance rate
    :math:`r_c` for each chain. It also computes empirical covariance matrices
    from the recent chain states. The ``n_pool`` argument groups chains before
    estimating these covariances. If :math:`\widehat C_g` is the empirical
    covariance for the group containing chain :math:`c`, the proposal covariance
    is updated as

    .. math::

       s_c \leftarrow s_c
       \exp\{\gamma(r_c-r_\star)\},
       \qquad
       \Sigma_c \leftarrow s_c\,\widehat C_g + \varepsilon I.

    Here :math:`r_\star` is the target acceptance rate (0.3 in the public
    wrapper), :math:`\gamma` is the phase-dependent adaptation factor, and
    :math:`\varepsilon I` is a small diagonal stabilization term. The
    public wrapper currently sets ``freeze_adaptation=False``: adaptation is
    used during burn-in and continues during the sampling phase, with a smaller
    sampling-phase adaptation factor. The lower-level ``MHOptions`` object can
    be used directly when a frozen post-burn-in proposal is preferred. The
    lower-level covariance helper also supports the common
    :math:`2.38^2/d` scaling, motivated by optimal-scaling results for
    random-walk Metropolis algorithms on regular high-dimensional targets
    :cite:p:`roberts1997weak`. This scaling is a heuristic for practical GP
    parameter posteriors, not a guarantee.

    The lower-level sampler also implements a Robbins-Monro scale update
    :cite:p:`robbins1951stochastic`, selected with
    ``adaptation_method="RM"``. In that mode, scalar or diagonal proposal
    scales are multiplied by
    :math:`\exp\{\gamma_t(r_c-r_\star)\}`.

    The MH object stores the full chain states in ``mh.x``, acceptance
    indicators in ``mh.accept``, and cached log-target values in
    ``mh.log_target_values``. The helper
    :func:`gpmp.mcmc.get_log_target_values` extracts these values, optionally
    after burn-in. GPmp evaluates the log target inside each MH step and reuses
    the cached value of the current state at the next step, because a criterion
    evaluation can be costly.

NUTS
    NUTS is an adaptive Hamiltonian Monte Carlo method
    :cite:p:`neal2011hmc,hoffman2014nuts`. It uses gradients of
    :math:`J(\theta)` to propose distant moves while reducing random-walk
    behavior. It can be effective when the posterior is smooth and moderately
    well scaled, but each transition may require many criterion and gradient
    evaluations.

Sequential Monte Carlo
    SMC evolves a population of particles through a sequence of tempered
    targets :cite:p:`delmoral2006smc`. GPmp uses targets proportional to
    :math:`\exp[-J(\theta)/T]`, starting from a high temperature and ending at
    the requested final temperature. Resampling and MH rejuvenation steps are
    used to keep particles in high-density regions. SMC is useful for comparing
    separated posterior regions, but it depends strongly on the initialization
    region and the tempering schedule.

How to interpret the output
---------------------------

REMAP provides a regularized point estimate and a prior definition. Sampling
then explores nearby and competing covariance-parameter values according to the
same criterion interpreted as a negative log density. Chains or particles
concentrated near a small region indicate locally well-identified covariance
parameters. Broad, skewed, or separated clouds indicate weak identification or
competing covariance explanations.

For MH and NUTS, inspect chain traces, acceptance rates, and log-probability
traces before interpreting summaries. For SMC, inspect the final particle
cloud, effective sample size behavior, and whether resampling has collapsed
particles into a small number of distinct states.

API points
----------

* ``select_parameters_with_remap_gaussian_logsigma2_and_logrho_prior`` selects
  a regularized covariance vector and defines criterion callables in ``info``.
* Posterior samplers in :mod:`gpmp.mcmc` use ``info.selection_criterion_nograd``
  as a negative log density when possible. NUTS requires the differentiable
  ``info.selection_criterion``.
* ``sampling_box`` defines hard bounds for the sampling target. ``init_box`` is
  only an initialization region, except that SMC requires it to draw the
  initial particles.
* Use sampler diagnostics before interpreting posterior parameter samples.

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
