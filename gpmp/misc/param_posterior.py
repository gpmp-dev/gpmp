"""
MCMC sampling from the negative log-posterior of the GP parameters.

sample_from_selection_criterion draws samples from the posterior distribution of
the GP parameters using a Metropolis–Hastings sampler. The negative log-posterior
is defined by info.selection_criterion.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import numpy as np
import gpmp.num as gnp
from gpmp.misc.designs import randunif
from gpmp.misc.mcmc import MHOptions, MetropolisHastings
from gpmp.misc.smc import run_smc_sampling


def sample_from_selection_criterion(
    info: object = None,
    custom_log_target: callable = None,
    param_initial_states: np.ndarray = None,
    random_init: bool = False,
    init_box: list = None,
    n_steps_total: int = 10_000,
    burnin_period: int = 4_000,
    n_chains: int = 2,
    n_pool: int = 2,
    silent: bool = False,
    show_progress: bool = True,
    plot_chains: bool = True,
    plot_empirical_distributions: bool = True,
):
    """
    Draw samples from the posterior distribution of the GP parameters.

    Parameters
    ----------
    info : object, optional
        Must have attributes:
          - covparam: initial GP parameters (1D array-like).
          - selection_criterion_nograd: returns the negative log-posterior.
        Not used if custom_log_target is provided.
    custom_log_target : callable, optional
        Custom log-target function. If provided, info must not be provided.
    param_initial_states : np.ndarray, optional
        Initial states (shape: (n_chains, dim)). Takes priority if info is provided.
    random_init : bool, optional
        If True, initialize chains randomly within init_box.
    init_box : list, optional
        Domain for random initialization: [lower, upper] or [[lower]*dim, [upper]*dim].
    n_steps_total : int
        Total number of MH steps.
    burnin_period : int
        Number of burn-in steps.
    n_chains : int
        Number of parallel chains.
    silent : bool, optional
        If True, suppress messages.
    show_progress : bool, optional
        If True, display progress information.
    plot_chains : bool, optional
        If True, plot chain traces after sampling.
    plot_empirical_distributions : bool, optional
        If True, plot empirical distributions after sampling.

    Returns
    -------
    samples : np.ndarray
        Post-burnin samples of shape (n_chains, n_steps_total - burnin_period, dim).
    mh : MetropolisHastings
        The sampler instance.
    """
    # --- Dimension Inference ---

    if custom_log_target is not None:
        # custom_log_target branch: info must not be provided.
        if info is not None:
            raise ValueError("When custom_log_target is provided, do not provide info.")
        log_target = custom_log_target
        if random_init:
            if init_box is None:
                raise ValueError("init_box must be provided when using random_init.")
            lower, _ = init_box
            if not isinstance(lower, (list, np.ndarray)):
                raise ValueError(
                    "With custom_log_target, init_box bounds must be lists."
                )
            dim = len(lower)
        else:
            if param_initial_states is None:
                raise ValueError(
                    "param_initial_states must be provided when random_init is False."
                )
            param_initial_states = np.asarray(param_initial_states)
            if param_initial_states.ndim == 1:
                param_initial_states = np.tile(param_initial_states, (n_chains, 1))
            if param_initial_states.ndim != 2:
                raise ValueError("param_initial_states must be a 1D or 2D array.")
            dim = param_initial_states.shape[1]
    else:
        # info branch
        if info is None:
            raise ValueError("Either info or custom_log_target must be provided.")
        if not hasattr(info, "covparam"):
            raise AttributeError("The 'info' object must have a 'covparam' attribute.")
        param_MAP = np.asarray(info.covparam)
        if param_MAP is None:
            raise ValueError("info.covparam is None.")
        if param_MAP.ndim != 1:
            raise ValueError("info.covparam must be a 1D array.")
        dim = param_MAP.shape[0]
        if not hasattr(info, "selection_criterion_nograd"):
            raise AttributeError(
                "The 'info' object must have a 'selection_criterion_nograd' attribute."
            )
        if info.selection_criterion_nograd is None:
            raise ValueError("info.selection_criterion_nograd is None.")
        if not callable(info.selection_criterion_nograd):
            raise TypeError("info.selection_criterion_nograd must be callable.")
        log_target = lambda p: -info.selection_criterion_nograd(p)

    # --- Chain State Initialization ---

    def rand_init(init_box, n_chains, dim):
        if not (isinstance(init_box, (list, tuple)) and len(init_box) == 2):
            raise ValueError("init_box must be of the form [lower, upper].")
        lower, upper = init_box
        if np.isscalar(lower) and np.isscalar(upper):
            lower = [lower] * dim
            upper = [upper] * dim
        elif isinstance(lower, (list, np.ndarray)) and len(lower) == 1:
            lower = [lower[0]] * dim
        elif isinstance(upper, (list, np.ndarray)) and len(upper) == 1:
            upper = [upper[0]] * dim
        else:
            lower = list(lower)
            upper = list(upper)
        if len(lower) != dim or len(upper) != dim:
            raise ValueError("init_box bounds must match dimension.")
        return randunif(dim, n_chains, [lower, upper])

    if random_init:
        if init_box is None:
            raise ValueError("init_box must be provided when using random_init.")
        param_initial_states = rand_init(init_box=init_box, n_chains=n_chains, dim=dim)
    else:
        if param_initial_states is None:
            if custom_log_target is not None:
                raise ValueError(
                    "param_initial_states must be provided when random_init is False."
                )
            # Use info.covparam from the info branch; already inferred as 1D.
            param_initial_states = np.tile(param_MAP, (n_chains, 1))
        else:
            param_initial_states = np.asarray(param_initial_states)
            if param_initial_states.ndim == 1:
                param_initial_states = np.tile(param_initial_states, (n_chains, 1))
            elif param_initial_states.ndim != 2:
                raise ValueError("param_initial_states must be a 1D or 2D array.")

    if param_initial_states.shape != (n_chains, dim):
        raise ValueError(f"param_initial_states must have shape ({n_chains}, {dim}).")

    if n_steps_total < burnin_period:
        raise ValueError("n_steps_total must be greater than burnin_period.")

    # --- Set Up and Run Metropolis-Hastings ---

    show_prog = show_progress and not silent
    options = MHOptions(
        dim=dim,
        n_chains=n_chains,
        target_acceptance=0.3,
        proposal_distribution_param_init=0.1 * np.ones(dim),
        adaptation_method="Haario",
        adaptation_interval=50,
        haario_adapt_factor_burnin_phase=1.0,
        haario_adapt_factor_sampling_phase=0.5,
        freeze_adaptation=False,
        discard_burnin=False,
        n_pool=n_pool,
        show_global_progress=show_prog,
        init_msg=(
            None
            if silent
            else "Sampling from posterior distribution of GP parameters..."
        ),
    )

    mh = MetropolisHastings(log_target=log_target, options=options)
    param_samples = mh.scheduler(
        chains_state_initial=param_initial_states,
        n_steps_total=n_steps_total,
        burnin_period=burnin_period,
    )

    if not silent:
        print("\n")
        mh.check_acceptance_rates(burnin_period=mh.burnin_period)
        mh.check_convergence_gelman_rubin(burnin_period=mh.burnin_period)

    if plot_chains:
        mh.plot_chains()
    if plot_empirical_distributions:
        mh.plot_empirical_distributions(smooth=False)

    samples_post_burnin = param_samples[:, mh.burnin_period :, :]
    return samples_post_burnin, mh




def sample_from_selection_criterion_smc(
    info: object = None,
    custom_log_target: callable = None,
    init_box: list = None,
    n_particles: int = 1000,
    initial_temperature: float = 1e6,
    final_temperature: float = 1.0,
    min_ess_ratio: float = 0.5,
    mh_steps: int = 20,
    max_stages: int = 50,
    debug: bool = False,
    plot_marginals: bool = False,
    plot_particles: bool = False,
):
    """
    Run SMC sampling for GP model parameters from the posterior distribution.

    Constructs a tempered log-probability density:

        logpdf(x, T) = -f(x) / T

    where f is either info.selection_criterion_nograd or custom_log_target.
    Particles that fall outside the search domain (equal to init_box) are penalized
    with -∞.

    Parameters
    ----------
    info : object, optional
        Must have attributes:
          - covparam: initial GP parameters.
          - selection_criterion_nograd: returns the negative log-posterior.
        Not used if custom_log_target is provided.
    custom_log_target : callable, optional
        Custom function to compute the negative log-posterior.
        If provided, info must not be provided.
    init_box : list
        Domain box for particle initialization, used as the search domain.
    n_particles : int, optional
        Number of particles. Default is 1000.
    initial_temperature : float, optional
        Starting temperature for the SMC process.
    final_temperature : float, optional
        Final temperature for the SMC process.
    min_ess_ratio : float, optional
        Minimum ratio of effective sample size to total particles.
    max_stages : int, optional
        Maximum number of stages before stopping.
    debug : bool, optional
        If True, print debug information.
    plot_marginals : bool, optional
        If True, display pairwise scatter plots of the particle distribution.
    plot_particles : bool, optional
        If True, display a matrix plot of the particle distribution.

    Returns
    -------
    particles : ndarray
        Final particle positions from the SMC process.
    smc : SMC
        The SMC instance with additional logs and diagnostics.
    """

    def create_logpdf_temp(f, search_domain_box=None):
        # Precompute domain bounds as numpy arrays
        if search_domain_box is not None:
            lower_bound = gnp.array(search_domain_box[0])
            upper_bound = gnp.array(search_domain_box[1])

        def logpdf_temp(x, temperature):
            if x.ndim == 1:
                if search_domain_box is not None:
                    # Check domain membership for a single particle.
                    if gnp.any(x < lower_bound) or gnp.any(x > upper_bound):
                        return -gnp.inf
                return -f(x) / temperature
            elif x.ndim == 2:
                # Vectorized boundary check: for each particle, check all coordinates.
                mask = gnp.all(x >= lower_bound, axis=1) & gnp.all(
                    x <= upper_bound, axis=1
                )
                # Compute f for each particle (using list comprehension if f is not vectorized)
                values = gnp.array([f(xi) for xi in x])
                results = -values / temperature
                # Set out-of-domain particles to -inf.
                results[~mask] = -gnp.inf
                return results
            else:
                raise ValueError("Input array x must be 1D or 2D.")

        return logpdf_temp

    if custom_log_target is not None:
        if info is not None:
            raise ValueError("When custom_log_target is provided, do not provide info.")
        f = custom_log_target
    else:
        if info is None:
            raise ValueError("Either info or custom_log_target must be provided.")
        if not hasattr(info, "covparam"):
            raise AttributeError("The 'info' object must have a 'covparam' attribute.")
        if not hasattr(info, "selection_criterion_nograd"):
            raise AttributeError(
                "The 'info' object must have a 'selection_criterion_nograd' attribute."
            )
        if not callable(info.selection_criterion_nograd):
            raise TypeError("'info.selection_criterion_nograd' must be callable.")
        f = info.selection_criterion_nograd

    # Use init_box as the search domain.
    search_domain_box = init_box

    logpdf_temp = create_logpdf_temp(f, search_domain_box=search_domain_box)

    particles, smc_instance = run_smc_sampling(
        logpdf_parameterized_function=logpdf_temp,
        initial_logpdf_param=initial_temperature,
        target_logpdf_param=final_temperature,
        compute_next_logpdf_param_method="ess",
        min_ess_ratio=min_ess_ratio,
        init_box=init_box,
        n_particles=n_particles,
        mh_steps=mh_steps,
        debug=debug,
        plot_empirical_distributions=plot_marginals,
    )

    return particles, smc_instance
