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
from gpmp.misc.mcmc import MHOptions, MetropolisHastings


def sample_from_selection_criterion(
    info,
    n_steps_total: int = 10_000,
    burnin_period: int = 4_000,
    n_chains: int = 2,
    param_initial_states: np.ndarray = None,
    silent: bool = False,
    show_progress: bool = True,
    plot_chains: bool = True,
):
    """
    Draw samples from the posterior distribution of the GP parameters.

    Parameters
    ----------
    info : object
        Must have attributes:
          - covparam: array-like initial GP parameters.
          - selection_criterion(p): returns the negative log-posterior for parameters p.
    n_steps_total : int
        Total number of MH steps.
    burnin_period : int
        Number of burn-in steps.
    n_chains : int
        Number of chains to run in parallel.
    param_initial_states : np.ndarray, optional
        Initial states of shape (n_chains, dim). Defaults to repeating info.covparam.
    silent : bool, optional
        If True, suppress messages.
    show_progress : bool, optional
        If True, display progress info.
    plot_chains : bool, optional
        If True, plot chain traces and densities after sampling.

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_chains, n_steps_total - burnin_period, dim) with post-burnin samples.
    mh : MetropolisHastings
        The sampler instance.
    """
    # Check that info has the required attributes.
    if not hasattr(info, "covparam"):
        raise AttributeError("The 'info' object must have a 'covparam' attribute.")
    param = np.asarray(info.covparam)
    if param is None:
        raise ValueError("info.covparam is None.")
    dim = param.shape[0]

    if not hasattr(info, "selection_criterion"):
        raise AttributeError(
            "The 'info' object must have a 'selection_criterion' attribute."
        )
    if info.selection_criterion is None:
        raise ValueError("info.selection_criterion is None.")
    if not callable(info.selection_criterion):
        raise TypeError("info.selection_criterion must be callable.")
    if n_steps_total < burnin_period:
        raise ValueError("n_steps_total must be greater than burnin_period.")

    if not silent:
        print()

    # Define log_target (posterior log-density)
    def log_target(p):
        return -info.selection_criterion(p)

    # Set up options for the MH sampler.
    show_prog = show_progress and (not silent)
    options = MHOptions(
        dim=dim,
        n_chains=n_chains,
        target_acceptance=0.3,
        proposal_distribution_param_init=0.1 * np.ones_like(param),
        adaptation_method="Haario",
        adaptation_interval=50,
        haario_adapt_factor_burnin_phase=1.0,
        haario_adapt_factor_sampling_phase=0.5,
        freeze_adaptation=False,
        discard_burnin=False,
        n_pool=2,
        show_global_progress=show_prog,
        init_msg=(
            None
            if silent
            else "Sampling from posterior distribution of GP parameters..."
        ),
    )

    # Initialize chain states.
    if param_initial_states is None:
        param_initial_states = np.tile(param, (n_chains, 1))

    # Create the sampler and run the MCMC.
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

    # Optionally plot the chains and empirical distributions.
    if plot_chains:
        mh.plot_chains()
        mh.plot_empirical_distributions(smooth=False)

    samples_post_burnin = param_samples[:, mh.burnin_period :, :]
    return samples_post_burnin, mh
