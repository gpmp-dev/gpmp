"""
MCMC sampling from a negative log-posterior.

The function `sample_from_selection_criterion` uses a Metropolis–Hastings sampler
to draw samples from a distribution whose negative log-posterior is defined by
`info.selection_criterion`. Users can configure the number of steps, burn-in,
number of chains, initial states, progress display, and optional plotting. 

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import numpy as np
from gpmp.misc.mcmc import MHOptions, MetropolisHastings


def sample_from_selection_criterion(
    info,
    n_steps_total=10_000,
    burnin_period=3_000,
    n_chains=2,
    initial_states=None,
    silent=False,
    show_progress=True,
    plot_chains=True,
):
    """
    Draw samples from the posterior distribution, using info.selection_criterion
    as (proportional to) the (negative) log posterior.

    Parameters
    ----------
    info : object
        Must have:
            - covparam: array-like, the optimal covariance parameters.
            - selection_criterion(param): returns a scalar (negative log posterior
              or something you want to exponentiate for sampling).
    n_steps_total : int
        Total number of Metropolis–Hastings steps.
    burnin_period : int
        Number of burn-in steps.
    n_chains : int
        How many chains to run in parallel.
    initial_states : np.ndarray, optional
        Array of shape (n_chains, dim) for the initial chain states.
        If None, defaults to using info.covparam repeated across n_chains.
    show_progress : bool
        Whether to show progress bars or summary info.
    plot_chains : bool, optional
        Whether to plot chain traces and empirical distributions
        after sampling. Default is True.
    silent : bool, optional
        If True, suppress printing and progress display.
        Default = False (meaning print a short message and allow progress).

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_chains, n_steps_total - burnin_period, dim)
        containing the post-burnin samples.
    mh : MetropolisHastings
        The MH sampler instance (in case you want to inspect it further).
    """

    # 1. Check arguments
    if not hasattr(info, "covparam"):
        raise AttributeError("The 'info' object must have a 'covparam' attribute.")
    param = getattr(info, "covparam")
    if param is None:
        raise ValueError(
            "info.covparam is None. A valid array of parameters is required."
        )
    param = np.asarray(param)
    dim = param.shape[0]

    if not hasattr(info, "selection_criterion"):
        raise AttributeError(
            "The 'info' object must have a 'selection_criterion' attribute."
        )
    if info.selection_criterion is None:
        raise ValueError(
            "info.selection_criterion is None. A valid callable is required."
        )
    if not callable(info.selection_criterion):
        raise TypeError("info.selection_criterion must be callable.")
    if n_steps_total < burnin_period:
        raise ValueError("n_steps_total must be strictly greater than burnin_period.")

    # 2. If not silent, print a brief message:
    if not silent:
        print()

    # 3. Define the log_target for the sampler.
    def log_target(p):
        # Because selection_criterion is negative log posterior,
        # use the negative sign to convert to log posterior:
        return -info.selection_criterion(p)

    # 4. Build MHOptions
    show_progress = show_progress and (not silent)
    options = MHOptions(
        dim=dim,
        n_chains=n_chains,
        target_acceptance=0.3,
        proposal_param_init=0.1 * np.ones_like(param),
        adaptation_method="Haario",
        adaptation_interval=50,
        freeze_adaptation=True,
        discard_burnin=False,
        n_pool=2,
        show_global_progress=show_progress,
        init_msg=(
            None
            if silent
            else "Sampling from posterior distribution using info.selection_criterion..."
        ),
    )

    # 5. Initialize states
    if initial_states is None:
        initial_states = np.tile(param, (n_chains, 1))

    # 6. Create sampler and run it
    mh = MetropolisHastings(log_target=log_target, options=options)
    samples_all = mh.scheduler(
        initial_states=initial_states,
        n_steps_total=n_steps_total,
        burnin_period=burnin_period,
    )
    mh.check_acceptance_rates(burnin_period=burnin_period)

    # 7. Optionally show plots
    if plot_chains:
        mh.plot_chains(burnin=burnin_period)
        mh.plot_empirical_distributions(burnin=burnin_period, smooth=False)

    # 8. Return the post-burnin samples
    samples_post_burnin = samples_all[:, burnin_period:, :]

    return samples_post_burnin, mh
