# gpmp/mcmc/param_posterior.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
MCMC sampling from a GP parameter posterior.

Convention
----------
selection_criterion(p) -> scalar
returns a negative log-posterior (or any negative objective whose exponential
defines a density on p).

All samplers below use the log density:
    log_prob(p) = -selection_criterion(p)
with an optional box constraint init_box giving log_prob = -inf outside.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import gpmp.config as config
import gpmp.num as gnp
from gpmp.misc.designs import randunif

from .mh import MHOptions, MetropolisHastings
from .nuts import NUTSOptions, nuts_sample, plot_nuts_diagnostics
from .smc import run_smc_sampling

gnp_dtype = gnp.get_dtype()
_backend = config.get_backend()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _resolve_selection_criterion(
    info: object,
    selection_criterion: Optional[Callable],
    *,
    require_differentiable: bool,
) -> Callable:
    if (info is None) == (selection_criterion is None):
        raise ValueError("Provide exactly one of: info or selection_criterion.")

    if selection_criterion is not None:
        return selection_criterion

    crit = getattr(info, "selection_criterion", None)
    if crit is None:
        crit = getattr(info, "selection_criterion_nograd", None)

    if crit is None or not callable(crit):
        raise ValueError(
            "info must provide selection_criterion or selection_criterion_nograd."
        )

    if (
        require_differentiable
        and _backend == "torch"
        and getattr(info, "selection_criterion", None) is None
    ):
        raise ValueError(
            "Torch backend requires info.selection_criterion (differentiable)."
        )

    return crit


def _infer_dim(
    info: object,
    param_initial_states,
    init_box,
) -> int:
    if param_initial_states is not None:
        theta = gnp.asarray(param_initial_states)
        if theta.ndim == 1:
            return int(theta.shape[0])
        if theta.ndim == 2:
            return int(theta.shape[1])
        raise ValueError("param_initial_states must be 1D or 2D.")

    if info is not None:
        x0 = gnp.asarray(info.covparam)
        if x0.ndim != 1:
            raise ValueError("info.covparam must be 1D.")
        return int(x0.shape[0])

    if init_box is not None:
        lower, _ = init_box
        if gnp.isscalar(lower):
            raise ValueError(
                "Cannot infer dim from scalar init_box. Provide param_initial_states or info.covparam."
            )
        return int(len(lower))

    raise ValueError(
        "Cannot infer dim. Provide param_initial_states or info.covparam, or a non-scalar init_box."
    )


def _normalize_bounds(
    init_box: list,
    dim: int,
) -> Tuple[gnp.ndarray, gnp.ndarray, object, object]:
    if not (isinstance(init_box, (list, tuple)) and len(init_box) == 2):
        raise ValueError("init_box must be of the form [lower, upper].")

    lower, upper = init_box

    if gnp.isscalar(lower) and gnp.isscalar(upper):
        lower_b = gnp.ones(dim, dtype=gnp_dtype) * float(lower)
        upper_b = gnp.ones(dim, dtype=gnp_dtype) * float(upper)
    else:
        lower_b = gnp.asarray(lower, dtype=gnp_dtype).reshape(-1)
        upper_b = gnp.asarray(upper, dtype=gnp_dtype).reshape(-1)

        if int(lower_b.shape[0]) == 1:
            lower_b = gnp.tile(lower_b, (dim,))
        if int(upper_b.shape[0]) == 1:
            upper_b = gnp.tile(upper_b, (dim,))

        if int(lower_b.shape[0]) != dim or int(upper_b.shape[0]) != dim:
            raise ValueError("init_box bounds must match dimension.")

    # randunif expects NumPy-like arrays/lists. gnp.to_np is a no-op on NumPy backend.
    lower_np = gnp.to_np(lower_b)
    upper_np = gnp.to_np(upper_b)
    return lower_b, upper_b, lower_np, upper_np


def _normalize_initial_states(
    info: object,
    param_initial_states,
    n_chains: int,
    dim: int,
) -> gnp.ndarray:
    if param_initial_states is None:
        if info is None:
            raise ValueError(
                "param_initial_states must be provided when info is None and random_init is False."
            )
        x0 = gnp.asarray(info.covparam, dtype=gnp_dtype).reshape(-1)
        if int(x0.shape[0]) != dim:
            raise ValueError("info.covparam has incompatible dimension.")
        theta = gnp.tile(x0, (n_chains, 1))
        return theta

    theta = gnp.asarray(param_initial_states, dtype=gnp_dtype)
    if theta.ndim == 1:
        theta = gnp.tile(theta, (n_chains, 1))
    elif theta.ndim != 2:
        raise ValueError("param_initial_states must be 1D or 2D.")

    if theta.shape != (n_chains, dim):
        raise ValueError(f"param_initial_states must have shape ({n_chains}, {dim}).")
    return theta


def _random_initial_states(
    lower_np,
    upper_np,
    dim: int,
    n_chains: int,
) -> gnp.ndarray:
    theta = randunif(dim, n_chains, [lower_np, upper_np])
    return gnp.asarray(theta, dtype=gnp_dtype)


def _make_log_prob(
    selection_criterion: Callable,
    lower_b: Optional[gnp.ndarray],
    upper_b: Optional[gnp.ndarray],
) -> Callable[[gnp.ndarray], gnp.ndarray]:
    def log_prob(p):
        p = gnp.asarray(p)
        if lower_b is not None:
            if gnp.any(p < lower_b) or gnp.any(p > upper_b):
                return gnp.safe_neginf()

        try:
            v = selection_criterion(p)
        except Exception:
            return gnp.safe_neginf()

        return -gnp.asarray(v)

    return log_prob


# ---------------------------------------------------------------------
# Metropolis-Hastings
# ---------------------------------------------------------------------


def sample_from_selection_criterion_mh(
    info: object = None,
    selection_criterion: Callable = None,  # negative log-posterior
    param_initial_states: gnp.ndarray = None,
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
    crit = _resolve_selection_criterion(
        info,
        selection_criterion,
        require_differentiable=False,
    )
    dim = _infer_dim(info, param_initial_states, init_box)

    lower_b = upper_b = None
    lower_np = upper_np = None
    if init_box is not None:
        lower_b, upper_b, lower_np, upper_np = _normalize_bounds(init_box, dim)

    if random_init:
        if init_box is None:
            raise ValueError("init_box must be provided when random_init is True.")
        theta0 = _random_initial_states(lower_np, upper_np, dim, n_chains)
    else:
        theta0 = _normalize_initial_states(info, param_initial_states, n_chains, dim)

    if n_steps_total < burnin_period:
        raise ValueError("n_steps_total must be greater than burnin_period.")

    log_target = _make_log_prob(crit, lower_b, upper_b)

    show_prog = show_progress and not silent
    options = MHOptions(
        dim=dim,
        n_chains=n_chains,
        target_acceptance=0.3,
        proposal_distribution_param_init=0.1 * gnp.ones(dim),
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
        chains_state_initial=theta0,
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


# ---------------------------------------------------------------------
# No-U-Turn Sampler (NUTS)
# ---------------------------------------------------------------------


def sample_from_selection_criterion_nuts(
    info: object = None,
    selection_criterion: Callable = None,  # negative log-posterior
    param_initial_states: gnp.ndarray = None,
    random_init: bool = False,
    init_box: list = None,
    num_samples: int = 2_000,
    num_warmup: int = 1_000,
    n_chains: int = 2,
    target_accept: float = 0.8,
    max_depth: int = 10,
    delta_max: float = 1_000.0,
    jitter: float = 1e-4,
    init_step_size: Optional[float] = None,
    init_mass_diag: gnp.ndarray = None,
    seed: int = None,
    progress: bool = True,
    verbose: int = 1,
    log_every: int = 50,
    options: NUTSOptions = None,
    plot_diagnostics: bool = False,
    diagnostics_window: int = 50,
    diagnostics_show: bool = True,
    diagnostics_save_dir: str = None,
):
    crit = _resolve_selection_criterion(
        info,
        selection_criterion,
        require_differentiable=True,
    )
    dim = _infer_dim(info, param_initial_states, init_box)

    lower_b = upper_b = None
    lower_np = upper_np = None
    if init_box is not None:
        lower_b, upper_b, lower_np, upper_np = _normalize_bounds(init_box, dim)

    if random_init:
        if init_box is None:
            raise ValueError("init_box must be provided when random_init is True.")
        theta0 = _random_initial_states(lower_np, upper_np, dim, n_chains)
    else:
        theta0 = _normalize_initial_states(info, param_initial_states, n_chains, dim)

    log_prob = _make_log_prob(crit, lower_b, upper_b)

    samples_raw, info_nuts = nuts_sample(
        log_prob=log_prob,
        q_init=gnp.asarray(theta0, dtype=gnp_dtype),
        num_samples=num_samples,
        num_warmup=num_warmup,
        target_accept=target_accept,
        max_depth=max_depth,
        delta_max=delta_max,
        jitter=jitter,
        init_step_size=init_step_size,
        init_mass_diag=init_mass_diag,
        seed=seed,
        progress=progress,
        verbose=verbose,
        log_every=log_every,
        options=options,
    )

    if plot_diagnostics:
        plot_nuts_diagnostics(
            info_nuts,
            window=diagnostics_window,
            show=diagnostics_show,
            save_dir=diagnostics_save_dir,
        )

    # gnp.transpose has torch signature (swap two dims)
    samples = gnp.transpose(samples_raw, 0, 1)  # (n_chains, num_samples, dim)
    return samples, info_nuts


# ---------------------------------------------------------------------
# Sequential Monte Carlo (SMC)
# ---------------------------------------------------------------------


def sample_from_selection_criterion_smc(
    info: object = None,
    selection_criterion: Callable = None,  # negative log-posterior
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
    f = _resolve_selection_criterion(
        info,
        selection_criterion,
        require_differentiable=False,
    )

    lower_b = upper_b = None
    if init_box is not None:
        dim = _infer_dim(info, None, init_box)
        lower_b, upper_b, _, _ = _normalize_bounds(init_box, dim)

    def logpdf_temp(x, temperature):
        x = gnp.asarray(x)

        if x.ndim == 1:
            if lower_b is not None:
                if bool(gnp.any(x < lower_b)) or bool(gnp.any(x > upper_b)):
                    return gnp.return_neginf()
            return -gnp.asarray(f(x)) / temperature

        if x.ndim == 2:
            if lower_b is None:
                vals = gnp.stack([gnp.asarray(f(x[i])) for i in range(x.shape[0])])
                return -vals / temperature

            in_box = gnp.all(x >= lower_b, axis=1) & gnp.all(x <= upper_b, axis=1)
            vals = gnp.stack([gnp.asarray(f(x[i])) for i in range(x.shape[0])])
            out = -vals / temperature
            return gnp.where(in_box, out, gnp.return_neginf())

        raise ValueError("x must be 1D or 2D.")

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
