# gpmp/mcmc/param_posterior.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Posterior sampling helpers from GP parameter selection criteria.

This module bridges parameter-selection objectives and posterior samplers:
it turns a selection criterion ``J(theta)`` (typically a negative
log-posterior) into a log-density

``log_prob(theta) = -J(theta)``.

A hard truncation box can optionally be applied through ``sampling_box``:
outside the box, ``log_prob(theta) = -inf``.

Public entry points
-------------------
sample_from_selection_criterion_mh
    Adaptive Metropolis-Hastings sampler wrapper.
sample_from_selection_criterion_nuts
    NUTS wrapper with warmup/adaptation controls.
sample_from_selection_criterion_smc
    Tempered Sequential Monte Carlo sampler wrapper.
get_log_target_values
    Helper returning stored MH log-target traces.

Input conventions
-----------------
- Exactly one of ``info`` or ``selection_criterion`` must be provided.
- ``info`` is expected to contain at least ``covparam`` and criterion callables
  as produced by parameter-selection routines.
- ``init_box`` controls initialization support (mandatory for SMC, optional for
  MH/NUTS when random initialization is requested).
- ``sampling_box`` controls truncation of the target density during sampling.
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

    if require_differentiable:
        crit = getattr(info, "selection_criterion", None)
    else:
        crit = getattr(info, "selection_criterion_nograd", None)
        if crit is None:
            crit = getattr(info, "selection_criterion", None)

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
    box,
) -> int:
    if param_initial_states is not None:
        theta = gnp.asarray(param_initial_states)
        if theta.ndim == 0:
            return 1
        if theta.ndim == 1:
            return int(theta.shape[0])
        if theta.ndim == 2:
            return int(theta.shape[1])
        raise ValueError("param_initial_states must be scalar, 1D or 2D.")

    if info is not None:
        x0 = gnp.asarray(info.covparam)
        if x0.ndim != 1:
            raise ValueError("info.covparam must be 1D.")
        return int(x0.shape[0])

    if box is not None:
        lower, _ = box
        if gnp.isscalar(lower):
            raise ValueError(
                "Cannot infer dim from scalar box. Provide param_initial_states or info.covparam."
            )
        return int(len(lower))

    raise ValueError(
        "Cannot infer dim. Provide param_initial_states or info.covparam, or a non-scalar box."
    )


def _normalize_bounds(
    box: list,
    dim: int,
    box_name: str = "box",
) -> Tuple[gnp.ndarray, gnp.ndarray, object, object]:
    if not (isinstance(box, (list, tuple)) and len(box) == 2):
        raise ValueError(f"{box_name} must be of the form [lower, upper].")

    lower, upper = box

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
            raise ValueError(f"{box_name} bounds must match dimension.")

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
    if theta.ndim == 0:
        if dim != 1:
            raise ValueError("Scalar param_initial_states is only valid when dim == 1.")
        theta = gnp.tile(theta.reshape(1, 1), (n_chains, 1))
    elif theta.ndim == 1:
        n0 = int(theta.shape[0])
        if n0 == dim:
            theta = gnp.tile(theta.reshape(1, -1), (n_chains, 1))
        elif dim == 1 and n0 == n_chains:
            theta = theta.reshape(n_chains, 1)
        else:
            raise ValueError(
                f"1D param_initial_states must have length {dim}"
                + (f" (or {n_chains} when dim == 1)." if dim == 1 else ".")
            )
    elif theta.ndim == 2:
        r, c = int(theta.shape[0]), int(theta.shape[1])
        if r == n_chains and c == dim:
            pass
        elif r == 1 and c == dim:
            theta = gnp.tile(theta, (n_chains, 1))
        elif r == dim and c == n_chains:
            theta = theta.T
        else:
            raise ValueError(
                "2D param_initial_states must have shape "
                + f"({n_chains}, {dim}), (1, {dim}), or ({dim}, {n_chains})."
            )
    else:
        raise ValueError("param_initial_states must be scalar, 1D, or 2D.")

    if int(theta.shape[0]) != n_chains or int(theta.shape[1]) != dim:
        raise ValueError(f"param_initial_states must have shape ({n_chains}, {dim}).")
    return gnp.asarray(theta, dtype=gnp_dtype)


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


def get_log_target_values(
    mh: MetropolisHastings,
    *,
    discard_burnin: bool = False,
) -> gnp.ndarray:
    """
    Return stored MH log-target values.

    Parameters
    ----------
    mh : MetropolisHastings
        MH sampler instance returned by
        ``sample_from_selection_criterion_mh``.
    discard_burnin : bool, default=False
        If True, return only values after ``mh.burnin_period``.

    Returns
    -------
    ndarray
        Log-target values with shape ``(n_chains, n_steps)`` when
        ``discard_burnin=False``, or
        ``(n_chains, n_steps - burnin_period)`` when ``discard_burnin=True``.

    Raises
    ------
    ValueError
        If ``mh.log_target_values`` is unavailable or burn-in indices are invalid.
    """
    vals = getattr(mh, "log_target_values", None)
    if vals is None:
        raise ValueError(
            "mh.log_target_values is not available. Run mh.scheduler(...) first."
        )

    vals = gnp.asarray(vals)
    if vals.ndim != 2:
        raise ValueError("mh.log_target_values must be a 2D array.")

    if not discard_burnin:
        return vals

    b = int(mh.burnin_period)
    if b < 0:
        raise ValueError("mh.burnin_period must be >= 0.")
    if b > int(vals.shape[1]):
        raise ValueError("mh.burnin_period cannot exceed the number of stored steps.")

    return vals[:, b:]


# ---------------------------------------------------------------------
# Metropolis-Hastings
# ---------------------------------------------------------------------


def sample_from_selection_criterion_mh(
    info: object = None,
    selection_criterion: Callable = None,  # negative log-posterior
    param_initial_states: gnp.ndarray = None,
    random_init: bool = False,
    init_box: list = None,
    sampling_box: list = None,
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
    Sample from a parameter posterior with adaptive Metropolis-Hastings.

    The target log-density is defined as ``log_target(theta) = -J(theta)``,
    where ``J`` is the selection criterion. When ``sampling_box`` is provided,
    points outside the box are assigned ``-inf`` log-density.

    Parameters
    ----------
    info : object, optional
        Optimization info object carrying at least ``covparam`` and a criterion
        callable (typically ``selection_criterion_nograd`` or
        ``selection_criterion``). Provide exactly one of ``info`` and
        ``selection_criterion``.
    selection_criterion : callable, optional
        Negative log-posterior (or any negative objective) with signature
        ``f(theta) -> scalar``. Provide exactly one of ``info`` and
        ``selection_criterion``.
    param_initial_states : array_like, optional
        Initial parameter states. Accepted shapes are scalar, ``(dim,)``,
        ``(n_chains, dim)``, ``(1, dim)``, and ``(dim, n_chains)``.
    random_init : bool, default=False
        If True, initialize chains uniformly in ``init_box``.
    init_box : list, optional
        Initialization box ``[lower, upper]`` used only when
        ``random_init=True``.
    sampling_box : list, optional
        Hard sampling bounds ``[lower, upper]`` used to truncate the target
        density during MH transitions.
    n_steps_total : int, default=10000
        Total number of MH steps per chain, including burn-in.
    burnin_period : int, default=4000
        Number of initial steps discarded in the returned samples.
    n_chains : int, default=2
        Number of MH chains.
    n_pool : int, default=2
        Chain pooling factor used by covariance adaptation.
    silent : bool, default=False
        If True, suppress MH textual diagnostics.
    show_progress : bool, default=True
        If True and ``silent=False``, print global progress messages.
    plot_chains : bool, default=True
        If True, display MH chain traces after sampling.
    plot_empirical_distributions : bool, default=True
        If True, display marginal empirical distributions after sampling.

    Returns
    -------
    samples_post_burnin : ndarray
        Posterior samples with shape ``(n_chains, n_steps_total - burnin_period, dim)``.
    mh : MetropolisHastings
        Sampler instance containing full trajectories, acceptance history, and
        diagnostics.
    """
    crit = _resolve_selection_criterion(
        info,
        selection_criterion,
        require_differentiable=False,
    )
    dim_box = init_box if init_box is not None else sampling_box
    dim = _infer_dim(info, param_initial_states, dim_box)

    lower_init = upper_init = None
    lower_init_np = upper_init_np = None
    if init_box is not None:
        lower_init, upper_init, lower_init_np, upper_init_np = _normalize_bounds(
            init_box, dim, box_name="init_box"
        )

    lower_b = upper_b = None
    if sampling_box is not None:
        lower_b, upper_b, _, _ = _normalize_bounds(
            sampling_box, dim, box_name="sampling_box"
        )

    if random_init:
        if init_box is None:
            raise ValueError("init_box must be provided when random_init is True.")
        theta0 = _random_initial_states(lower_init_np, upper_init_np, dim, n_chains)
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
    sampling_box: list = None,
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
    """
    Sample from a parameter posterior with the No-U-Turn Sampler (NUTS).

    The target log-density is defined as ``log_prob(theta) = -J(theta)``,
    where ``J`` is the selection criterion. When ``sampling_box`` is provided,
    points outside the box are assigned ``-inf`` log-density.

    Parameters
    ----------
    info : object, optional
        Optimization info object carrying at least ``covparam`` and a
        differentiable criterion callable. Provide exactly one of ``info`` and
        ``selection_criterion``.
    selection_criterion : callable, optional
        Differentiable negative log-posterior with signature
        ``f(theta) -> scalar``. Provide exactly one of ``info`` and
        ``selection_criterion``.
    param_initial_states : array_like, optional
        Initial parameter states. Accepted shapes are scalar, ``(dim,)``,
        ``(n_chains, dim)``, ``(1, dim)``, and ``(dim, n_chains)``.
    random_init : bool, default=False
        If True, initialize chains uniformly in ``init_box``.
    init_box : list, optional
        Initialization box ``[lower, upper]`` used only when
        ``random_init=True``.
    sampling_box : list, optional
        Hard sampling bounds ``[lower, upper]`` used to truncate the target
        density during NUTS transitions.
    num_samples : int, default=2000
        Number of post-warmup samples per chain.
    num_warmup : int, default=1000
        Number of warmup iterations per chain.
    n_chains : int, default=2
        Number of chains.
    target_accept : float, default=0.8
        Target acceptance probability for dual averaging.
    max_depth : int, default=10
        Maximum tree depth per NUTS iteration.
    delta_max : float, default=1000.0
        Divergence threshold on Hamiltonian error.
    jitter : float, default=1e-4
        Lower clipping value for diagonal mass entries.
    init_step_size : float, optional
        User-provided initial step size. If None, a heuristic is used.
    init_mass_diag : array_like, optional
        User-provided initial diagonal mass vector of shape ``(dim,)``.
    seed : int, optional
        Random seed used by NUTS.
    progress : bool, default=True
        If True, display progress bars.
    verbose : int, default=1
        Logging verbosity level.
    log_every : int, default=50
        Logging frequency in iterations.
    options : NUTSOptions, optional
        Optional fully specified NUTS options object.
    plot_diagnostics : bool, default=False
        If True, generate NUTS diagnostic figures.
    diagnostics_window : int, default=50
        Moving-average window for diagnostics.
    diagnostics_show : bool, default=True
        If True, show diagnostic plots interactively.
    diagnostics_save_dir : str, optional
        Directory where diagnostic plots are saved when provided.

    Returns
    -------
    samples : ndarray
        Posterior samples with shape ``(n_chains, num_samples, dim)``.
    info_nuts : dict
        NUTS diagnostics and adaptation outputs (acceptance, divergences,
        tree depth, leapfrog counts, final step size, and final mass diagonal).
    """
    crit = _resolve_selection_criterion(
        info,
        selection_criterion,
        require_differentiable=True,
    )
    dim_box = init_box if init_box is not None else sampling_box
    dim = _infer_dim(info, param_initial_states, dim_box)

    lower_init = upper_init = None
    lower_init_np = upper_init_np = None
    if init_box is not None:
        lower_init, upper_init, lower_init_np, upper_init_np = _normalize_bounds(
            init_box, dim, box_name="init_box"
        )

    lower_b = upper_b = None
    if sampling_box is not None:
        lower_b, upper_b, _, _ = _normalize_bounds(
            sampling_box, dim, box_name="sampling_box"
        )

    if random_init:
        if init_box is None:
            raise ValueError("init_box must be provided when random_init is True.")
        theta0 = _random_initial_states(lower_init_np, upper_init_np, dim, n_chains)
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
    sampling_box: list = None,
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
    Sample from a parameter posterior with Sequential Monte Carlo tempering.

    The sampler targets a sequence of tempered densities of the form
    ``exp(-J(theta) / temperature)`` from ``initial_temperature`` down to
    ``final_temperature``. When ``sampling_box`` is provided, points outside
    the box are assigned ``-inf`` log-density.

    Parameters
    ----------
    info : object, optional
        Optimization info object carrying at least ``covparam`` and a criterion
        callable. Provide exactly one of ``info`` and ``selection_criterion``.
    selection_criterion : callable, optional
        Negative log-posterior with signature ``f(theta) -> scalar``.
        Provide exactly one of ``info`` and ``selection_criterion``.
    init_box : list
        Mandatory initialization box ``[lower, upper]`` used to initialize SMC
        particles.
    sampling_box : list, optional
        Hard sampling bounds ``[lower, upper]`` used to truncate the target
        density during SMC.
    n_particles : int, default=1000
        Number of SMC particles.
    initial_temperature : float, default=1e6
        Initial tempering parameter (high temperature, easier target).
    final_temperature : float, default=1.0
        Final tempering parameter (target posterior scale).
    min_ess_ratio : float, default=0.5
        Minimum effective sample size ratio used by the tempering scheduler.
    mh_steps : int, default=20
        Number of MH move steps per SMC stage.
    max_stages : int, default=50
        Reserved stage budget parameter for compatibility.
    debug : bool, default=False
        If True, print detailed SMC diagnostics.
    plot_marginals : bool, default=False
        If True, plot empirical marginal distributions from SMC.
    plot_particles : bool, default=False
        Reserved plotting flag for compatibility.

    Returns
    -------
    particles : ParticlesSet
        Final particle set after tempering.
    smc_instance : SMC
        SMC driver instance containing execution logs and diagnostics.
    """
    f = _resolve_selection_criterion(
        info,
        selection_criterion,
        require_differentiable=False,
    )

    if init_box is None:
        raise ValueError("init_box must be provided for SMC.")

    dim = _infer_dim(info, None, init_box)
    _, _, _, _ = _normalize_bounds(init_box, dim, box_name="init_box")

    lower_b = upper_b = None
    if sampling_box is not None:
        lower_b, upper_b, _, _ = _normalize_bounds(
            sampling_box, dim, box_name="sampling_box"
        )

    def _criterion_scalar(theta):
        return gnp.to_scalar(gnp.asarray(f(theta)).reshape(()))

    def logpdf_temp(x, temperature):
        x = gnp.asarray(x)

        if x.ndim == 1:
            if lower_b is not None:
                if bool(gnp.any(x < lower_b)) or bool(gnp.any(x > upper_b)):
                    return gnp.return_neginf()
            return -_criterion_scalar(x) / temperature

        if x.ndim == 2:
            vals = gnp.asarray([_criterion_scalar(x[i]) for i in range(x.shape[0])])
            out = -vals / temperature
            if lower_b is None:
                return out
            in_box = gnp.all(x >= lower_b, axis=1) & gnp.all(x <= upper_b, axis=1)
            return gnp.where(in_box, out, gnp.safe_neginf())

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
