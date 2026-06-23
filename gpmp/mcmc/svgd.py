# gpmp/mcmc/svgd.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Stein variational gradient descent (SVGD) for approximate posterior sampling.

This module implements a minimal SVGD sampler using ``gpmp.num`` primitives.
It is intended as a lightweight particle transport method for differentiable
targets. In GP parameter problems, it is best viewed as an approximate
posterior explorer rather than an exact MCMC replacement.

Public API
----------
SVGDOptions
    Configuration for annealing, kernel bandwidth, preconditioning, and logs.
rbf_kernel_matrix
    RBF kernel matrix with median-heuristic bandwidth.
svgd_step
    One SVGD update on a particle cloud.
svgd_sample
    Multi-step annealed SVGD driver.
plot_svgd_empirical_distributions
    Histogram/KDE plots for final SVGD particle marginals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import gpmp.num as gnp
from gpmp.misc.designs import randunif

gnp_dtype = gnp.get_dtype()


@dataclass
class SVGDOptions:
    n_steps: int = 500
    step_size: float = 1e-2
    bandwidth: float | None = None
    bandwidth_scale: float = 1.0
    bandwidth_min: float | None = None
    preconditioner_diag: gnp.ndarray | None = None
    initial_temperature: float = 10.0
    final_temperature: float = 1.0
    annealing_schedule: str = "geometric"
    sampling_box: list | None = None
    store_particles_history: bool = False
    verbose: int = 1
    progress: bool = True
    log_every: int = 50
    jitter: float = 1e-12


def _to_float(x) -> float:
    return float(gnp.to_np(gnp.asarray(x).reshape(())))


def _normalize_bounds(
    box: list,
    dim: int,
    *,
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

    return lower_b, upper_b, gnp.to_np(lower_b), gnp.to_np(upper_b)


def _project_to_box(
    particles: gnp.ndarray,
    lower_b: Optional[gnp.ndarray],
    upper_b: Optional[gnp.ndarray],
) -> gnp.ndarray:
    if lower_b is None:
        return particles
    return gnp.clip(particles, min=lower_b.reshape(1, -1), max=upper_b.reshape(1, -1))


def _annealed_temperature(
    step: int,
    n_steps: int,
    *,
    initial_temperature: float,
    final_temperature: float,
    schedule: str,
):
    t0 = float(initial_temperature)
    t1 = float(final_temperature)
    if t0 <= 0.0 or t1 <= 0.0:
        raise ValueError("Temperatures must be > 0.")
    if n_steps <= 1:
        return gnp.asarray(t1, dtype=gnp_dtype)

    u = float(step) / float(n_steps - 1)
    if schedule == "linear":
        temp = t0 + u * (t1 - t0)
    elif schedule == "geometric":
        temp = t0 * (t1 / t0) ** u
    else:
        raise ValueError("annealing_schedule must be 'linear' or 'geometric'.")
    return gnp.asarray(temp, dtype=gnp_dtype)


def _resolve_preconditioner(
    preconditioner_diag,
    dim: int,
    *,
    jitter: float,
) -> gnp.ndarray:
    if preconditioner_diag is None:
        return gnp.ones(dim, dtype=gnp_dtype)

    diag = gnp.asarray(preconditioner_diag, dtype=gnp_dtype).reshape(-1)
    if int(diag.shape[0]) == 1:
        diag = gnp.tile(diag, (dim,))
    if int(diag.shape[0]) != dim:
        raise ValueError("preconditioner_diag must have length equal to particle dimension.")
    if bool(gnp.to_np(gnp.any(diag <= 0.0))):
        raise ValueError("preconditioner_diag must be strictly positive.")
    return gnp.clip(diag, min=float(jitter))


def _safe_eval_log_prob(log_prob: Callable, x: gnp.ndarray):
    try:
        value = gnp.asarray(log_prob(x), dtype=gnp_dtype).reshape(())
    except Exception:
        return gnp.asarray(gnp.safe_neginf(), dtype=gnp_dtype).reshape(())
    return value


def _safe_value_and_grad(log_prob: Callable, x: gnp.ndarray):
    try:
        value, grad = gnp.value_and_grad(log_prob, x)
    except Exception:
        return (
            gnp.asarray(gnp.safe_neginf(), dtype=gnp_dtype).reshape(()),
            gnp.zeros_like(x, dtype=gnp_dtype),
        )

    value = gnp.asarray(value, dtype=gnp_dtype).reshape(())
    grad = gnp.asarray(grad, dtype=gnp_dtype).reshape(-1)
    if not _to_float(gnp.isfinite(value)):
        return value, gnp.zeros_like(x, dtype=gnp_dtype)
    if bool(gnp.to_np(gnp.any(gnp.logical_not(gnp.isfinite(grad))))):
        grad = gnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return value, grad


def rbf_kernel_matrix(
    particles: gnp.ndarray,
    *,
    bandwidth: float | None = None,
    bandwidth_scale: float = 1.0,
    bandwidth_min: float | None = None,
    jitter: float = 1e-12,
):
    """
    Compute an RBF kernel matrix on particles.

    Parameters
    ----------
    particles : array_like, shape (n_particles, dim)
        Particle cloud.
    bandwidth : float, optional
        RBF bandwidth. If None, a median heuristic is used on pairwise squared
        distances, scaled by ``1 / log(n_particles + 1)``.
    bandwidth_scale : float, default=1.0
        Multiplicative factor applied to the inferred or provided bandwidth.
    bandwidth_min : float, optional
        Lower bound enforced on the effective bandwidth.
    jitter : float, default=1e-12
        Minimum bandwidth safeguard.

    Returns
    -------
    kernel : ndarray
        RBF kernel matrix.
    sq_dists : ndarray
        Pairwise squared Euclidean distances.
    bandwidth : scalar
        Effective bandwidth used in the kernel.
    """
    particles = gnp.asarray(particles, dtype=gnp_dtype)
    if particles.ndim != 2:
        raise ValueError("particles must have shape (n_particles, dim).")
    if float(bandwidth_scale) <= 0.0:
        raise ValueError("bandwidth_scale must be > 0.")
    if bandwidth_min is not None and float(bandwidth_min) <= 0.0:
        raise ValueError("bandwidth_min must be > 0 when provided.")

    diffs = particles[:, None, :] - particles[None, :, :]
    sq_dists = gnp.sum(diffs * diffs, axis=2)
    n_particles = int(particles.shape[0])

    if bandwidth is None:
        positive_sq_dists = sq_dists[sq_dists > 0.0]
        if int(positive_sq_dists.reshape(-1).shape[0]) == 0:
            h = gnp.asarray(float(bandwidth_scale), dtype=gnp_dtype)
        else:
            median_sq = gnp.asarray(
                gnp.percentile(positive_sq_dists, 50.0), dtype=gnp_dtype
            ).reshape(())
            scale = gnp.log(
                gnp.asarray(float(n_particles) + 1.0, dtype=gnp_dtype)
            ).reshape(())
            if _to_float(scale) <= 0.0:
                scale = gnp.asarray(1.0, dtype=gnp_dtype)
            h = float(bandwidth_scale) * median_sq / scale
    else:
        h = gnp.asarray(float(bandwidth_scale) * float(bandwidth), dtype=gnp_dtype)

    if not _to_float(gnp.isfinite(h)) or _to_float(h) <= float(jitter):
        h = gnp.asarray(max(float(bandwidth_scale), float(jitter)), dtype=gnp_dtype)
    if bandwidth_min is not None:
        h = gnp.maximum(h, gnp.asarray(float(bandwidth_min), dtype=gnp_dtype))

    kernel = gnp.exp(-sq_dists / h)
    return kernel, sq_dists, h


def svgd_step(
    log_prob: Callable[[gnp.ndarray], gnp.ndarray],
    particles: gnp.ndarray,
    *,
    step_size: float,
    temperature: float = 1.0,
    bandwidth: float | None = None,
    bandwidth_scale: float = 1.0,
    bandwidth_min: float | None = None,
    preconditioner_diag=None,
    sampling_box: list | None = None,
    jitter: float = 1e-12,
):
    """
    Perform one SVGD update.

    Parameters
    ----------
    log_prob : callable
        Differentiable log-density function ``log_prob(theta) -> scalar``.
    particles : array_like, shape (n_particles, dim)
        Current particles.
    step_size : float
        SVGD transport step size.
    temperature : float, default=1.0
        Annealing temperature. The transported target is ``log_prob / temperature``.
    bandwidth, bandwidth_scale, bandwidth_min, preconditioner_diag, sampling_box, jitter :
        Kernel, preconditioning, and box-projection parameters.

    Returns
    -------
    particles_next : ndarray
        Updated particle cloud.
    info : dict
        Step diagnostics.
    """
    particles = gnp.asarray(particles, dtype=gnp_dtype)
    if particles.ndim != 2:
        raise ValueError("particles must have shape (n_particles, dim).")
    if float(step_size) <= 0.0:
        raise ValueError("step_size must be > 0.")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0.")

    n_particles, dim = int(particles.shape[0]), int(particles.shape[1])
    lower_b = upper_b = None
    if sampling_box is not None:
        lower_b, upper_b, _, _ = _normalize_bounds(
            sampling_box, dim, box_name="sampling_box"
        )
        particles = _project_to_box(particles, lower_b, upper_b)

    preconditioner = _resolve_preconditioner(
        preconditioner_diag, dim, jitter=float(jitter)
    )

    def tempered_log_prob(theta):
        theta = gnp.asarray(theta, dtype=gnp_dtype).reshape(-1)
        if lower_b is not None:
            if bool(gnp.to_np(gnp.any(theta < lower_b))) or bool(
                gnp.to_np(gnp.any(theta > upper_b))
            ):
                return gnp.safe_neginf()
        return gnp.asarray(log_prob(theta), dtype=gnp_dtype).reshape(()) / float(
            temperature
        )

    log_prob_values = []
    score_values = []
    for i in range(n_particles):
        value_i, grad_i = _safe_value_and_grad(tempered_log_prob, particles[i])
        log_prob_values.append(value_i)
        score_values.append(grad_i)

    log_prob_values = gnp.stack(log_prob_values)
    score_values = gnp.stack(score_values)
    alive_mask = gnp.isfinite(log_prob_values).reshape(-1)
    alive_count = int(gnp.to_np(gnp.sum(alive_mask)))
    score_values = gnp.where(alive_mask[:, None], score_values, 0.0)

    kernel, sq_dists, h = rbf_kernel_matrix(
        particles,
        bandwidth=bandwidth,
        bandwidth_scale=bandwidth_scale,
        bandwidth_min=bandwidth_min,
        jitter=jitter,
    )
    kernel = kernel * alive_mask[:, None] * alive_mask[None, :]

    denom = float(max(alive_count, 1))
    score_term = gnp.einsum("ij,jd->id", kernel, score_values) / denom
    diffs = particles[:, None, :] - particles[None, :, :]
    repulsion_term = (
        (2.0 / h) * gnp.sum(kernel[:, :, None] * diffs, axis=1) / denom
    )
    velocity = (score_term + repulsion_term) * preconditioner.reshape(1, -1)
    velocity = gnp.where(alive_mask[:, None], velocity, 0.0)
    if bool(gnp.to_np(gnp.any(gnp.logical_not(gnp.isfinite(velocity))))):
        velocity = gnp.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

    particles_next = particles + float(step_size) * velocity
    if lower_b is not None:
        particles_next = _project_to_box(particles_next, lower_b, upper_b)

    return particles_next, {
        "temperature": gnp.asarray(float(temperature), dtype=gnp_dtype),
        "bandwidth": gnp.asarray(h, dtype=gnp_dtype).reshape(()),
        "kernel": kernel,
        "sq_dists": sq_dists,
        "log_prob_values": log_prob_values,
        "score_values": score_values,
        "velocity": velocity,
        "preconditioner_diag": preconditioner,
        "alive_mask": alive_mask,
        "alive_count": alive_count,
    }


def svgd_sample(
    log_prob: Callable[[gnp.ndarray], gnp.ndarray],
    particles_initial: gnp.ndarray | None = None,
    *,
    n_particles: int | None = None,
    dim: int | None = None,
    init_box: list | None = None,
    options: SVGDOptions | None = None,
):
    """
    Run annealed SVGD for a differentiable target density.

    Parameters
    ----------
    log_prob : callable
        Differentiable log-density function ``log_prob(theta) -> scalar``.
    particles_initial : array_like, optional
        Initial particles with shape ``(n_particles, dim)`` or ``(dim,)``.
    n_particles : int, optional
        Number of particles when initializing from ``init_box``.
    dim : int, optional
        Particle dimension when it cannot be inferred from inputs.
    init_box : list, optional
        Initialization box ``[lower, upper]`` used when ``particles_initial`` is
        not provided.
    options : SVGDOptions, optional
        SVGD driver configuration.

    Returns
    -------
    particles : ndarray
        Final particle cloud.
    info : dict
        Run diagnostics and traces.
    """
    opts = SVGDOptions() if options is None else options
    if int(opts.n_steps) < 0:
        raise ValueError("n_steps must be >= 0.")

    if particles_initial is None:
        if init_box is None:
            raise ValueError("Provide particles_initial or init_box.")
        if n_particles is None or int(n_particles) <= 0:
            raise ValueError("n_particles must be provided and > 0 when init_box is used.")
        if dim is None:
            lower = init_box[0]
            if gnp.isscalar(lower):
                raise ValueError(
                    "dim must be provided when init_box lower bound is scalar."
                )
            dim = int(len(lower))
        _, _, lower_np, upper_np = _normalize_bounds(
            init_box, int(dim), box_name="init_box"
        )
        particles = randunif(int(dim), int(n_particles), [lower_np, upper_np])
        particles = gnp.asarray(particles, dtype=gnp_dtype)
    else:
        particles = gnp.asarray(particles_initial, dtype=gnp_dtype)
        if particles.ndim == 1:
            particles = particles.reshape(1, -1)
        elif particles.ndim != 2:
            raise ValueError("particles_initial must be 1D or 2D.")

    n_particles_eff, dim_eff = int(particles.shape[0]), int(particles.shape[1])
    if n_particles is not None and int(n_particles) != n_particles_eff:
        raise ValueError("n_particles does not match particles_initial.")
    if dim is not None and int(dim) != dim_eff:
        raise ValueError("dim does not match particles_initial.")

    particles_history = [gnp.copy(particles)] if opts.store_particles_history else None
    log_prob_trace = []
    bandwidth_trace = []
    temperature_trace = []
    velocity_norm_trace = []

    for step in range(int(opts.n_steps)):
        temperature = _annealed_temperature(
            step,
            int(opts.n_steps),
            initial_temperature=opts.initial_temperature,
            final_temperature=opts.final_temperature,
            schedule=opts.annealing_schedule,
        )
        particles, step_info = svgd_step(
            log_prob,
            particles,
            step_size=opts.step_size,
            temperature=_to_float(temperature),
            bandwidth=opts.bandwidth,
            bandwidth_scale=opts.bandwidth_scale,
            bandwidth_min=opts.bandwidth_min,
            preconditioner_diag=opts.preconditioner_diag,
            sampling_box=opts.sampling_box,
            jitter=opts.jitter,
        )
        log_prob_trace.append(step_info["log_prob_values"])
        bandwidth_trace.append(step_info["bandwidth"])
        temperature_trace.append(step_info["temperature"])
        velocity_norm_trace.append(gnp.mean(gnp.norm(step_info["velocity"], axis=1)))
        if opts.store_particles_history:
            particles_history.append(gnp.copy(particles))

        should_log = (
            bool(opts.progress)
            and int(opts.verbose) > 0
            and (
                step == 0
                or step + 1 == int(opts.n_steps)
                or ((step + 1) % max(int(opts.log_every), 1) == 0)
            )
        )
        if should_log:
            alive_mask = step_info["alive_mask"]
            alive_count = int(step_info["alive_count"])
            if alive_count > 0:
                alive_log_probs = step_info["log_prob_values"][alive_mask]
                alive_velocities = gnp.norm(step_info["velocity"][alive_mask], axis=1)
                mean_log_prob = _to_float(gnp.mean(alive_log_probs))
                best_log_prob = _to_float(gnp.max(alive_log_probs))
                mean_velocity = _to_float(gnp.mean(alive_velocities))
                temperature_float = _to_float(step_info["temperature"])
                best_criterion = -temperature_float * best_log_prob
            else:
                mean_log_prob = float("-inf")
                best_log_prob = float("-inf")
                mean_velocity = 0.0
                best_criterion = float("inf")
            print(
                f"svgd iter {step + 1}/{int(opts.n_steps)}: "
                f"T={_to_float(step_info['temperature']):.6g}, "
                f"bandwidth={_to_float(step_info['bandwidth']):.6g}, "
                f"n_alive={alive_count}/{n_particles_eff}, "
                f"mean_log_prob={mean_log_prob:.6g}, "
                f"best_log_prob={best_log_prob:.6g}, "
                f"best_criterion={best_criterion:.6g}, "
                f"mean_velocity_norm={mean_velocity:.6g}"
            )

    final_log_prob_values = []
    for i in range(n_particles_eff):
        final_log_prob_values.append(_safe_eval_log_prob(log_prob, particles[i]))
    final_log_prob_values = gnp.stack(final_log_prob_values)

    info = {
        "options": opts,
        "log_prob_trace": (
            gnp.stack(log_prob_trace) if len(log_prob_trace) > 0 else gnp.empty((0,))
        ),
        "bandwidth_trace": (
            gnp.stack(bandwidth_trace) if len(bandwidth_trace) > 0 else gnp.empty((0,))
        ),
        "temperature_trace": (
            gnp.stack(temperature_trace)
            if len(temperature_trace) > 0
            else gnp.empty((0,))
        ),
        "velocity_norm_trace": (
            gnp.stack(velocity_norm_trace)
            if len(velocity_norm_trace) > 0
            else gnp.empty((0,))
        ),
        "log_prob_final": final_log_prob_values,
        "particles_final": gnp.copy(particles),
    }
    if opts.store_particles_history:
        info["particles_history"] = gnp.stack(particles_history)

    return particles, info


def plot_svgd_empirical_distributions(
    particles_or_info,
    parameter_indices=None,
    parameter_indices_pooled=None,
    bins=50,
):
    """Plot empirical marginal distributions from an SVGD particle cloud.

    Parameters
    ----------
    particles_or_info : array_like or dict
        Final particle cloud with shape ``(n_particles, dim)`` or SVGD info
        dictionary returned by :func:`svgd_sample`. When an info dictionary is
        provided, only particles with finite ``log_prob_final`` are plotted.
    parameter_indices : sequence of int, optional
        Parameter indices to plot individually.
    parameter_indices_pooled : sequence of int, optional
        Parameter indices to plot together on one pooled figure.
    bins : int, default=50
        Number of histogram bins.

    Returns
    -------
    dict
        Dictionary with keys ``"individual"`` and ``"pooled"`` containing the
        created matplotlib figures, or ``None`` when the corresponding plot was
        not requested.
    """
    import matplotlib.pyplot as plt
    from itertools import cycle
    from scipy import stats

    if isinstance(particles_or_info, dict):
        if "particles_final" not in particles_or_info:
            raise ValueError(
                "SVGD info dictionary must contain a 'particles_final' entry."
            )
        particles = particles_or_info["particles_final"]
        log_prob_values = particles_or_info.get("log_prob_final", None)
    else:
        particles = particles_or_info
        log_prob_values = None

    particles = gnp.to_np(gnp.asarray(particles))
    if particles.ndim != 2:
        raise ValueError("particles must have shape (n_particles, dim).")
    if particles.shape[0] == 0:
        raise ValueError("No particles available for plotting.")

    if log_prob_values is not None:
        finite_mask = gnp.to_np(
            gnp.isfinite(gnp.asarray(log_prob_values, dtype=gnp_dtype))
        ).reshape(-1)
        if finite_mask.shape[0] != particles.shape[0]:
            raise ValueError("log_prob_final must have length n_particles.")
        if not finite_mask.any():
            raise ValueError("No finite SVGD particles available for plotting.")
        particles = particles[finite_mask]

    n_particles, dim = particles.shape
    color_cycler = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figs = {"individual": None, "pooled": None}

    if parameter_indices is not None:
        pidx = list(parameter_indices)
        for param in pidx:
            if not 0 <= int(param) < dim:
                raise ValueError("parameter_indices contains an out-of-range index.")
        n_plots = len(pidx)
        desired_height_in = 2.5 * n_plots if n_plots <= 4 else 9
        fig, axes = plt.subplots(
            n_plots, 1, figsize=(10, desired_height_in), sharex=False
        )
        if n_plots == 1:
            axes = [axes]
        colors = cycle(color_cycler)
        for i, param in enumerate(pidx):
            ax = axes[i]
            vals = particles[:, param]
            color = next(colors)
            ax.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.3,
                color=color,
                label=f"$\\theta_{{{int(param)+1}}}$",
            )

            lo, hi = vals.min(), vals.max()
            span = hi - lo
            if span <= 0.0:
                span = 1.0
            xx = gnp.to_np(gnp.linspace(lo - 0.1 * span, hi + 0.1 * span, 100))
            if n_particles >= 2 and span > 0.0:
                kde = stats.gaussian_kde(vals, bw_method="scott")
                ax.plot(xx, kde(xx), color=color)
            ax.set_xlabel(rf"$\theta_{{{int(param)+1}}}$")
            ax.set_ylabel("Density")
            ax.legend()
        plt.tight_layout()
        plt.show()
        figs["individual"] = fig

    if parameter_indices_pooled is not None:
        pidx_pool = list(parameter_indices_pooled)
        for param in pidx_pool:
            if not 0 <= int(param) < dim:
                raise ValueError(
                    "parameter_indices_pooled contains an out-of-range index."
                )
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = cycle(color_cycler)
        for param in pidx_pool:
            vals = particles[:, param]
            color = next(colors)
            ax.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.3,
                color=color,
                label=f"$\\theta_{{{int(param)+1}}}$",
            )
            lo, hi = vals.min(), vals.max()
            span = hi - lo
            if span <= 0.0:
                span = 1.0
            xx = gnp.to_np(gnp.linspace(lo - 0.1 * span, hi + 0.1 * span, 100))
            if n_particles >= 2 and span > 0.0:
                kde = stats.gaussian_kde(vals, bw_method="scott")
                ax.plot(xx, kde(xx), color=color)
        ax.set_xlabel(r"$\theta_i$")
        ax.set_ylabel("Density")
        ax.set_title("Marginal distributions")
        ax.legend()
        plt.tight_layout()
        plt.show()
        figs["pooled"] = fig

    return figs


# ---------------------------
# Example
# ---------------------------


def rosenbrock_U(xy, a: float = 1.0, b: float = 100.0):
    x = xy[0]
    y = xy[1]
    return (a - x) ** 2 + b * (y - x**2) ** 2


def gaussian_2d_log_prob(q, mu, inv_cov, log_det_cov):
    """Log-density of a 2D Gaussian including the normalizing constant."""
    dq = q - mu
    quad = gnp.einsum("i,ij,j->", dq, inv_cov, dq)
    norm_const = 2.0 * gnp.log(2.0 * gnp.pi) + log_det_cov
    return -0.5 * (quad + norm_const)


def gaussian_mixture_2d_log_prob(q, mus, inv_covs, log_det_covs, weights):
    """Stable log-density of a finite 2D Gaussian mixture."""
    log_terms = []
    for k in range(len(weights)):
        log_terms.append(
            gnp.log(weights[k])
            + gaussian_2d_log_prob(q, mus[k], inv_covs[k], log_det_covs[k])
        )
    log_terms = gnp.stack(log_terms)
    m = gnp.max(log_terms)
    return m + gnp.log(gnp.sum(gnp.exp(log_terms - m)))


def plot_svgd_particles(
    particles,
    *,
    title: str,
    objective=None,
    xlim=None,
    ylim=None,
    grid_n: int = 220,
    contour_levels: int = 30,
):
    import numpy as np
    import matplotlib.pyplot as plt

    x = gnp.asarray(particles)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    x = np.asarray(x)

    fig = plt.figure()
    if objective is not None and x.shape[1] == 2:
        if xlim is None:
            xmin, xmax = x[:, 0].min(), x[:, 0].max()
            dx = xmax - xmin
            xlim = (xmin - 0.2 * (dx + 1e-12), xmax + 0.2 * (dx + 1e-12))
        if ylim is None:
            ymin, ymax = x[:, 1].min(), x[:, 1].max()
            dy = ymax - ymin
            ylim = (ymin - 0.2 * (dy + 1e-12), ymax + 0.2 * (dy + 1e-12))

        xs = np.linspace(xlim[0], xlim[1], grid_n)
        ys = np.linspace(ylim[0], ylim[1], grid_n)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)
        for i in range(grid_n):
            for j in range(grid_n):
                Z[i, j] = float(objective(np.array([X[i, j], Y[i, j]])))
        plt.contour(X, Y, Z, levels=contour_levels)

    plt.scatter(x[:, 0], x[:, 1], s=30, alpha=0.8)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title(title)
    plt.show()
    return fig


if __name__ == "__main__":
    import numpy as np

    case = "gaussian2d"  # "rosenbrock" or "gaussian2d"

    if case == "rosenbrock":
        a = 1.0
        b = 100.0
        n_particles = 500
        init_box = [[-1.8, -0.4], [1.8, 2.4]]

        def log_prob(q):
            return -rosenbrock_U(q, a=a, b=b)

        debug_points = [[-1.25, 1.50], [-0.50, 0.25], [1.00, 1.00]]
        print(f"[debug value_and_grad] backend={gnp._gpmp_backend_}")
        for i, q_dbg in enumerate(debug_points):
            q_dbg = gnp.asarray(q_dbg)
            val, grad = gnp.value_and_grad(log_prob, q_dbg)
            print(
                f"[debug value_and_grad] i={i} q={gnp.to_np(q_dbg)} "
                f"value={gnp.to_scalar(val)} grad={gnp.to_np(grad)}"
            )

        particles, info = svgd_sample(
            log_prob=log_prob,
            particles_initial=None,
            n_particles=n_particles,
            dim=2,
            init_box=init_box,
            options=SVGDOptions(
                n_steps=400,
                step_size=2e-2,
                initial_temperature=5.0,
                final_temperature=1.0,
                annealing_schedule="geometric",
                progress=True,
                verbose=1,
                log_every=100,
                store_particles_history=True,
            ),
        )

        print("final particles:")
        print(gnp.to_np(particles))
        print(
            "final mean log_prob:",
            float(gnp.to_np(gnp.mean(info["log_prob_final"]))),
        )
        print(
            "final bandwidth:",
            float(gnp.to_np(info["bandwidth_trace"][-1])),
        )

        plot_svgd_particles(
            particles,
            title="SVGD particles on Rosenbrock",
            objective=lambda xy: rosenbrock_U(xy, a=a, b=b),
            xlim=(-2, 2),
            ylim=(-0.7, 3.8),
            contour_levels=35,
        )

    elif case == "gaussian2d":
        mus_np = [
            np.array([-1.3, 1.0], dtype=float),
            np.array([2.5, -1.0], dtype=float),
        ]
        covs_np = [
            np.array([[0.15, 0.08], [0.08, 0.10]], dtype=float),
            np.array([[0.25, -0.10], [-0.10, 0.20]], dtype=float),
        ]
        weights_np = np.array([0.5, 0.5], dtype=float)
        inv_covs_np = []
        log_det_covs_np = []
        for cov_np in covs_np:
            inv_cov_np = np.linalg.inv(cov_np)
            sign, log_det_cov = np.linalg.slogdet(cov_np)
            if sign <= 0:
                raise RuntimeError("Each covariance matrix must be SPD.")
            inv_covs_np.append(inv_cov_np)
            log_det_covs_np.append(float(log_det_cov))
        n_particles = 500
        init_box = [[-3.0, -2.5], [3.0, 2.5]]

        mus = [gnp.asarray(mu_np, dtype=gnp_dtype) for mu_np in mus_np]
        inv_covs = [
            gnp.asarray(inv_cov_np, dtype=gnp_dtype) for inv_cov_np in inv_covs_np
        ]
        log_det_covs = [
            gnp.asarray(log_det_cov, dtype=gnp_dtype) for log_det_cov in log_det_covs_np
        ]
        weights = gnp.asarray(weights_np, dtype=gnp_dtype)

        def log_prob(q):
            return gaussian_mixture_2d_log_prob(
                q, mus, inv_covs, log_det_covs, weights
            )

        debug_points = [[-1.0, 1.0], [0.0, 0.0], [1.2, -0.8]]
        print(f"[debug value_and_grad] backend={gnp._gpmp_backend_}")
        for i, q_dbg in enumerate(debug_points):
            q_dbg = gnp.asarray(q_dbg)
            val, grad = gnp.value_and_grad(log_prob, q_dbg)
            print(
                f"[debug value_and_grad] i={i} q={gnp.to_np(q_dbg)} "
                f"value={gnp.to_scalar(val)} grad={gnp.to_np(grad)}"
            )

        particles, info = svgd_sample(
            log_prob=log_prob,
            particles_initial=None,
            n_particles=n_particles,
            dim=2,
            init_box=init_box,
            options=SVGDOptions(
                n_steps=300,
                step_size=5e-2,
                initial_temperature=3.0,
                final_temperature=1.0,
                annealing_schedule="geometric",
                progress=True,
                verbose=1,
                log_every=100,
                store_particles_history=True,
            ),
        )

        particles_np = gnp.to_np(particles)
        emp_mean = np.mean(particles_np, axis=0)
        particle_component_scores = np.vstack(
            [
                np.asarray(
                    [
                        float(
                            gaussian_2d_log_prob(
                                gnp.asarray(p, dtype=gnp_dtype),
                                mus[k],
                                inv_covs[k],
                                log_det_covs[k],
                            )
                        )
                        for p in particles_np
                    ]
                )
                for k in range(len(weights_np))
            ]
        ).T
        component_assign = np.argmax(particle_component_scores, axis=1)

        print("[gaussian2d] mixture means:")
        print(np.vstack(mus_np))
        print("[gaussian2d] mixture covariances:")
        for cov_np in covs_np:
            print(cov_np)
        print("[gaussian2d] mixture weights:", weights_np)
        print("[gaussian2d] particle mean:", emp_mean)
        print(
            "[gaussian2d] particle counts by closest component:",
            np.bincount(component_assign, minlength=len(weights_np)),
        )
        print(
            "final mean log_prob:",
            float(gnp.to_np(gnp.mean(info["log_prob_final"]))),
        )
        print(
            "final bandwidth:",
            float(gnp.to_np(info["bandwidth_trace"][-1])),
        )

        mixture_objective = lambda xy: -float(
            gnp.to_np(
                gaussian_mixture_2d_log_prob(
                    gnp.asarray(xy, dtype=gnp_dtype),
                    mus,
                    inv_covs,
                    log_det_covs,
                    weights,
                )
            )
        )
        plot_svgd_particles(
            particles,
            title="SVGD particles on 2D Gaussian mixture",
            objective=mixture_objective,
            xlim=(-4.0, 4.0),
            ylim=(-4.0, 4.0),
            contour_levels=25,
        )

    else:
        raise ValueError("Unknown case. Use 'rosenbrock' or 'gaussian2d'.")
