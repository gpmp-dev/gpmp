# gpmp/mcmc/nuts.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
NUTS (No-U-Turn Sampler) with a gpmp backend.

This file implements a Euclidean-metric NUTS kernel for targets on R^d.
It uses a leapfrog integrator, a slice-variable construction, and a binary tree
expansion with a U-turn stopping rule. It includes warmup with dual-averaging
step size adaptation and windowed diagonal mass adaptation.

Target, potential, Hamiltonian
------------------------------
User provides:
  log_prob(q) -> scalar
with q of shape (dim,).

Define the target density:
  $\\pi(q) \\propto \\exp(\\mathrm{log\\_prob}(q))$.

Define the potential energy:
  $U(q) = -\\mathrm{log\\_prob}(q)$.

Introduce an auxiliary momentum p and define the Hamiltonian:
  $H(q,p) = U(q) + K(p)$.

Diagonal mass matrix and kinetic energy
---------------------------------------
This implementation uses a diagonal positive vector mass_diag and its inverse
inv_mass_diag, with:
  $M = \\mathrm{diag}(\\mathrm{mass\\_diag})$,
  $M^{-1} = \\mathrm{diag}(\\mathrm{inv\\_mass\\_diag})$,
  $\\mathrm{inv\\_mass\\_diag} = 1 / \\mathrm{mass\\_diag}$.

Momentum is sampled as:
  $p \\sim \\mathcal{N}(0, M)$,
implemented as p = randn(*) * sqrt(mass_diag).

Kinetic energy is:
  $K(p) = \\tfrac12 p^\\top M^{-1} p = \\tfrac12 \\sum_i p_i^2 \\,(M^{-1})_{ii}$.

Hamiltonian dynamics then satisfy:
  $\\dot q = \\nabla_p K(p) = M^{-1} p$,
  $\\dot p = -\\nabla_q U(q)$.

Leapfrog integrator
-------------------
One leapfrog step with step size eps updates:
  p_{n+1/2} = p_n - (eps/2) * gradU(q_n)
  q_{n+1}   = q_n + eps * (M^{-1} p_{n+1/2})
  p_{n+1}   = p_{n+1/2} - (eps/2) * gradU(q_{n+1})

The function potential_and_grad returns (U(q), gradU(q)) with:
  U(q) = -log_prob(q),
  gradU(q) = \\nabla_q U(q).

Slice variable construction
---------------------------
At the start of one NUTS transition, sample momentum p0 and compute:
  H0 = U(q0) + K(p0).

Sample a slice variable:
  u ~ Uniform(0, exp(-H0)),
and work with log_u = log(u):
  log_u = -H0 + log(rand()).

A state (q,p) is "valid" for the slice if:
  log_u <= -H(q,p).

Divergences
-----------
Define the energy error at a candidate state:
  DeltaH = H(q,p) - H0.

A divergence is flagged if:
  DeltaH > delta_max,
or if H becomes NaN/Inf.

The recursive tree builder also uses the continuation test:
  log_u < delta_max - H(q,p),
equivalent to DeltaH < delta_max in log space.

No-U-Turn stopping rule (diagonal metric)
-----------------------------------------
Let dq = q_plus - q_minus.
The correct U-turn check uses the velocity v = M^{-1} p, not p itself.

For diagonal M:
  v_minus = M^{-1} p_minus = inv_mass_diag * p_minus
  v_plus  = M^{-1} p_plus  = inv_mass_diag * p_plus

Stop if:
  dq^\\top v_minus < 0  or  dq^\\top v_plus < 0.

Using dq^\\top p_{\\pm} is only correct when M = I.

Tree building and proposal selection
------------------------------------
The tree is expanded by doubling. At depth j, the subtree contains 2^j leapfrog
steps in direction v in {+1,-1}. The recursion returns:
- left edge (q_minus, p_minus) and right edge (q_plus, p_plus),
- a candidate q_prop drawn from the set of valid states in the subtree,
- n_valid, the number of valid states in the subtree,
- an acceptance statistic accumulator (alpha_sum, n_alpha),
- flags for continuation and divergence.

Candidate selection uses multinomial-style selection:
- when combining two subtrees with n_valid_left and n_valid_right,
  pick the right candidate with probability
  n_valid_right / (n_valid_left + n_valid_right).

Acceptance statistic for adaptation
-----------------------------------
During tree construction, accumulate:
  alpha = exp(min(0, log_alpha)),
  log_alpha = -(H(q,p) - H0).

Return:
  accept_stat = alpha_sum / n_alpha.

This is used by dual averaging. It is not a final Metropolis accept/reject
probability, since NUTS selects from a set of valid states.

Warmup and adaptation
---------------------
Warmup uses:
1) Dual averaging to adapt eps toward target_accept, updated every warmup iteration.
2) Windowed diagonal mass adaptation:
   - an initial buffer with no mass updates,
   - several middle windows: accumulate q values and update mass_diag at each window end,
   - a terminal buffer with no mass updates.

After each mass update:
- inv_mass_diag is updated,
- dual averaging is restarted, since eps must be retuned for the new metric.

Numerical notes
---------------
- Comparisons and control flow require Python booleans. If gnp returns 0-d arrays,
  convert comparisons with bool(...).
- For acceptance computations, use exp(min(0, log_alpha)) to avoid alpha > 1.


References
----------
[1] R. M. Neal (2011). "MCMC Using Hamiltonian Dynamics." In: Handbook of Markov Chain Monte Carlo. https://arxiv.org/abs/1206.1901
[2] M. D. Hoffman and A. Gelman (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." JMLR 15:1593-1623.
[3] M. Betancourt (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
[4] Y. Nesterov (2009). "Primal-dual subgradient methods for convex problems." Mathematical Programming 120(1):221-259. (Dual averaging.)
[5] B. P. Welford (1962). "Note on a method for calculating corrected sums of squares and products." Technometrics 4(3):419-420. (Online variance.)
[6] Stan Development Team (ongoing). Stan Reference Manual, sections on HMC/NUTS and adaptation. (Practical windowed adaptation scheme.)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import math
import os
import time

import gpmp.num as gnp

ArrayLike = any  # Placeholder for unified array type

gnp_dtype = gnp.get_dtype()

_DEFAULT_NUM_WARMUP = 1000
_DEFAULT_TARGET_ACCEPT = 0.80
_DEFAULT_MAX_DEPTH = 10
_DEFAULT_DELTA_MAX = 1000.0
_DEFAULT_JITTER = 1e-4
_DEFAULT_PROGRESS = True
_DEFAULT_VERBOSE = 1
_DEFAULT_LOG_EVERY = 50


@dataclass
class NUTSOptions:
    """Configuration object for NUTS sampling and warmup adaptation."""

    # Main sampler settings
    num_warmup: int = _DEFAULT_NUM_WARMUP
    target_accept: gnp_dtype = _DEFAULT_TARGET_ACCEPT
    max_depth: int = _DEFAULT_MAX_DEPTH
    delta_max: gnp_dtype = _DEFAULT_DELTA_MAX
    jitter: gnp_dtype = _DEFAULT_JITTER
    init_step_size: Optional[gnp_dtype] = None
    init_mass_diag: Optional[ArrayLike] = None
    seed: Optional[int] = None
    progress: bool = _DEFAULT_PROGRESS
    verbose: int = _DEFAULT_VERBOSE
    log_every: int = _DEFAULT_LOG_EVERY

    # Dual-averaging hyperparameters
    dual_averaging_gamma: gnp_dtype = 0.05
    dual_averaging_t0: gnp_dtype = 10.0
    dual_averaging_kappa: gnp_dtype = 0.75
    dual_averaging_mu_factor: gnp_dtype = 10.0

    # Warmup window policy
    warmup_min_no_window: int = 20
    warmup_large_threshold: int = 150
    warmup_large_init_buffer: int = 75
    warmup_large_term_buffer: int = 50
    warmup_large_base_window: int = 25
    warmup_init_buffer_ratio: gnp_dtype = 0.15
    warmup_term_buffer_ratio: gnp_dtype = 0.10
    warmup_base_window_divisor: gnp_dtype = 3.0

    # Initial step-size search policy
    find_eps_init: gnp_dtype = 1.0
    find_eps_target_accept: gnp_dtype = 0.5
    find_eps_scale_base: gnp_dtype = 2.0
    find_eps_min: gnp_dtype = 1e-6
    find_eps_max: gnp_dtype = 1e2


def _resolve_nuts_options(
    options: Optional["NUTSOptions"],
    *,
    num_warmup: int,
    target_accept: gnp_dtype,
    max_depth: int,
    delta_max: gnp_dtype,
    jitter: gnp_dtype,
    init_step_size: Optional[gnp_dtype],
    init_mass_diag: Optional[ArrayLike],
    seed: Optional[int],
    progress: bool,
    verbose: int,
    log_every: int,
) -> "NUTSOptions":
    """Merge default nuts_sample args with an optional NUTSOptions object.

    Rule:
    - If `options` is None, use default args as-is.
    - If `options` is provided, only non-default default args override it.
    """
    opts = replace(options) if options is not None else NUTSOptions()

    if options is None or num_warmup != _DEFAULT_NUM_WARMUP:
        opts.num_warmup = num_warmup
    if options is None or target_accept != _DEFAULT_TARGET_ACCEPT:
        opts.target_accept = target_accept
    if options is None or max_depth != _DEFAULT_MAX_DEPTH:
        opts.max_depth = max_depth
    if options is None or delta_max != _DEFAULT_DELTA_MAX:
        opts.delta_max = delta_max
    if options is None or jitter != _DEFAULT_JITTER:
        opts.jitter = jitter
    if options is None or init_step_size is not None:
        opts.init_step_size = init_step_size
    if options is None or init_mass_diag is not None:
        opts.init_mass_diag = init_mass_diag
    if options is None or seed is not None:
        opts.seed = seed
    if options is None or progress != _DEFAULT_PROGRESS:
        opts.progress = progress
    if options is None or verbose != _DEFAULT_VERBOSE:
        opts.verbose = verbose
    if options is None or log_every != _DEFAULT_LOG_EVERY:
        opts.log_every = log_every

    return opts


# ---------------------------
# Logging
# ---------------------------


class SimpleLogger:
    """
    verbose:
      0: silent
      1: phase + window events + periodic summaries
      2: more frequent summaries
    """

    def __init__(self, verbose: int = 1):
        self.verbose = int(verbose)

    def log(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)


# ---------------------------
# Adaptation utilities
# ---------------------------


@dataclass
class DualAveragingState:
    mu: gnp_dtype
    log_eps: gnp_dtype
    log_eps_bar: gnp_dtype
    h_bar: gnp_dtype
    t: int

    def update(
        self,
        accept_stat: gnp_dtype,
        target: gnp_dtype = 0.80,
        gamma: gnp_dtype = 0.05,
        t0: gnp_dtype = 10.0,
        kappa: gnp_dtype = 0.75,
    ) -> gnp_dtype:
        self.t += 1
        eta = 1.0 / (self.t + t0)
        self.h_bar = (1.0 - eta) * self.h_bar + eta * (target - accept_stat)
        self.log_eps = self.mu - (math.sqrt(self.t) / gamma) * self.h_bar
        w = self.t ** (-kappa)
        self.log_eps_bar = w * self.log_eps + (1.0 - w) * self.log_eps_bar
        return math.exp(self.log_eps)

    def final(self) -> gnp_dtype:
        return math.exp(self.log_eps_bar)


class RunningDiagVar:
    def __init__(self, dim: int):
        self.n = 0
        self.mean = gnp.zeros(dim)
        self.m2 = gnp.zeros(dim)

    def update_one(self, x: ArrayLike) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = x - self.mean
        self.m2 = self.m2 + delta * delta2

    def update_batch(self, x: ArrayLike) -> None:
        # x: (batch, dim)
        for i in range(x.shape[0]):
            self.update_one(x[i])

    def var(self) -> ArrayLike:
        if self.n < 2:
            return gnp.ones_like(self.mean)
        return self.m2 / (self.n - 1)


def make_warmup_windows(
    num_warmup: int,
    *,
    min_no_window: int = 20,
    large_threshold: int = 150,
    large_init_buffer: int = 75,
    large_term_buffer: int = 50,
    large_base_window: int = 25,
    init_buffer_ratio: gnp_dtype = 0.15,
    term_buffer_ratio: gnp_dtype = 0.10,
    base_window_divisor: gnp_dtype = 3.0,
) -> List[Tuple[int, int]]:
    """
    Stan-like warmup windows for diagonal mass adaptation.
    Mass is updated at the end of each window.
    """
    if num_warmup <= min_no_window:
        return []

    if num_warmup >= large_threshold:
        init_buffer = large_init_buffer
        term_buffer = large_term_buffer
        base_window = large_base_window
    else:
        init_buffer = max(1, int(init_buffer_ratio * num_warmup))
        term_buffer = max(1, int(term_buffer_ratio * num_warmup))
        base_window = max(
            1,
            int((num_warmup - init_buffer - term_buffer) / base_window_divisor),
        )

    start = init_buffer
    end_middle = num_warmup - term_buffer
    if end_middle <= start:
        return []

    win = min(base_window, end_middle - start)
    windows: List[Tuple[int, int]] = []
    while start + win < end_middle:
        windows.append((start, start + win))
        start += win
        win = min(2 * win, end_middle - start)
        if win <= 0:
            break
    if start < end_middle:
        windows.append((start, end_middle))
    return windows


def describe_windows(windows: List[Tuple[int, int]]) -> str:
    if not windows:
        return "no mass adaptation windows"
    parts = []
    for a, b in windows:
        parts.append(f"[{a},{b})")
    return "mass windows: " + " ".join(parts)


# ---------------------------
# Autograd helpers
# ---------------------------

def potential_and_grad(
    log_prob: Callable[[ArrayLike], ArrayLike],
    q: ArrayLike,
    *,
    use_helper: bool = True,
) -> Tuple[ArrayLike, ArrayLike]:

    nlp = lambda q: -log_prob(q)
    lp, glp = gnp.value_and_grad(nlp, q)
    return lp, glp


def kinetic(p: ArrayLike, inv_mass_diag: ArrayLike) -> ArrayLike:
    return 0.5 * gnp.sum(p * p * inv_mass_diag)


def leapfrog(
    log_prob: Callable[[ArrayLike], ArrayLike],
    q: ArrayLike,
    p: ArrayLike,
    gradU: ArrayLike,
    eps: gnp_dtype,
    inv_mass_diag: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    p_half = p - 0.5 * eps * gradU
    q_new = q + eps * (p_half * inv_mass_diag)
    U_new, g_new = potential_and_grad(log_prob, q_new)
    p_new = p_half - 0.5 * eps * g_new
    return q_new, p_new, U_new, g_new


def is_uturn(q_minus, q_plus, p_minus, p_plus, inv_mass_diag):
    dq = q_plus - q_minus
    v_minus = inv_mass_diag * p_minus
    v_plus = inv_mass_diag * p_plus
    return bool(gnp.sum(dq * v_minus) < 0.0) or bool(gnp.sum(dq * v_plus) < 0.0)


# def is_uturn(q_minus: ArrayLike, q_plus: ArrayLike, p_minus: ArrayLike, p_plus: ArrayLike) -> bool:
#     dq = q_plus - q_minus
#     return (gnp.sum(dq * p_minus) < 0.0) or (gnp.sum(dq * p_plus) < 0.0)


def find_reasonable_step_size(
    log_prob: Callable[[ArrayLike], ArrayLike],
    q: ArrayLike,
    inv_mass_diag: ArrayLike,
    init_eps: gnp_dtype = 1.0,
    target_accept: gnp_dtype = 0.5,
    scale_base: gnp_dtype = 2.0,
    min_eps: gnp_dtype = 1e-6,
    max_eps: gnp_dtype = 1e2,
) -> gnp_dtype:
    eps = float(init_eps)
    mass_diag = 1.0 / inv_mass_diag
    p0 = gnp.randn(*q.shape) * gnp.sqrt(mass_diag)

    U0, g0 = potential_and_grad(log_prob, q)
    H0 = U0 + kinetic(p0, inv_mass_diag)

    q1, p1, U1, _ = leapfrog(log_prob, q, p0, g0, eps, inv_mass_diag)
    H1 = U1 + kinetic(p1, inv_mass_diag)

    log_alpha = -(H1 - H0)
    alpha = float(gnp.exp(gnp.clip(log_alpha, max=0.0)))
    direction = 1.0 if alpha > target_accept else -1.0

    while True:
        eps *= scale_base**direction
        q1, p1, U1, _ = leapfrog(log_prob, q, p0, g0, eps, inv_mass_diag)
        H1 = U1 + kinetic(p1, inv_mass_diag)
        log_alpha = -(H1 - H0)
        alpha2 = float(gnp.exp(gnp.clip(log_alpha, max=0.0)))
        if (alpha2 > target_accept and direction < 0) or (
            alpha2 < target_accept and direction > 0
        ):
            break
        if eps < min_eps or eps > max_eps:
            break

    return float(eps)


# ---------------------------
# NUTS transition
# ---------------------------


def build_tree(
    log_prob: Callable[[ArrayLike], ArrayLike],
    q: ArrayLike,
    p: ArrayLike,
    gradU: ArrayLike,
    log_u: ArrayLike,
    v: int,
    depth: int,
    eps: gnp_dtype,
    inv_mass_diag: ArrayLike,
    H0: ArrayLike,
    delta_max: gnp_dtype,
):
    """
    Returns:
      q_minus, p_minus, grad_minus,
      q_plus,  p_plus,  grad_plus,
      q_proposal, n_valid, s_continue,
      alpha_sum, n_alpha, n_leapfrog, divergent
    """
    if depth == 0:
        q1, p1, U1, g1 = leapfrog(log_prob, q, p, gradU, eps * v, inv_mass_diag)
        H1 = U1 + kinetic(p1, inv_mass_diag)

        if gnp.isnan(H1) or gnp.isinf(H1):
            return q, p, gradU, q, p, gradU, q, 0, False, 0.0, 0, 1, True

        n_valid = 1 if (log_u <= -H1) else 0
        energy_error = H1 - H0
        divergent = energy_error > delta_max
        s_continue = (log_u < (delta_max - H1)) and (not divergent)

        log_alpha = -(H1 - H0)
        alpha = float(gnp.exp(gnp.clip(log_alpha, max=0.0)))
        if alpha > 1.0:
            alpha = 1.0

        return q1, p1, g1, q1, p1, g1, q1, n_valid, s_continue, alpha, 1, 1, divergent

    out = build_tree(
        log_prob, q, p, gradU, log_u, v, depth - 1, eps, inv_mass_diag, H0, delta_max
    )
    (
        q_minus,
        p_minus,
        g_minus,
        q_plus,
        p_plus,
        g_plus,
        q_prop,
        n_valid,
        s_continue,
        alpha_sum,
        n_alpha,
        n_leapfrog,
        divergent,
    ) = out

    if s_continue and (not divergent):
        if v == -1:
            out2 = build_tree(
                log_prob,
                q_minus,
                p_minus,
                g_minus,
                log_u,
                v,
                depth - 1,
                eps,
                inv_mass_diag,
                H0,
                delta_max,
            )
            (
                q_minus2,
                p_minus2,
                g_minus2,
                _,
                _,
                _,
                q_prop2,
                n_valid2,
                s2,
                alpha_sum2,
                n_alpha2,
                nlf2,
                div2,
            ) = out2
            q_minus, p_minus, g_minus = q_minus2, p_minus2, g_minus2
        else:
            out2 = build_tree(
                log_prob,
                q_plus,
                p_plus,
                g_plus,
                log_u,
                v,
                depth - 1,
                eps,
                inv_mass_diag,
                H0,
                delta_max,
            )
            (
                _,
                _,
                _,
                q_plus2,
                p_plus2,
                g_plus2,
                q_prop2,
                n_valid2,
                s2,
                alpha_sum2,
                n_alpha2,
                nlf2,
                div2,
            ) = out2
            q_plus, p_plus, g_plus = q_plus2, p_plus2, g_plus2

        if (n_valid + n_valid2) > 0:
            u = gnp.rand()
            if u < (n_valid2 / (n_valid + n_valid2)):
                q_prop = q_prop2

        n_valid += n_valid2
        s_continue = s2 and (
            not is_uturn(q_minus, q_plus, p_minus, p_plus, inv_mass_diag)
        )
        alpha_sum += alpha_sum2
        n_alpha += n_alpha2
        n_leapfrog += nlf2
        divergent = divergent or div2

    return (
        q_minus,
        p_minus,
        g_minus,
        q_plus,
        p_plus,
        g_plus,
        q_prop,
        n_valid,
        s_continue,
        alpha_sum,
        n_alpha,
        n_leapfrog,
        divergent,
    )


def nuts_transition(
    log_prob: Callable[[ArrayLike], ArrayLike],
    q0: ArrayLike,
    step_size: gnp_dtype,
    inv_mass_diag: ArrayLike,
    max_depth: int,
    delta_max: gnp_dtype,
) -> Tuple[ArrayLike, gnp_dtype, int, int, bool]:
    mass_diag = 1.0 / inv_mass_diag
    p0 = gnp.randn(*q0.shape) * gnp.sqrt(mass_diag)

    U0, g0 = potential_and_grad(log_prob, q0)
    H0 = U0 + kinetic(p0, inv_mass_diag)
    if gnp.isnan(H0) or gnp.isinf(H0):
        return q0, 0.0, 0, 0, True

    log_u = -H0 + gnp.log(gnp.rand())

    q_minus = gnp.copy(q0)
    q_plus = gnp.copy(q0)
    p_minus = gnp.copy(p0)
    p_plus = gnp.copy(p0)
    g_minus = gnp.copy(g0)
    g_plus = gnp.copy(g0)

    q_prop = gnp.copy(q0)
    n_valid = 1
    s_continue = True

    alpha_sum = 0.0
    n_alpha = 0
    divergent_any = False
    depth = 0
    n_leapfrog = 0

    while s_continue and depth < max_depth:
        v = -1 if gnp.rand() < 0.5 else 1

        if v == -1:
            out = build_tree(
                log_prob,
                q_minus,
                p_minus,
                g_minus,
                log_u,
                v,
                depth,
                step_size,
                inv_mass_diag,
                H0,
                delta_max,
            )
            (
                q_minus,
                p_minus,
                g_minus,
                _,
                _,
                _,
                q_prop2,
                n_valid2,
                s2,
                alpha2,
                n_alpha2,
                nlf2,
                div2,
            ) = out
        else:
            out = build_tree(
                log_prob,
                q_plus,
                p_plus,
                g_plus,
                log_u,
                v,
                depth,
                step_size,
                inv_mass_diag,
                H0,
                delta_max,
            )
            (
                _,
                _,
                _,
                q_plus,
                p_plus,
                g_plus,
                q_prop2,
                n_valid2,
                s2,
                alpha2,
                n_alpha2,
                nlf2,
                div2,
            ) = out

        if s2 and (not div2) and (n_valid + n_valid2) > 0:
            u = gnp.rand()
            if u < (n_valid2 / (n_valid + n_valid2)):
                q_prop = q_prop2

        n_valid += n_valid2
        s_continue = s2 and (
            not is_uturn(q_minus, q_plus, p_minus, p_plus, inv_mass_diag)
        )
        alpha_sum += alpha2
        n_alpha += n_alpha2
        n_leapfrog += nlf2
        divergent_any = divergent_any or div2
        depth += 1

    accept_stat = alpha_sum / max(1, n_alpha)
    return q_prop, float(accept_stat), int(n_leapfrog), int(depth), bool(divergent_any)


# ---------------------------
# Sampling driver
# ---------------------------


def nuts_sample(
    log_prob: Callable[[ArrayLike], ArrayLike],
    q_init: ArrayLike,
    num_samples: int,
    num_warmup: int = _DEFAULT_NUM_WARMUP,
    target_accept: gnp_dtype = _DEFAULT_TARGET_ACCEPT,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    delta_max: gnp_dtype = _DEFAULT_DELTA_MAX,
    jitter: gnp_dtype = _DEFAULT_JITTER,
    init_step_size: Optional[gnp_dtype] = None,
    init_mass_diag: Optional[ArrayLike] = None,
    seed: Optional[int] = None,
    progress: bool = _DEFAULT_PROGRESS,
    verbose: int = _DEFAULT_VERBOSE,
    log_every: int = _DEFAULT_LOG_EVERY,
    options: Optional[NUTSOptions] = None,
) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
    """q_init: (chains, dim)
    log_prob: takes (dim,) and returns scalar

    options:
      Optional NUTSOptions object. Default keyword arguments override
      `options` when set to non-default values.

    verbose:
      0: silent
      1: phase + window events + periodic summaries
      2: more frequent summaries

    """
    if q_init.ndim != 2:
        raise ValueError("q_init must have shape (chains, dim)")

    opts = _resolve_nuts_options(
        options,
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
    )

    num_warmup = int(opts.num_warmup)
    target_accept = float(opts.target_accept)
    max_depth = int(opts.max_depth)
    delta_max = float(opts.delta_max)
    jitter = float(opts.jitter)
    init_step_size = opts.init_step_size
    init_mass_diag = opts.init_mass_diag
    seed = opts.seed
    progress = bool(opts.progress)
    verbose = int(opts.verbose)
    log_every = int(opts.log_every)

    logger = SimpleLogger(verbose=verbose)

    chains, dim = q_init.shape
    eps_min = float(opts.find_eps_min)
    eps_max = float(opts.find_eps_max)
    if (not math.isfinite(eps_min)) or eps_min <= 0.0:
        eps_min = 1e-12
    if (not math.isfinite(eps_max)) or eps_max <= eps_min:
        eps_max = max(1.0, 10.0 * eps_min)

    def _clamp_step_size(eps: gnp_dtype) -> float:
        eps = float(eps)
        if (not math.isfinite(eps)) or eps <= 0.0:
            return eps_min
        if eps < eps_min:
            return eps_min
        if eps > eps_max:
            return eps_max
        return eps

    logger.log(f"chains={chains}, dim={dim}", level=1)
    logger.log(f"num_warmup={num_warmup}, num_samples={num_samples}", level=1)
    logger.log(
        f"target_accept={target_accept}, max_depth={max_depth}, delta_max={delta_max}",
        level=1,
    )

    if seed is not None:
        gnp.set_seed(seed)
        logger.log(f"seed={seed}", level=1)

    if init_mass_diag is None:
        mass_diag = gnp.ones(dim)
        logger.log("mass_diag init: identity (ones)", level=1)
    else:
        if init_mass_diag.shape != (dim,):
            raise ValueError("init_mass_diag must have shape (dim,)")
        mass_diag = gnp.clip(gnp.asarray(init_mass_diag), min=jitter)
        logger.log("mass_diag init: provided (clamped)", level=1)

    inv_mass_diag = 1.0 / mass_diag

    if init_step_size is None:
        t0 = time.time()
        eps0 = find_reasonable_step_size(
            log_prob,
            q_init[0],
            inv_mass_diag,
            init_eps=opts.find_eps_init,
            target_accept=opts.find_eps_target_accept,
            scale_base=opts.find_eps_scale_base,
            min_eps=opts.find_eps_min,
            max_eps=opts.find_eps_max,
        )
        logger.log(
            f"initial step size heuristic: eps0={eps0:.6g} (took {time.time()-t0:.2f}s)",
            level=1,
        )
    else:
        eps0 = float(init_step_size)
        logger.log(f"initial step size: provided eps0={eps0:.6g}", level=1)
    eps0 = _clamp_step_size(eps0)
    mu0 = max(eps_min, float(opts.dual_averaging_mu_factor) * eps0)

    da = DualAveragingState(
        mu=math.log(mu0),
        log_eps=math.log(eps0),
        log_eps_bar=math.log(eps0),
        h_bar=0.0,
        t=0,
    )
    step_size = eps0

    windows = make_warmup_windows(
        num_warmup,
        min_no_window=opts.warmup_min_no_window,
        large_threshold=opts.warmup_large_threshold,
        large_init_buffer=opts.warmup_large_init_buffer,
        large_term_buffer=opts.warmup_large_term_buffer,
        large_base_window=opts.warmup_large_base_window,
        init_buffer_ratio=opts.warmup_init_buffer_ratio,
        term_buffer_ratio=opts.warmup_term_buffer_ratio,
        base_window_divisor=opts.warmup_base_window_divisor,
    )
    window_end_set = {end for (_, end) in windows}
    logger.log(describe_windows(windows), level=1)
    rv = RunningDiagVar(dim)

    q = gnp.copy(q_init)

    # traces
    warmup_accept = gnp.empty((num_warmup, chains), dtype=gnp_dtype)
    warmup_div = gnp.empty((num_warmup, chains), dtype=bool)
    warmup_depth = gnp.empty((num_warmup, chains), dtype=int)
    warmup_nlf = gnp.empty((num_warmup, chains), dtype=int)
    warmup_eps = gnp.empty(num_warmup, dtype=gnp_dtype)

    # tqdm
    use_tqdm = False
    pbar_warm = None
    pbar_samp = None
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            use_tqdm = True
            pbar_warm = tqdm(range(num_warmup), desc="warmup", leave=True)
        except Exception:
            use_tqdm = False

    it_warm = pbar_warm if use_tqdm else range(num_warmup)

    logger.log("warmup: start", level=1)
    t_warm0 = time.time()

    for t in it_warm:
        acc_sum = 0.0
        div_sum = 0

        for c in range(chains):
            q_new, a, nlf, depth, div = nuts_transition(
                log_prob=log_prob,
                q0=q[c],
                step_size=step_size,
                inv_mass_diag=inv_mass_diag,
                max_depth=max_depth,
                delta_max=delta_max,
            )
            q[c] = q_new
            warmup_accept[t, c] = a
            warmup_div[t, c] = div
            warmup_depth[t, c] = depth
            warmup_nlf[t, c] = nlf
            acc_sum += a
            div_sum += int(div)

        warmup_eps[t] = float(step_size)
        mean_accept = float(acc_sum / chains)
        mean_div = float(div_sum / chains)

        step_size = da.update(
            mean_accept,
            target=target_accept,
            gamma=opts.dual_averaging_gamma,
            t0=opts.dual_averaging_t0,
            kappa=opts.dual_averaging_kappa,
        )
        step_size = _clamp_step_size(step_size)

        # mass window membership
        in_mass_window = False
        for start, end in windows:
            if start <= t < end:
                in_mass_window = True
                break
        if in_mass_window:
            rv.update_batch(q)

        # window end: update mass, restart dual averaging
        if (t + 1) in window_end_set:
            old_mass_mean = float(gnp.mean(mass_diag))
            mass_diag = gnp.clip(rv.var(), min=jitter)
            inv_mass_diag = 1.0 / mass_diag
            new_mass_mean = float(gnp.mean(mass_diag))

            logger.log(
                f"warmup iter {t+1}: mass update at window end; mean(mass_diag) {old_mass_mean:.6g} -> {new_mass_mean:.6g}",
                level=1,
            )

            rv = RunningDiagVar(dim)
            step_size = _clamp_step_size(step_size)
            mu_ref = max(eps_min, float(opts.dual_averaging_mu_factor) * step_size)
            da = DualAveragingState(
                mu=math.log(mu_ref),
                log_eps=math.log(step_size),
                log_eps_bar=math.log(step_size),
                h_bar=0.0,
                t=0,
            )
            logger.log(
                f"warmup iter {t+1}: dual averaging restart; eps={step_size:.6g}",
                level=1,
            )

        # periodic warmup logs
        do_log = ((t + 1) % max(1, log_every) == 0) or (t == 0) or (t + 1 == num_warmup)
        if verbose >= 2:
            do_log = ((t + 1) % max(1, log_every // 5) == 0) or do_log

        if do_log:
            logger.log(
                f"warmup iter {t+1}/{num_warmup}: eps={step_size:.6g}, mean_accept={mean_accept:.3f}, div_rate={mean_div:.3f}",
                level=1,
            )

        if use_tqdm and pbar_warm is not None:
            pbar_warm.set_postfix(
                eps=f"{step_size:.3g}",
                acc=f"{mean_accept:.3f}",
                div=f"{mean_div:.3f}",
            )

    warmup_time = time.time() - t_warm0
    step_size_final = _clamp_step_size(da.final())
    step_size = step_size_final

    logger.log(f"warmup: done in {warmup_time:.2f}s", level=1)
    logger.log(f"warmup: step_size_final={step_size_final:.6g}", level=1)
    logger.log(
        f"warmup: mass_diag_final mean={float(gnp.mean(mass_diag)):.6g}", level=1
    )

    # sampling traces
    samples = gnp.empty((num_samples, chains, dim), dtype=gnp_dtype)
    accept = gnp.empty((num_samples, chains), dtype=gnp_dtype)
    divergent = gnp.empty((num_samples, chains), dtype=bool)
    tree_depth = gnp.empty((num_samples, chains), dtype=int)
    n_leapfrog = gnp.empty((num_samples, chains), dtype=int)

    if use_tqdm:
        try:
            from tqdm.auto import tqdm  # type: ignore

            pbar_samp = tqdm(range(num_samples), desc="sample", leave=True)
        except Exception:
            pbar_samp = None

    it_samp = pbar_samp if pbar_samp is not None else range(num_samples)

    logger.log("sample: start", level=1)
    t_samp0 = time.time()

    for t in it_samp:
        acc_sum = 0.0
        div_sum = 0

        for c in range(chains):
            q_new, a, nlf, depth, div = nuts_transition(
                log_prob=log_prob,
                q0=q[c],
                step_size=step_size,
                inv_mass_diag=inv_mass_diag,
                max_depth=max_depth,
                delta_max=delta_max,
            )
            q[c] = q_new
            samples[t, c] = q_new
            accept[t, c] = a
            divergent[t, c] = div
            tree_depth[t, c] = depth
            n_leapfrog[t, c] = nlf
            acc_sum += a
            div_sum += int(div)

        mean_accept = float(acc_sum / chains)
        mean_div = float(div_sum / chains)

        do_log = (
            ((t + 1) % max(1, log_every) == 0) or (t == 0) or (t + 1 == num_samples)
        )
        if verbose >= 2:
            do_log = ((t + 1) % max(1, log_every // 5) == 0) or do_log

        if do_log:
            logger.log(
                f"sample iter {t+1}/{num_samples}: mean_accept={mean_accept:.3f}, div_rate={mean_div:.3f}",
                level=1,
            )

        if pbar_samp is not None:
            pbar_samp.set_postfix(
                eps=f"{step_size:.3g}",
                acc=f"{mean_accept:.3f}",
                div=f"{mean_div:.3f}",
            )

    samp_time = time.time() - t_samp0
    logger.log(f"sample: done in {samp_time:.2f}s", level=1)

    info = {
        "warmup_step_size": warmup_eps,
        "warmup_accept_stat": warmup_accept,
        "warmup_divergent": warmup_div,
        "warmup_tree_depth": warmup_depth,
        "warmup_n_leapfrog": warmup_nlf,
        "accept_stat": accept,
        "divergent": divergent,
        "tree_depth": tree_depth,
        "n_leapfrog": n_leapfrog,
        "step_size_final": gnp.asarray(step_size_final),
        "mass_diag_final": gnp.copy(mass_diag),
    }
    return samples, info


# ---------------------------
# Diagnostics plots
# ---------------------------


def moving_average(y, window: int):
    if window <= 1:
        return y

    w = gnp.ones(window, dtype=gnp_dtype) / window
    return gnp.convolve(y, w, mode="valid")


def plot_nuts_diagnostics(
    info: Dict[str, ArrayLike],
    window: int = 50,
    show: bool = True,
    save_dir: Optional[str] = None,
):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "plot_nuts_diagnostics requires numpy and matplotlib."
        ) from e

    figs = []

    def to_numpy(x: ArrayLike) -> np.ndarray:
        """Convert backend-agnostic array to numpy."""
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    warm_eps = to_numpy(info["warmup_step_size"])
    warm_acc_raw = to_numpy(info["warmup_accept_stat"])
    warm_acc = np.mean(warm_acc_raw, axis=1) if warm_acc_raw.ndim > 1 else warm_acc_raw

    warm_div_raw = to_numpy(
        info["warmup_divergent"].astype(float)
        if hasattr(info["warmup_divergent"], "astype")
        else info["warmup_divergent"]
    )
    warm_div = np.mean(warm_div_raw, axis=1) if warm_div_raw.ndim > 1 else warm_div_raw

    samp_acc_raw = to_numpy(info["accept_stat"])
    samp_acc = np.mean(samp_acc_raw, axis=1) if samp_acc_raw.ndim > 1 else samp_acc_raw

    samp_div_raw = to_numpy(
        info["divergent"].astype(float)
        if hasattr(info["divergent"], "astype")
        else info["divergent"]
    )
    samp_div = np.mean(samp_div_raw, axis=1) if samp_div_raw.ndim > 1 else samp_div_raw

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure()
    plt.plot(warm_eps)
    plt.xlabel("warmup iteration")
    plt.ylabel("step size")
    figs.append(fig)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "warmup_step_size.png"), dpi=150)

    fig = plt.figure()
    plt.plot(warm_acc)
    if window > 1 and len(warm_acc) >= window:
        ma = moving_average(warm_acc, window)
        plt.plot(np.arange(window - 1, len(warm_acc)), ma)
    plt.xlabel("warmup iteration")
    plt.ylabel("mean accept stat")
    figs.append(fig)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "warmup_accept.png"), dpi=150)

    fig = plt.figure()
    plt.plot(warm_div)
    plt.xlabel("warmup iteration")
    plt.ylabel("divergence rate")
    figs.append(fig)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "warmup_divergence.png"), dpi=150)

    fig = plt.figure()
    plt.plot(samp_acc)
    if window > 1 and len(samp_acc) >= window:
        ma = moving_average(samp_acc, window)
        plt.plot(np.arange(window - 1, len(samp_acc)), ma)
    plt.xlabel("sample iteration")
    plt.ylabel("mean accept stat")
    figs.append(fig)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "sample_accept.png"), dpi=150)

    fig = plt.figure()
    plt.plot(samp_div)
    plt.xlabel("sample iteration")
    plt.ylabel("divergence rate")
    figs.append(fig)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "sample_divergence.png"), dpi=150)

    if show:
        plt.show()

    return figs


# ---------------------------
# Example
# ---------------------------
# --- Replace the "Example" section at the bottom of the script with this ---


def rosenbrock_U(xy: ArrayLike, a: float = 1.0, b: float = 100.0) -> ArrayLike:
    # xy: (2,)
    x = xy[0]
    y = xy[1]
    return (a - x) ** 2 + b * (y - x**2) ** 2


def gaussian_2d_log_prob(
    q: ArrayLike,
    mu: ArrayLike,
    inv_cov: ArrayLike,
    log_det_cov: ArrayLike,
) -> ArrayLike:
    """Log-density of a 2D Gaussian up to the exact normalizing constant."""
    dq = q - mu
    quad = gnp.einsum("i,ij,j->", dq, inv_cov, dq)
    norm_const = 2.0 * gnp.log(2.0 * gnp.pi) + log_det_cov
    return -0.5 * (quad + norm_const)


def plot_rosenbrock_samples(
    samples: ArrayLike,
    a: float = 1.0,
    b: float = 100.0,
    title: str = "NUTS samples on Rosenbrock",
    max_points: int = 5000,
    pad_frac: float = 0.15,
    grid_n: int = 250,
    contour_levels: int = 30,
    automatically_set_size: bool = False,
):
    """
    samples: (num_samples, chains, 2)
    Produces one figure: contour(U) + scatter(samples).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError("samples must have shape (num_samples, chains, 2)")

    x = gnp.reshape(samples, (-1, 2))
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    x = np.asarray(x)

    n = x.shape[0]
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        x = x[idx]

    if automatically_set_size:
        xmin, ymin = x.min(axis=0)
        xmax, ymax = x.max(axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        xmin -= pad_frac * (dx + 1e-12)
        xmax += pad_frac * (dx + 1e-12)
        ymin -= pad_frac * (dy + 1e-12)
        ymax += pad_frac * (dy + 1e-12)
    else:
        xmin = -1.525
        xmax = 2.3
        ymin = -0.72
        ymax = 3.9

    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    X, Y = np.meshgrid(xs, ys)

    U = (a - X) ** 2 + b * (Y - X**2) ** 2

    fig = plt.figure()
    plt.contour(X, Y, U, levels=contour_levels)
    plt.scatter(x[:, 0], x[:, 1], s=6, alpha=0.5)
    plt.xlim(-1.525, 2.3)
    plt.ylim(-0.72, 3.9)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()
    return fig


if __name__ == "__main__":
    case = "rosenbrock"  # "rosenbrock" or "gaussian2d"

    if case == "rosenbrock":
        a = 1.0
        b = 100.0  # 100.0 is the classic Rosenbrock; 50.0 is often easier to sample
        T = 1.0  # temperature; larger => wider, easier geometry

        def log_prob(q: ArrayLike) -> ArrayLike:
            return -rosenbrock_U(q, a=a, b=b) / T

        debug_points = [[-1.25, 1.50], [-0.50, 0.25], [1.00, 1.00]]
        print(f"[debug value_and_grad] backend={gnp._gpmp_backend_}")
        for i, q_dbg in enumerate(debug_points):
            q_dbg = gnp.asarray(q_dbg)
            val, grad = gnp.value_and_grad(log_prob, q_dbg)
            print(
                f"[debug value_and_grad] i={i} q={gnp.to_np(q_dbg)} "
                f"value={gnp.to_scalar(val)} grad={gnp.to_np(grad)}"
            )

        chains = 4
        q_start = gnp.asarray([-1.5, 1.5])
        q0 = gnp.tile(q_start, (chains, 1))

        samples, info = nuts_sample(
            log_prob=log_prob,
            q_init=q0,
            num_samples=400,
            num_warmup=1000,
            target_accept=0.8,
            max_depth=10,
            seed=42,
            progress=True,
            verbose=2,
            log_every=100,
        )

        print("step_size_final:", gnp.to_scalar(info["step_size_final"]))
        mass_diag_final = info["mass_diag_final"]
        if hasattr(mass_diag_final, "cpu"):
            mass_diag_final = mass_diag_final.cpu()
        print("mass_diag_final:", gnp.asarray(mass_diag_final))

        divergent = info["divergent"]
        if hasattr(divergent, "float"):
            divergent = divergent.float()
        print("divergent rate:", float(gnp.mean(gnp.asarray(divergent, dtype=gnp_dtype))))

        plot_rosenbrock_samples(
            samples=samples,
            a=a,
            b=b,
            title=f"NUTS samples on Rosenbrock (a={a}, b={b}, T={T})",
            max_points=6000,
            contour_levels=35,
        )

    elif case == "gaussian2d":
        mu_np = np.array([0.75, -0.35], dtype=gnp_dtype)
        cov_np = np.array([[0.9, 0.35], [0.35, 1.4]], dtype=gnp_dtype)
        inv_cov_np = np.linalg.inv(cov_np)
        sign, log_det_cov = np.linalg.slogdet(cov_np)
        if sign <= 0:
            raise RuntimeError("cov_np must be SPD.")

        mu = gnp.asarray(mu_np)
        inv_cov = gnp.asarray(inv_cov_np)
        log_det_cov = float(log_det_cov)

        def log_prob(q: ArrayLike) -> ArrayLike:
            return gaussian_2d_log_prob(q, mu, inv_cov, log_det_cov)

        debug_points = [[-1.0, 1.0], [0.0, 0.0], [1.2, -0.8]]
        print(f"[debug value_and_grad] backend={gnp._gpmp_backend_}")
        for i, q_dbg in enumerate(debug_points):
            q_dbg = gnp.asarray(q_dbg)
            val, grad = gnp.value_and_grad(log_prob, q_dbg)
            print(
                f"[debug value_and_grad] i={i} q={gnp.to_np(q_dbg)} "
                f"value={gnp.to_scalar(val)} grad={gnp.to_np(grad)}"
            )

        chains = 4
        q_start = gnp.asarray([-1.5, 1.5])
        q0 = gnp.tile(q_start, (chains, 1))

        samples, info = nuts_sample(
            log_prob=log_prob,
            q_init=q0,
            num_samples=4000,
            num_warmup=1000,
            target_accept=0.8,
            max_depth=10,
            seed=0,
            progress=True,
            verbose=2,
            log_every=200,
        )

        samples_np = gnp.to_np(gnp.reshape(samples, (-1, 2)))
        emp_mean = np.mean(samples_np, axis=0)
        emp_cov = np.cov(samples_np.T, ddof=1)

        print("[gaussian2d] target mean:", mu_np)
        print("[gaussian2d] empirical mean:", emp_mean)
        print("[gaussian2d] mean error:", emp_mean - mu_np)
        print("[gaussian2d] target cov:")
        print(cov_np)
        print("[gaussian2d] empirical cov:")
        print(emp_cov)
        print("[gaussian2d] cov error:")
        print(emp_cov - cov_np)
        print("step_size_final:", gnp.to_scalar(info["step_size_final"]))
        print("mass_diag_final:", gnp.to_np(info["mass_diag_final"]))
        div = gnp.asarray(info["divergent"], dtype=gnp_dtype)
        print("divergent rate:", float(gnp.mean(div)))

    else:
        raise ValueError("Unknown case. Use 'rosenbrock' or 'gaussian2d'.")
