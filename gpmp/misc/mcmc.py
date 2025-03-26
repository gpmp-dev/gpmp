"""
Metropolis–Hastings sampler

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gaussian_kde, ks_2samp
from typing import Callable, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass, field


@dataclass
class MHOptions:
    """
    Configuration for the Metropolis–Hastings sampler.
    """

    dim: int = 1
    n_chains: int = 1
    symmetric: bool = True
    target_acceptance: float = 0.3
    acceptance_tol: float = 0.15
    adaptation_method: str = "Haario"
    proposal_distribution_param_init: Union[np.ndarray, None] = field(default=None)
    adaptation_interval: int = 50
    freeze_adaptation: bool = True
    discard_burnin: bool = False
    n_pool: int = 1
    RM_adapt_factor: float = 1.0
    RM_diminishing: bool = True
    haario_adapt_factor_burnin_phase: float = 1.0
    haario_adapt_factor_sampling_phase: float = 0.5
    haario_initial_scaling_factor: float = 1.0
    sliding_rate_width: int = 200
    show_global_progress: bool = False
    progress_interval: int = 200  # Print every 200 iterations
    init_msg: Union[str, None] = field(default="Sampling from target distribution...")

    def __post_init__(self):
        # If user didn’t supply a proposal_param_init, default to np.ones(dim)
        if self.proposal_distribution_param_init is None:
            self.proposal_distribution_param_init = np.ones(self.dim, dtype=float)
        self.acceptance_min = self.target_acceptance - self.acceptance_tol
        self.acceptance_max = self.target_acceptance + self.acceptance_tol


class MetropolisHastings:
    """
    Metropolis–Hastings sampler with two adaptive strategies:

    - RM (Robbins–Monro) for diagonal or scalar proposals
      ~ adjust proposal_params[i] *= exp(step * (rate - target))

    - Haario for full covariance:
      self.haario_scaling_factors[i] gets updated similarly,
      and each chain's proposal = scaling * EmpCov + eps*I.
      The default 2.38^2/dim factor comes from empirical results
      on multivariate Gaussians suggesting optimal acceptance ~0.23.
    """

    def __init__(
        self,
        log_target: Callable[[np.ndarray], float],
        prop_rnd: Callable[[np.ndarray, int], np.ndarray] = None,
        options: MHOptions = None,
    ):
        """
        log_target: function returning log of target pdf
        prop_rnd: optional function giving random-walk proposal
        options: MHOptions controlling chain count, dimension, etc.
        """
        self.options = options or MHOptions()
        self.log_target = log_target
        self.prop_rnd = prop_rnd or self.default_prop_rnd
        self.rng = np.random.default_rng()

        # Basic config from options
        self.n_chains = self.options.n_chains
        self.dim = self.options.dim
        self.symmetric = self.options.symmetric
        self.target_acceptance = self.options.target_acceptance

        # Proposal params for each chain
        self.proposal_distribution_params = None

        # Full-cov scaling factor for Haario policy
        self.haario_adapt_factor = None
        if self.options.haario_initial_scaling_factor is not None:
            # Use user-supplied initial scale
            self.haario_scaling_factors = [
                self.options.haario_initial_scaling_factor for _ in range(self.n_chains)
            ]
        else:
            # Use default 2.38^2 / dim from literature
            self.haario_scaling_factors = [
                (2.38**2 / self.dim) for _ in range(self.n_chains)
            ]
        # chain history: shape(n_chains, n_steps, dim)
        self.x = None
        self.accept = None
        self.rates = None

        # States & Counters
        self.sampling_mode = "init"
        self.burnin_period = 0
        self.global_iter = 0  # Overall iteration across blocks
        self.global_total = 0  # Total iterations for the entire run
        self.start_time = None  # When we began the entire run

    def _log_prop(self, x: np.ndarray, x_new: np.ndarray, chain_idx: int) -> float:
        """
        Log-proposal density for x_new given x, used if proposal is not symmetric.
        """
        return multivariate_normal.logpdf(
            x_new, mean=x, cov=self._get_cov_parameter(chain_idx)
        )

    def _get_cov_parameter(self, chain_idx: int) -> np.ndarray:
        """
        Build a covariance matrix for chain chain_idx from self.proposal_params.
        """
        p = self.proposal_distribution_params[chain_idx]
        if np.isscalar(p):
            return p * np.eye(self.dim)
        elif p.ndim == 1:
            return np.diag(p)
        elif p.ndim == 2:
            return p
        else:
            raise ValueError("proposal_params must be scalar, 1D, or 2D per chain.")

    def _initialize_proposal_distribution_params(self, p_init: np.ndarray) -> list:
        """
        Convert user-supplied proposal_param_init into a list, one per chain.
        """
        if p_init.ndim == 1 and p_init.shape[0] == self.dim:
            return [p_init.copy() for _ in range(self.n_chains)]
        if p_init.ndim == 2 and p_init.shape == (self.dim, self.dim):
            return [p_init.copy() for _ in range(self.n_chains)]
        if p_init.ndim == 3 and p_init.shape[0] == self.n_chains:
            return [p_init[i].copy() for i in range(self.n_chains)]
        raise ValueError("Invalid proposal_param_init shape.")

    def _get_pooled_samples(self, burnin=0, n_pool=1) -> list[np.ndarray]:
        if self.x is None:
            raise ValueError("No chain data yet.")
        if self.n_chains % n_pool != 0:
            raise ValueError("n_pool must divide n_chains")
        x_pooled = []
        for i in range(0, self.n_chains, n_pool):
            chunk = self.x[i : i + n_pool, burnin:].reshape(-1, self.dim)
            x_pooled.append(chunk)
        return x_pooled

    def _compute_covariances_for_block(
        self, x_block: np.ndarray, n_pool: int = 1
    ) -> np.ndarray:
        """
        Compute covariance per group of chains (each group has n_pool chains),
        given an array x_block of shape (n_chains, block_size, dim).

        Returns:
          covs: shape (n_groups, dim, dim). Each slice covs[i] is the covariance
                for the i-th group of n_pool chains.
        """
        n_chains, block_size, _ = x_block.shape
        if n_chains % n_pool != 0:
            raise ValueError("n_chains must be divisible by n_pool.")
        n_groups = n_chains // n_pool

        covs = np.empty((n_groups, self.dim, self.dim), dtype=float)
        start_indices = range(0, n_chains, n_pool)

        for i, start in enumerate(start_indices):
            x_group = x_block[start : start + n_pool].reshape(-1, self.dim)
            covs[i] = np.cov(x_group.T, ddof=1)

        return covs

    def _diminishing_adaptation_schedule(
        self,
        n: int,
        n_total: int,
        base: float,
        final_frac: float = 0.1,
    ) -> float:
        """Compute an adaptation factor for the current step based on
        a cosine schedule.  At step 0, returns base; at step n,
        returns base * final_frac.

        """
        cosine_component = math.cos(math.pi * n / n_total)
        return base * (final_frac + (1 - final_frac) * cosine_component)

    def _validate_scheduler_args(self, n_steps_total: int, burnin: int):
        if n_steps_total < burnin:
            raise ValueError("Total steps < burnin")

    def _print_progress(
        self, iteration: int, total_steps: int, start_time: float
    ) -> None:
        """
        Print progress info on a line, including % complete and
        estimated remaining time.
        """
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / (iteration + 1)
        remaining = avg_time * (total_steps - (iteration + 1))
        pct = (iteration + 1) / total_steps * 100
        msg = f"  Progress: {pct:5.2f}% | Time left: {remaining:5.1f}s"
        print(f"{msg:70s}", end="\r")

    def _print_final_time(self, total_steps: int, start_time: float) -> None:
        """
        Print final summary of total elapsed time and number of steps
        """
        elapsed_time = time.time() - start_time
        print(f"  Progress: 100.00% complete | Total time: {elapsed_time:.3f}s")
        print(f"  Total proposals: {total_steps * self.n_chains}")

    def set_mode(self, mode: str):
        self.sampling_mode = mode
        if mode == "burnin":
            self.haario_adapt_factor = self.options.haario_adapt_factor_burnin_phase
        elif mode == "sampling_adaptation":
            self.haario_adapt_factor = self.options.haario_adapt_factor_sampling_phase

    def default_prop_rnd(self, x: np.ndarray, chain_idx: int) -> np.ndarray:
        """
        Default random-walk: draw from N(x, Cov) where Cov depends on proposal_params.
        """
        cov = self._get_cov_parameter(chain_idx)
        perturbation = self.rng.multivariate_normal(np.zeros(self.dim), cov)
        return x + perturbation

    def update_proposal_covariance_from_samples(
        self,
        x_chain: np.ndarray = None,
        raw_cov: np.ndarray = None,
        scaling: float = None,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """
        Build a proposal covariance matrix according to Haario’s formula:
          new_cov = scaling * raw_cov + epsilon * I
        Either supply x_chain (from which raw_cov is computed) or raw_cov directly.

        Parameters:
          x_chain : (n_samples, dim) array, optional
            The chain’s samples from which covariance is computed.
          raw_cov : (dim, dim) array, optional
            A precomputed covariance matrix. Supply this instead of x_chain.
          scaling : float, optional
            Multiplicative scale. Defaults to 2.38^2/dim if not set.
          epsilon : float, optional
            Small diagonal shift for stability. Default=1e-6.

        Returns:
          cov : (dim, dim) The updated proposal covariance.
        """
        if (x_chain is None) == (raw_cov is None):
            raise ValueError("Must supply exactly one of x_chain or raw_cov.")
        if scaling is None:
            scaling = (2.38**2) / self.dim

        if raw_cov is not None:
            used_cov = raw_cov
        else:
            used_cov = np.cov(x_chain.T, ddof=1)

        return scaling * used_cov + epsilon * np.eye(self.dim)

    def compute_sliding_rates(self, n_block_size: int) -> np.ndarray:
        """
        Compute the sliding mean of the acceptance rate for each chain
        using a   averages of size n_block_size over iterations 0 to self.global_iter.
        For early iterations (t < n_block_size), the mean is computed over t+1 samples.

        Parameters
        ----------
        n_block_size : int
            Size of the sliding   averages.

        Returns
        -------
        np.ndarray
            Array of shape (self.n_chains, self.global_iter) with sliding mean acceptance rates.
        """
        if self.accept is None:
            raise ValueError("No acceptance data available to compute sliding rates.")

        n_chains = self.n_chains
        n_max = self.global_iter
        rates = np.empty((n_chains, n_max))

        for c in range(n_chains):
            acc = self.accept[c, :n_max].astype(float)
            cumsum = np.cumsum(acc)
            # For the first n_block_size samples, average over available samples.
            rates[c, :n_block_size] = cumsum[:n_block_size] / (
                np.arange(n_block_size) + 1
            )
            # For samples t >= n_block_size, average over the last n_block_size samples.
            rates[c, n_block_size:] = (
                cumsum[n_block_size:] - cumsum[:-n_block_size]
            ) / n_block_size

        return rates

    def mhstep(self, x_current: np.ndarray, chain_idx: int) -> Tuple[np.ndarray, bool]:
        """
        Single Metropolis–Hastings update for chain chain_idx.
        If symmetric=False, includes reverse-proposal terms in acceptance.
        """
        y = self.prop_rnd(x_current, chain_idx)
        log_a = self.log_target(y) - self.log_target(x_current)
        if not self.symmetric:
            log_a += self._log_prop(y, x_current, chain_idx) - self._log_prop(
                x_current, y, chain_idx
            )
        accept = np.log(self.rng.random()) < log_a
        return (y, True) if accept else (x_current, False)

    def run_samples(
        self,
        n_steps: int,
        show_global_progress: bool = False,
    ) -> np.ndarray:
        """
        Run n_steps of MCMC sampling and return acceptation rate.
        """
        i0 = self.global_iter + 1
        i1 = self.global_iter + 1 + n_steps
        for t in range(i0, i1):
            for c in range(self.n_chains):
                self.x[c, t], self.accept[c, t] = self.mhstep(self.x[c, t - 1], c)
            self.global_iter += 1
            if show_global_progress:
                if self.global_iter % self.options.progress_interval == 0:
                    self._print_progress(
                        self.global_iter, self.global_total, self.start_time
                    )
        block_rates = np.mean(self.accept[:, i0:i1], axis=1)
        return block_rates

    def run_adaptive_RM(self, n_block_size: int, diminishing: bool = True):
        """
        Run one block of Robbins–Monro adaptation.

        Parameters:
          n_block_size : int
              Number of steps in this block.
          diminishing : bool, optional
              Whether to use a diminishing adaptation schedule (default: True).
        """
        gamma_base = self.options.RM_adapt_factor
        rates = self.run_samples(
            n_block_size,
            show_global_progress=self.options.show_global_progress,
        )
        if diminishing:
            gamma = self._diminishing_adaptation_schedule(
                self.global_iter, self.burnin_period, gamma_base, final_frac=0.1
            )
        else:
            gamma = gamma_base
        for c in range(self.n_chains):
            self.proposal_distribution_params[c] *= np.exp(
                gamma * (rates[c] - self.target_acceptance)
            )

    def run_adaptive_Haario(self, n_block_size: int, epsilon: float = 1e-6):
        """
        Run one block of Haario adaptation.

        Parameters:
          n_block_size : int
              Number of steps in this block.
          init : np.ndarray
              The initial state for the block.
          epsilon : float, optional
              Small diagonal shift for stability (default: 1e-6).
        """
        block_rates = self.run_samples(
            n_block_size,
            show_global_progress=self.options.show_global_progress,
        )
        # Compute pooled covariances for groups of chains.
        i0 = self.global_iter - n_block_size + 1
        i1 = self.global_iter + 1
        covs = self._compute_covariances_for_block(
            self.x[:, i0:i1, :], self.options.n_pool
        )
        for c in range(self.n_chains):
            grp = c // self.options.n_pool
            self.haario_scaling_factors[c] *= np.exp(
                self.haario_adapt_factor * (block_rates[c] - self.target_acceptance)
            )
            self.proposal_distribution_params[c] = (
                self.update_proposal_covariance_from_samples(
                    raw_cov=covs[grp],
                    scaling=self.haario_scaling_factors[c],
                    epsilon=epsilon,
                )
            )

    def run_adaptive(self, n_samples: int):
        """
        Runs the chain for n_samples samples using phase block by block adpation.

        Parameters
        ----------
        n_samples : int
            Total number of steps.
        """
        n_blocks = n_samples // self.options.adaptation_interval
        method = self.options.adaptation_method.lower()
        for _ in range(n_blocks):
            if method == "rm":
                self.run_adaptive_RM(
                    self.options.adaptation_interval,
                    diminishing=None,
                )
            elif method == "haario":
                self.run_adaptive_Haario(self.options.adaptation_interval)
            else:
                raise ValueError("adaptation_method must be 'RM' or 'Haario'.")

    def run_burnin(
        self,
        burnin_period: int,
        diag: bool = True,
        n_blocks_convergence_diag: int = 20,
    ) -> None:
        """
        Run the burn-in phase block by block with optional early stopping based on convergence diagnostics.

        Parameters
        ----------
        burnin_period : int
            Total number of burn-in steps.
        diag : bool, optional
            If True, run diagnostics (sliding acceptance rates and Gelman–Rubin) after each block (default: True).
        n_blocks_convergence_diag : int, optional
            Number of burn-in blocks to use for diagnostics; each block is self.options.adaptation_interval steps (default: 4).
        """
        n_blocks = burnin_period // self.options.adaptation_interval
        method = self.options.adaptation_method.lower()
        # Diagnostic sample count (used for early stopping)
        n_diag_samples = n_blocks_convergence_diag * self.options.adaptation_interval

        for block in range(n_blocks):
            if method == "rm":
                self.run_adaptive_RM(
                    self.options.adaptation_interval,
                    diminishing=self.options.RM_diminishing,
                )
            elif method == "haario":
                self.run_adaptive_Haario(self.options.adaptation_interval)
            else:
                raise ValueError("adaptation_method must be 'RM' or 'Haario'.")
            # Early stopping diagnostics: check if we have enough samples and convergence is achieved
            if diag and self.global_iter >= n_diag_samples:
                rates = self.compute_sliding_rates(self.options.sliding_rate_width)
                i0 = max(0, self.global_iter - n_diag_samples)
                i1 = self.global_iter
                rates = rates[:, i0:i1]
                min_ar = np.min(rates, axis=1)
                max_ar = np.max(rates, axis=1)
                gr_results = self.check_convergence_gelman_rubin(
                    last_n_samples=n_diag_samples, verbose=False
                )
                if (
                    np.all(min_ar > self.options.acceptance_min)
                    and np.all(max_ar < self.options.acceptance_max)
                ) and gr_results.get("ok", False):
                    print(
                        f"\nEarly stopping: convergence detected during burn-in at iter = {self.global_iter}."
                    )
                    print(
                        f"  Min / Max acceptance rate: {np.min(min_ar):.3f} / {np.max(max_ar):.3f}"
                    )
                    print(f"  Gelman-Rubin: {gr_results}")
                    self.burnin_period = self.global_iter
                    break

        if diag:
            print("\nConvergence Diagnostics after burn-in:")
            rates = self.compute_sliding_rates(self.options.sliding_rate_width)
            self.check_acceptance_rates(
                last_n_samples=n_diag_samples,
                rates=rates,
                low_threshold=self.options.acceptance_min,
                high_threshold=self.options.acceptance_max,
            )
            self.check_convergence_gelman_rubin(last_n_samples=n_diag_samples)

    def scheduler(
        self,
        chains_state_initial: np.ndarray,
        n_steps_total: int,
        burnin_period: int,
        replicate_initial_state: bool = True,
    ) -> np.ndarray:
        """
        Run the full MCMC process:
          - Runs the burn-in phase.
          - Runs the sampling phase.
        """
        chains_state_initial = np.atleast_2d(chains_state_initial)
        if (
            chains_state_initial.shape == (1, self.dim)
            and replicate_initial_state
            and self.n_chains > 1
        ):
            chains_state_initial = np.tile(chains_state_initial, (self.n_chains, 1))
        if chains_state_initial.shape != (self.n_chains, self.dim):
            raise ValueError(
                f"chains_state_initial must have shape ({self.n_chains}, {self.dim})"
                + f" or be 1D if replicate_initial_state=True. Got {chains_state_initial.shape}."
            )
        self._validate_scheduler_args(n_steps_total, burnin_period)
        self.proposal_distribution_params = (
            self._initialize_proposal_distribution_params(
                self.options.proposal_distribution_param_init
            )
        )
        # Set up iteration tracking.
        self.x = np.empty((self.n_chains, 1 + n_steps_total, self.dim), dtype=float)
        self.accept = np.empty((self.n_chains, 1 + n_steps_total), dtype=bool)
        self.burnin_period = burnin_period
        self.global_iter = 0
        self.global_total = 1 + n_steps_total
        self.start_time = time.time()
        self.x[:, 0, :] = chains_state_initial
        self.accept[:, 0] = True

        print(self.options.init_msg)
        print(f"  Dimension: {self.dim}")
        print(f"  Total steps: {n_steps_total}")
        print(f"  Burn-in: {burnin_period}")
        print(f"  Chains: {self.n_chains}")

        # Run burn-in.
        self.set_mode("burnin")
        self.run_burnin(burnin_period)

        # Run sampling phase.
        n_remain = n_steps_total - burnin_period
        if self.options.freeze_adaptation:
            self.set_mode("sampling_freeze_adaptation")
            self.run_samples(
                n_remain,
                show_global_progress=self.options.show_global_progress,
            )
        else:
            self.set_mode("sampling_adaptation")
            self.run_adaptive(n_remain)

        self.global_total = self.global_iter
        if self.options.show_global_progress:
            self._print_final_time(self.global_total, self.start_time)

        # Compute acceptance rates
        self.rates = self.compute_sliding_rates(self.options.sliding_rate_width)

        return (
            self.x[:, self.burnin_period : self.global_total]
            if self.options.discard_burnin
            else self.x[:, : self.global_total]
        )

    def check_acceptance_rates(
        self,
        burnin_period: Optional[int] = None,
        last_n_samples: Optional[int] = None,
        low_threshold: float = 0.15,
        high_threshold: float = 0.40,
        rates: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, Union[float, bool]]:
        """
        Check acceptance rates using a provided 'rates' array or self.rates.

        Uses samples from burnin_period to self.global_iter (or only the last n samples if specified).
        Prints warnings if the min rate is below low_threshold or the max rate is above high_threshold.

        Returns
        -------
        Dict[str, Union[float, bool]]
            Dictionary with keys:
              - "min_ar": minimum acceptance rate
              - "max_ar": maximum acceptance rate
              - "ok": True if min_ar >= low_threshold and max_ar <= high_threshold, else False.
        """
        if burnin_period is None:
            burnin_period = self.burnin_period
        if rates is None:
            if self.rates is None:
                if verbose:
                    print(
                        "No sliding acceptance rates available. Please compute self.rates first."
                    )
                return {}
            rates_data = self.rates
        else:
            rates_data = rates

        i0 = (
            burnin_period
            if last_n_samples is None
            else max(0, self.global_iter - last_n_samples)
        )
        i1 = self.global_iter
        if (n_block_size := i1 - i0) <= 1:
            raise ValueError("Not enough samples to compute acceptance rates.")

        data = rates_data[:, i0:i1]
        min_ar = data.min()
        max_ar = data.max()
        ok = (min_ar >= low_threshold) and (max_ar <= high_threshold)

        if verbose:
            print("[check_acceptance_rates]")
            if not ok:
                if min_ar < low_threshold:
                    print(
                        f"WARNING: Min acceptance rate ({min_ar:.3f}) is below the threshold of {low_threshold:.2f}."
                    )
                if max_ar > high_threshold:
                    print(
                        f"WARNING: Max acceptance rate ({max_ar:.3f}) is above the threshold of {high_threshold:.2f}."
                    )
            else:
                print("PASS: Acceptance rates within tolerance bounds")
            print(f"  Min = {min_ar:.3f},  Max = {max_ar:.3f}")

        return {"min_ar": min_ar, "max_ar": max_ar, "ok": ok}

    def compute_gelman_rubin_rhat(
        self,
        burnin_period: Optional[int] = None,
        last_n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the Gelman-Rubin R-hat statistic using a slice of self.x.

        Parameters
        ----------
        burnin_period : int, optional
            Number of initial iterations to ignore.
        last_n_samples : Optional[int], optional
            Use only the last n samples (from self.global_iter); if None, use samples from burnin_period to self.global_iter.

        Returns
        -------
        np.ndarray
            R-hat values for each parameter (shape: (dim,)).
        """
        if burnin_period is None:
            burnin_period = self.burnin_period
        if self.x is None:
            raise ValueError("No chain data available.")
        n_chains, n_steps, dim = self.x.shape
        if n_chains < 2:
            raise ValueError("At least 2 chains are required.")

        i0 = (
            burnin_period
            if last_n_samples is None
            else max(0, self.global_iter - last_n_samples)
        )
        i1 = self.global_iter
        n_block = i1 - i0
        if n_block <= 1:
            raise ValueError("Not enough samples to compute Gelman-Rubin diagnostic.")

        block = self.x[:, i0:i1, :]  # shape: (n_chains, n_block, dim)
        chain_means = np.mean(block, axis=1)  # shape: (n_chains, dim)
        chain_vars = np.var(block, axis=1, ddof=1)  # shape: (n_chains, dim)
        W = np.mean(chain_vars, axis=0)  # within-chain variance
        # between-chain variance
        B = n_block * np.var(chain_means, axis=0, ddof=1)
        var_post = ((n_block - 1) / n_block) * W + (1.0 / n_block) * B
        rhat = np.sqrt(var_post / W)
        return rhat

    def check_convergence_gelman_rubin(
        self,
        burnin_period: int = 0,
        last_n_samples: Optional[int] = None,
        threshold: float = 1.1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Check convergence via Gelman-Rubin diagnostic by calling compute_gelman_rubin_rhat.

        Parameters
        ----------
        burnin_period : int, optional
            Number of initial iterations to ignore.
        last_n_samples : Optional[int], optional
            Use only the last n samples; if None, use samples from burnin_period to self.global_iter.
        threshold : float, optional
            Convergence threshold for R-hat (default: 1.1).
        verbose : bool, optional
            If True, print the diagnostic result.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys 'rhat' (np.ndarray) and 'ok' (bool).
        """
        rhat = self.compute_gelman_rubin_rhat(
            burnin_period=burnin_period, last_n_samples=last_n_samples
        )
        ok = np.all(rhat < threshold)
        if verbose:
            if ok:
                print(f"[check_gelman_rubin_rhat]\nPASS: All R-hat < {threshold}.")
            else:
                print(f"[check_gelman_rubin_rhat]\nWARNING: Some R-hat ≥ {threshold}.")
            print(f"  R-hat values: {rhat}")
        return {"rhat": rhat, "ok": ok}

    def ks_statistics(
        self,
        n_blocks: int,
        n_block_size: int,
        alpha: float = 0.01,
        return_significance: bool = True,
        return_statistic: bool = False,
    ):
        """
        Compare the empirical distributions in the last n_blocks blocks
        of size n_block_size on each chain using two-sample KS tests.

        Parameters
        ----------
        n_blocks : int
            Number of blocks to consider (per chain), counting from the end.
        n_block_size : int
            Number of samples in each block.
        alpha : float, optional
            Significance level (default=0.01). If return_significance=True,
            we also produce a boolean mask where p < alpha.
        return_significance : bool, optional
            If True (default), return a boolean mask indicating which comparisons
            are significant at level alpha.
        return_statistic : bool, optional
            If True, also return the 2D array of KS statistic values (dim, B, B).

        Returns
        -------
        Depending on the boolean parameters:
          - If return_statistic=False, return_significance=False:
                pvalue_matrix
          - If return_statistic=False, return_significance=True:
                pvalue_matrix, significance
          - If return_statistic=True,  return_significance=False:
                ks_matrix, pvalue_matrix
          - If return_statistic=True,  return_significance=True:
                ks_matrix, pvalue_matrix, significance

        Where
          ks_matrix      has shape (dim, B, B)        with KS statistics
          pvalue_matrix  has shape (dim, B, B)        with corresponding p-values
          significance   has shape (dim, B, B) bool   where p < alpha
          B = n_blocks * n_chains

        Notes
        -----
        - This uses a two-sided KS test for each dimension's pairwise block comparison.
        - A significant fraction of p-values < alpha may indicate non-convergence.
        """
        if self.x is None:
            raise ValueError("No chain data available. Run sampler first.")

        n_chains, n_steps, dim = self.x.shape

        needed = n_blocks * n_block_size
        if needed > n_steps:
            raise ValueError(
                f"Requested {n_blocks} blocks of size {n_block_size} ({needed} total) "
                f"but chain only has {n_steps} samples."
            )

        # Collect the last n_blocks blocks from each chain.
        blocks = []
        start_index = n_steps - needed
        for chain_idx in range(n_chains):
            for b in range(n_blocks):
                block_start = start_index + b * n_block_size
                block_end = block_start + n_block_size
                blocks.append(self.x[chain_idx, block_start:block_end, :])

        B = len(blocks)  # total number of blocks = n_blocks * n_chains

        # p-values in a (dim, B, B) array
        pvalue_matrix = np.zeros((dim, B, B), dtype=float)

        # Optionally store KS statistics in a separate matrix
        ks_matrix = np.zeros((dim, B, B), dtype=float) if return_statistic else None

        # Fill pairwise results dimension-by-dimension
        for d in range(dim):
            for i in range(B):
                # the diagonal remains zero
                for j in range(i + 1, B):
                    result = ks_2samp(
                        blocks[i][:, d], blocks[j][:, d], alternative="two-sided"
                    )
                    stat, pval = result.statistic, result.pvalue
                    if return_statistic:
                        ks_matrix[d, i, j] = stat
                        ks_matrix[d, j, i] = stat
                    pvalue_matrix[d, i, j] = pval
                    pvalue_matrix[d, j, i] = pval

        if return_significance:
            significance = pvalue_matrix < alpha
            if return_statistic:
                return ks_matrix, pvalue_matrix, significance
            else:
                return pvalue_matrix, significance
        else:
            if return_statistic:
                return ks_matrix, pvalue_matrix
            else:
                return pvalue_matrix

    def check_convergence_ks(
        self,
        multi_block_n_blocks: int = 5,
        multi_block_size: int = 100,
        single_block_size: int = None,
        alpha: float = 0.01,
        fraction_threshold: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """
        Perform two KS-diagnosis checks to assess convergence:
          1) Multi-block: multiple blocks per chain.
          2) Single-block: one block per chain (lumping samples).

        The KS test compares the empirical distributions of blocks.
        A low fraction of significant tests (p < alpha) indicates convergence.

        Parameters
        ----------
        multi_block_n_blocks : int, optional
            Number of sub-blocks to use per chain (default: 5).
        multi_block_size : int, optional
            Size of each sub-block (default: 100).
        single_block_size : int, optional
            Size of the single block. If None, defaults to multi_block_n_blocks * multi_block_size.
        alpha : float, optional
            Significance level for the KS test (default: 0.05).
        fraction_threshold : float, optional
            Convergence is assumed if the fraction of significant comparisons is below this threshold (default: 0.5).
        verbose : bool, optional
            If True, prints a summary.

        Returns
        -------
        results : dict
            {
                "multi_block": {
                    "n_blocks": int,
                    "block_size": int,
                    "frac_significant": float
                },
                "single_block": {
                    "n_blocks": 1,
                    "block_size": int,
                    "frac_significant": float
                },
                "ok": bool
            }
        """
        if self.x is None:
            raise ValueError("No chain data. Please run or load the sampler first.")

        n_chains, n_steps, dim = self.x.shape

        # Multi-block check.
        needed_multi = multi_block_n_blocks * multi_block_size
        if n_steps < needed_multi:
            raise ValueError(
                f"Need at least {needed_multi} samples for multi-block check."
            )
        ks_matA, pval_matA, sigA = self.ks_statistics(
            n_blocks=multi_block_n_blocks,
            n_block_size=multi_block_size,
            alpha=alpha,
            return_significance=True,
            return_statistic=True,
        )
        frac_sig_multi = np.mean(sigA)

        # Single-block check.
        if single_block_size is None:
            single_block_size = needed_multi  # default lumps the same samples
        if n_steps < single_block_size:
            raise ValueError(
                f"Need at least {single_block_size} samples for single-block check."
            )
        ks_matB, pval_matB, sigB = self.ks_statistics(
            n_blocks=1,
            n_block_size=single_block_size,
            alpha=alpha,
            return_significance=True,
            return_statistic=True,
        )
        frac_sig_single = np.mean(sigB)

        # Convergence is declared if the fraction of significant comparisons is low.
        ok = (frac_sig_multi < fraction_threshold) and (
            frac_sig_single < fraction_threshold
        )

        results = {
            "multi_block": {
                "n_blocks": multi_block_n_blocks,
                "block_size": multi_block_size,
                "frac_significant": frac_sig_multi,
            },
            "single_block": {
                "n_blocks": 1,
                "block_size": single_block_size,
                "frac_significant": frac_sig_single,
            },
            "ok": ok,
        }

        if verbose:
            print("[check_convergence_ks]")
            if ok:
                print("PASS: Both KS checks below threshold.")
            else:
                print("WARNING: At least one KS check exceeded threshold.")
            print(
                f"  Multi-block: frac_significant = {frac_sig_multi:.2%} (blocks = {multi_block_n_blocks} x {multi_block_size})"
            )
            print(
                f"  Single-block: frac_significant = {frac_sig_single:.2%} (1 x {single_block_size})"
            )
            print(f"  Threshold = {fraction_threshold:.2%}, alpha = {alpha}")

        return results

    def plot_chains(self, burnin=None, parameter_indices=None, show_rate=True):
        """Trace plots of each dimension, optional acceptance-rate subplot."""
        if burnin is None:
            burnin = self.burnin_period
        if self.x is None:
            raise ValueError("No chain data.")
        n_chains, n_steps, _ = self.x.shape
        pidx = parameter_indices or list(range(self.dim))
        n_plots = len(pidx)
        total_plots = n_plots + 1 if show_rate else n_plots
        fig, axes = plt.subplots(
            total_plots, 1, figsize=(10, 3 * total_plots), sharex=True
        )
        if total_plots == 1:
            axes = [axes]
        # Param traces
        for i, param in enumerate(pidx):
            for c in range(n_chains):
                axes[i].plot(self.x[c, : self.global_iter, param], label=f"Chain {c+1}")
            axes[i].set_ylabel(f"x_{param}")
            if burnin > 0:
                axes[i].axvline(
                    burnin,
                    color="red",
                    linestyle="--",
                    label="End Burn-in" if i == 0 else None,
                )
            axes[i].legend(loc="best")
        if show_rate:
            axr = axes[-1]
            if self.rates is not None:
                for c in range(n_chains):
                    axr.plot(self.rates[c, : self.global_iter], label=f"Chain {c+1}")
                    if burnin > 0:
                        axr.axvline(
                            burnin,
                            color="red",
                            linestyle="--",
                            label="End Burn-in" if i == 0 else None,
                        )
            else:
                print("No acceptance data to display.")
            axr.set_ylabel("Acceptance")
            axr.legend(loc="best")
            axr.set_xlabel("Iteration")
        else:
            axes[-1].set_xlabel("Iteration")
        plt.tight_layout()
        plt.show()

    def plot_empirical_distributions(
        self, burnin=None, parameter_indices=None, bins=50, smooth=True
    ):
        """Plot either histogram or KDE for each parameter dimension, chain by chain."""
        if burnin is None:
            burnin = self.burnin_period
        if self.x is None:
            raise ValueError("No chain data.")
        n_chains, n_steps, _ = self.x.shape
        pidx = parameter_indices or list(range(self.dim))
        n_plots = len(pidx)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=False)
        if n_plots == 1:
            axes = [axes]
        for i, param in enumerate(pidx):
            ax = axes[i]
            for c in range(n_chains):
                vals = self.x[c, burnin : self.global_iter, param]
                if not smooth:
                    ax.hist(
                        vals,
                        bins=bins,
                        density=True,
                        histtype="step",
                        label=f"Chain {c+1}",
                    )
                else:
                    lo, hi = vals.min(), vals.max()
                    xx = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 100)
                    kde = gaussian_kde(vals)
                    ax.plot(xx, kde(xx), label=f"Chain {c+1}")
            ax.set_xlabel(rf"$\theta_{{{param+1}}}$")
            ax.set_ylabel("Density")
            ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def recompute_all_chains_full_covariance(
        self, burnin=None, scaling=None, epsilon=1e-6
    ):
        """Recompute each chain's param from post-burnin samples (Haario approach)."""
        if burnin is None:
            burnin = self.burnin_period
        if self.x is None:
            raise ValueError("No chain data available.")
        for c in range(self.n_chains):
            x = self.x[c, burnin:]
            self.proposal_distribution_params[c] = (
                self.update_proposal_covariance_from_samples(x, scaling, epsilon)
            )

    def compute_empirical_covariance_whole_chain(
        self, burnin=None, pooled=False, n_pool=1
    ) -> Union[np.ndarray, list]:
        """
        If pooled=True, single covariance from all chains post-burnin.
        Otherwise, group in n_pool lumps. Return list of cov per group.
        """
        if burnin is None:
            burnin = self.burnin_period
        if self.x is None:
            raise ValueError("No samples yet.")
        if pooled:
            big = self.x[:, burnin:].reshape(-1, self.dim)
            return np.cov(big.T, ddof=1)
        x_pooled = self._get_pooled_samples(burnin, n_pool)
        return [np.cov(x.T, ddof=1) for x in x_pooled]


def test_linear_regression():
    """Linear regression example"""
    np.random.seed(42)

    # Example: linear regression with optional dummy dims
    n = 60
    d_0 = 5
    X = np.random.randn(n, d_0)
    true_beta = np.array([1.0, -2.0, 0.5, 0.0, 1.5])
    sigma = 1.0
    y = X @ true_beta + sigma * np.random.randn(n)

    def log_target(beta: np.ndarray) -> float:
        # Suppose only first d are real coefs, rest dummy
        ll = -0.5 * np.sum((y - X @ beta[:d_0]) ** 2) / (sigma**2)
        # mild Gaussian prior
        lp = -0.5 * np.sum(beta**2) / 100.0
        return ll + lp

    # Extend dimension
    d_inactive = 2
    d_1 = d_0 + d_inactive

    # Configure
    opts = MHOptions(
        dim=d_1,
        n_chains=2,
        target_acceptance=0.25,
        adaptation_method="Haario",
        adaptation_interval=50,
        freeze_adaptation=True,
        discard_burnin=True,
        n_pool=2,
        haario_adapt_factor_burnin_phase=1.0,
        haario_adapt_factor_sampling_phase=0.5,
        haario_initial_scaling_factor=1.0,
        show_global_progress=True,
    )

    # Sampler instance
    mh = MetropolisHastings(log_target=log_target, options=opts)

    # Initial states
    init_states = np.zeros((opts.n_chains, opts.dim))

    n_steps_total = 10000
    burnin_period = 3000

    # Adaptation + sampling
    x = mh.scheduler(
        chains_state_initial=init_states,
        n_steps_total=n_steps_total,
        burnin_period=burnin_period,
    )

    # Plot
    mh.plot_chains()
    mh.plot_empirical_distributions(smooth=False)

    # Posterior means
    post_mean = np.mean(x, axis=(0, 1))
    print("Posterior means:", post_mean)
    print("True beta:", true_beta)


def test_chi2():
    """
    2D Chi-square(ν) distribution
    """
    import math
    from scipy.stats import chi2

    # -------------------------------
    # 1) Configuration of distribution
    # -------------------------------
    dof = 3
    scale = [1e-3, 1.0]

    def pdf_1d(x, df=dof, scale=1.0):
        return chi2.pdf(x, df=df, scale=scale)

    def logpdf_1d(x, df=dof, scale=1.0):
        return chi2.logpdf(x, df=df, scale=scale)

    def log_chi2_2d(z: np.ndarray) -> float:
        return logpdf_1d(z[0], df=dof, scale=scale[0]) + logpdf_1d(
            z[1], df=dof, scale=scale[1]
        )

    def pdf_2d(x, y):
        return pdf_1d(x, df=dof, scale=scale[0]) * pdf_1d(y, df=dof, scale=scale[1])

    # -------------------------------
    # 2) MCMC Setup
    # -------------------------------
    np.random.seed(42)
    dim = 2

    # Metropolis–Hastings options
    opts = MHOptions(
        dim=dim,
        n_chains=2,
        target_acceptance=0.3,
        adaptation_method="Haario",
        adaptation_interval=50,
        freeze_adaptation=False,
        discard_burnin=False,
        n_pool=2,
        haario_adapt_factor_burnin_phase=1.0,
        haario_adapt_factor_sampling_phase=0.5,
        haario_initial_scaling_factor=1.0,
        show_global_progress=True,
    )

    # Sampler using our 2D log-target
    mh = MetropolisHastings(log_target=log_chi2_2d, options=opts)

    # Start chains at a positive point, e.g. (3,3)
    init_states = np.full((opts.n_chains, opts.dim), 1.0)

    # -------------------------------
    # 3) Run the sampler
    # -------------------------------
    n_steps_total = 10_000
    burnin_period = 5_000

    x = mh.scheduler(
        chains_state_initial=init_states,
        n_steps_total=n_steps_total,
        burnin_period=burnin_period,
    )
    # x has shape (n_chains, n_steps_total, 2) or (n_chains, n_steps_total - burnin, 2)
    #   depending on discard_burnin (currently False => we keep burnin).

    # ------------------------------
    # 4) Convergence diagnosis
    # ------------------------------
    mh.check_acceptance_rates()
    res = mh.check_convergence_gelman_rubin(last_n_samples=1000, threshold=1.1)

    # -------------------------------
    # 5) Plot chain traces & hist
    # -------------------------------
    mh.plot_chains()

    # Merge chains (post-burnin) for dimension-by-dimension hist
    x = x[:, mh.burnin_period :, :].reshape(-1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(dim):
        ax = axes[i]
        x_1d = x[:, i]

        # Histogram of MCMC samples
        ax.hist(x_1d, bins=50, density=True, alpha=0.5, label="MCMC samples")

        # Overlay the true marginal pdf
        xs = np.linspace(0.01, 15.0, 300) * scale[i]
        # dimension i's scale
        scale_i = scale[0] if i == 0 else scale[1]
        pdf_vals = pdf_1d(xs, df=dof, scale=scale_i)

        ax.plot(xs, pdf_vals, "r-", label="True marginal pdf")
        ax.set_xlabel(f"Dimension {i}")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle(f"2D chi2 (nu={dof}) – Marginal Distributions", fontsize=14)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 6) 2D contour of joint pdf + scatter
    # -------------------------------
    grid_pts = 60
    max_val = 15.0
    xs = np.linspace(0.01, max_val, grid_pts) * scale[0]
    ys = np.linspace(0.01, max_val, grid_pts) * scale[1]
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    ZZ = np.zeros_like(XX)

    for r in range(grid_pts):
        for c in range(grid_pts):
            ZZ[r, c] = pdf_2d(XX[r, c], YY[r, c])

    plt.figure(figsize=(6, 5))
    plt.contour(XX, YY, ZZ, levels=12)
    plt.scatter(x[:, 0], x[:, 1], alpha=0.2, label="MCMC samples")
    plt.title(
        f"2D chi2 (nu={dof}): PDF Contour + Samples\n(scale0={scale[0]}, scale1={scale[1]})"
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # case = "linear_regression"
    case = "chi2"

    if case == "linear_regression":
        test_linear_regression()
    elif case == "chi2":
        test_chi2()
