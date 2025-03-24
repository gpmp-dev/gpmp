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
from typing import Callable, Tuple, Union
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
    adaptation_method: str = "Haario"
    proposal_param_init: Union[np.ndarray, None] = field(default=None)
    adaptation_interval: int = 50
    freeze_adaptation: bool = True
    discard_burnin: bool = False
    n_pool: int = 1
    RM_adapt_factor: float = 1.0
    haario_adapt_factor: float = 1.0
    haario_initial_scaling_factor: float = 1.0
    show_global_progress: bool = False
    progress_interval: int = 200  # Print every 200 iterations
    init_msg: Union[str, None] = field(default="Sampling from target distribution...")

    def __post_init__(self):
        # If user didn’t supply a proposal_param_init, default to np.ones(dim)
        if self.proposal_param_init is None:
            self.proposal_param_init = np.ones(self.dim, dtype=float)


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
        self.proposal_params = None

        # Full-cov scaling factor for Haario policy
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
        self.rates = []  # acceptance rates per block

        # Counters
        self._global_iter = 0  # Overall iteration across blocks
        self._global_total = 0  # Total iterations for the entire run
        self._start_time = None  # When we began the entire run

    def _log_prop(self, x: np.ndarray, x_new: np.ndarray, chain_idx: int) -> float:
        """
        Log-proposal density for x_new given x, used if proposal is not symmetric.
        """
        return multivariate_normal.logpdf(
            x_new, mean=x, cov=self._get_cov_parameter(chain_idx)
        )

    def _get_cov_parameter(self, chain_idx: int) -> np.ndarray:
        """
        Retrieve the covariance matrix for chain chain_idx from self.proposal_params.
        """
        p = self.proposal_params[chain_idx]
        if np.isscalar(p):
            return p * np.eye(self.dim)
        elif p.ndim == 1:
            return np.diag(p)
        elif p.ndim == 2:
            return p
        else:
            raise ValueError("proposal_params must be scalar, 1D, or 2D per chain.")

    def _initialize_proposal_params(self, p_init: np.ndarray) -> list:
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
        self, n_blocks: int, base: float, final_frac=0.1
    ) -> np.ndarray:
        """
        Cosine schedule: from base to final_frac*base over n_blocks.
        """
        cosvals = 0.5 * (1 + np.cos(np.linspace(0, np.pi, n_blocks)))
        return base * (final_frac + (1 - final_frac) * cosvals)

    def _validate_scheduler_args(self, n_steps_total: int, burnin: int):
        if n_steps_total < burnin:
            raise ValueError("Total steps < burnin")

    def _print_progress(
        self, iteration: int, total_steps: int, start_time: float
    ) -> None:
        """
        Print progress info on a single line, including % complete and
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
        Print final summary of total elapsed time and possibly
        # of function evaluations or other info if desired.
        """
        elapsed_time = time.time() - start_time
        print(f"  Progress: 100.00% complete | Total time: {elapsed_time:.3f}s")
        print(f"  Total proposals: {total_steps * self.n_chains}")

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
        Builds a proposal covariance matrix according to Haario’s formula:
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
        initial_states: np.ndarray,
        return_rates: bool = False,
        show_global_progress: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        x_init = np.atleast_2d(initial_states)
        if x_init.shape != (self.n_chains, self.dim):
            raise ValueError("initial_states dimension mismatch.")
        x = np.empty((self.n_chains, n_steps, self.dim))
        x[:, 0] = x_init
        accept_counts = np.zeros(self.n_chains, dtype=int)

        for t in range(1, n_steps):
            for c in range(self.n_chains):
                x_new, acc = self.mhstep(x[c, t - 1], c)
                x[c, t] = x_new
                accept_counts[c] += acc

            if show_global_progress:
                if (self._global_iter + 1) % self.options.progress_interval == 0:
                    self._print_progress(
                        self._global_iter, self._global_total, self._start_time
                    )
                self._global_iter += 1

        block_rates = accept_counts / (n_steps - 1)
        self.rates.append(block_rates)
        if return_rates:
            return x, block_rates
        return x

    def run_adaptive_RM(
        self, n_samples: int, init: np.ndarray, diminishing=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Robbins–Monro adaptation. Each block:
          gamma = schedule[block]
          proposal_params[i] *= exp(gamma*(rate[i] - target))
        If diminishing=True, use a cosine-based schedule. Otherwise, constant base=update_factor.
        """
        interval = self.options.adaptation_interval
        if interval < 2:
            raise ValueError("adaptation_interval < 2")
        if n_samples % interval != 0:
            raise ValueError("n_samples not multiple of adaptation_interval")

        n_blocks = n_samples // interval
        base = self.options.RM_adapt_factor
        gamma_seq = (
            self._diminishing_adaptation_schedule(n_blocks, base)
            if diminishing
            else np.full(n_blocks, base)
        )

        x_blocks = []
        x_curr = init
        for b in range(n_blocks):
            x_blk, rates = self.run_samples(
                interval,
                x_curr,
                return_rates=True,
                show_global_progress=self.options.show_global_progress,
            )
            gamma = gamma_seq[b]
            for c in range(self.n_chains):
                self.proposal_params[c] *= np.exp(
                    gamma * (rates[c] - self.target_acceptance)
                )
            x_blocks.append(x_blk)
            x_curr = x_blk[:, -1, :]
        return np.concatenate(x_blocks, axis=1), x_curr

    def run_adaptive_Haario(
        self,
        n_samples: int,
        init: np.ndarray,
        epsilon=1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Haario adaptation in blocks. For each block:
          - partial-pool covariance by group of n_pool
          - scale_factors[c] updated by a Robbins–Monro step
          - proposal_params[c] updated via update_proposal_covariance_from_samples(..., raw_cov=..., scaling=...)
        """
        n_interval = self.options.adaptation_interval
        if n_interval < 2:
            raise ValueError("adaptation_interval < 2")
        if n_samples % n_interval != 0:
            raise ValueError("n_samples not multiple of adaptation_interval")
        if self.n_chains % self.options.n_pool != 0:
            raise ValueError("n_chains not divisible by n_pool")

        n_blocks = n_samples // n_interval
        x_blocks = []
        x_curr = init

        for _ in range(n_blocks):
            x_blk, rates = self.run_samples(
                n_interval,
                x_curr,
                return_rates=True,
                show_global_progress=self.options.show_global_progress,
            )
            # Compute partial-pooled covariances
            covs = self._compute_covariances_for_block(x_blk, self.options.n_pool)

            for c in range(self.n_chains):
                grp = c // self.options.n_pool
                # Update the scaling factor by RM
                self.haario_scaling_factors[c] *= np.exp(
                    self.options.haario_adapt_factor
                    * (rates[c] - self.target_acceptance)
                )
                # Rebuild the proposal covariance using the unified method
                self.proposal_params[c] = self.update_proposal_covariance_from_samples(
                    raw_cov=covs[grp],
                    scaling=self.haario_scaling_factors[c],
                    epsilon=epsilon,
                )

            x_blocks.append(x_blk)
            x_curr = x_blk[:, -1, :]

        return np.concatenate(x_blocks, axis=1), x_curr

    def scheduler(
        self,
        initial_states: np.ndarray,
        n_steps_total: int,
        burnin_period: int,
        replicate_initial_state: bool = True,
    ) -> np.ndarray:
        """
        Orchestrates adaptation (burnin) then sampling, as per MHOptions.
        """
        initial_states = np.atleast_2d(initial_states)  # ensures at least 2D shape
        # If user gave a 1D array, x_init.shape will be (1, dim).
        # If replicate_init_state=True, replicate for n_chains.
        if (
            initial_states.shape == (1, self.dim)
            and replicate_initial_state
            and self.n_chains > 1
        ):
            # replicate the single init across all n_chains
            initial_states = np.tile(initial_states, (self.n_chains, 1))
        if initial_states.shape != (self.n_chains, self.dim):
            raise ValueError(
                f"initial_states must have shape ({self.n_chains}, {self.dim}) or be 1D if replicate_init_state=True. "
                f"Got {initial_states.shape}."
            )
        self._validate_scheduler_args(n_steps_total, burnin_period)
        self.proposal_params = self._initialize_proposal_params(
            self.options.proposal_param_init
        )

        # We'll track iteration from 1..n_steps_total
        self._global_iter = 0
        self._global_total = n_steps_total
        self._start_time = time.time()

        method = self.options.adaptation_method.lower()
        freeze = self.options.freeze_adaptation
        discard = self.options.discard_burnin

        # Adaptive burn-in
        print(self.options.init_msg)
        print(f"  Dimension: {self.dim}")
        print(f"  Total steps: {n_steps_total}")
        print(f"  Burn-in: {burnin_period}")
        print(f"  Chains: {self.n_chains}")

        if method == "rm":
            x_burnin, x_curr = self.run_adaptive_RM(burnin_period, initial_states)
        elif method == "haario":
            x_burnin, x_curr = self.run_adaptive_Haario(burnin_period, initial_states)
        else:
            raise ValueError("adaptation_method must be 'RM' or 'Haario'.")

        # Run with optional freeze
        remain = n_steps_total - burnin_period
        if freeze:
            x_sampling = self.run_samples(
                remain, x_curr, show_global_progress=self.options.show_global_progress
            )
        else:
            if method == "rm":
                x_sampling, x_curr = self.run_adaptive_RM(remain, x_curr)
            else:
                x_sampling, x_curr = self.run_adaptive_Haario(remain, x_curr)

        if self.options.show_global_progress:
            self._print_final_time(self._global_total, self._start_time)

        self.x = np.concatenate([x_burnin, x_sampling], axis=1)
        return self.x[:, burnin_period:] if discard else self.x

    def check_acceptance_rates(
        self, burnin_period=0, low_threshold=0.15, high_threshold=0.40
    ):
        """
        Check acceptance rates after `burnin` steps and print warnings if they
        fall below `low_threshold` or exceed `high_threshold`.

        Parameters
        ----------
        burnin_period : int, optional
            Number of initial samples to ignore. Default=0.
        low_threshold : float, optional
            Warning if mean acceptance is below this. Default=0.15.
        high_threshold : float, optional
            Warning if mean acceptance is above this. Default=0.40.
        """
        if self.x is None:
            print("No chain data available to compute acceptance rates.")
            return

        n_chains, n_steps, _ = self.x.shape
        if burnin_period >= n_steps - 1:
            print("Burn-in period is too long; cannot compute acceptance rates.")
            return

        acceptance_rates = np.zeros(n_chains, dtype=float)

        # For each chain, check how often x[i] != x[i-1] after burnin
        for c in range(n_chains):
            chain = self.x[c]
            # We consider transitions from (burnin -> burnin+1, ..., n_steps-1)
            n_accepted = 0
            n_transitions = n_steps - burnin_period - 1
            for i in range(burnin_period + 1, n_steps):
                # If the new sample differs from the old sample, it indicates acceptance.
                # For a high-dimensional chain, check if they differ in at least one dimension.
                if not np.array_equal(chain[i], chain[i - 1]):
                    n_accepted += 1

            acceptance_rates[c] = n_accepted / n_transitions

        mean_ar = acceptance_rates.mean()
        min_ar = acceptance_rates.min()
        max_ar = acceptance_rates.max()

        print("[check_acceptance_rates]")
        if mean_ar < low_threshold:
            print(
                f"WARNING: Mean acceptance rate ({mean_ar:.3f}) is below "
                f"the low threshold of {low_threshold:.2f}. "
                "You may want to increase the proposal variance."
            )
        elif mean_ar > high_threshold:
            print(
                f"WARNING: Mean acceptance rate ({mean_ar:.3f}) is above "
                f"the high threshold of {high_threshold:.2f}. "
                "You may want to decrease the proposal variance."
            )
        else:
            print("PASS: Acceptance rates within tolerance bounds")
        print(f"  Acceptance rates after burn-in = {acceptance_rates}")
        print(f"  Mean = {mean_ar:.3f},  Min = {min_ar:.3f},  Max = {max_ar:.3f}")

    def ks_statistics(
        self,
        n_blocks: int,
        n_block_size: int,
        alpha: float = 0.1,
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
            Significance level (default=0.1). If return_significance=True,
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

                    # Fill either or both
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
        alpha: float = 0.05,
        fraction_threshold: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """
        Perform two KS-diagnosis checks to assess convergence:
          1) "multi_block":  multiple blocks per chain
          2) "single_block": just one block per chain

        Parameters
        ----------
        multi_block_n_blocks : int, optional
            Number of sub-blocks to use (default=5).
        multi_block_size : int, optional
            Size of each sub-block (default=100).
        single_block_size : int, optional
            Size of the single block. If None (default), equals
            multi_block_n_blocks * multi_block_size.
        alpha : float, optional
            Significance level for the KS test (default=0.05).
        fraction_threshold : float, optional
            If the fraction of significantly similar comparisons
            is < this threshold, we consider it a convergence warning.
        verbose : bool, optional
            If True, print a short summary (default=True).

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
                "converged": bool
            }

        Notes
        -----
        - The 'multi_block' check is more sensitive to time variation within chains.
        - The 'single_block' check (with n_blocks=1) lumps the same or more samples
          to test cross-chain consistency.
        """

        if self.x is None:
            raise ValueError("No chain data. Please run or load the sampler first.")

        n_chains, n_steps, dim = self.x.shape

        # -- Multi-block check --
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
        frac_sig_multi = sigA.mean()

        # -- Single-block check (n_blocks=1) --
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
        frac_sig_single = sigB.mean()

        # Decide convergence => require both checks below threshold
        converged = (frac_sig_multi > fraction_threshold) and (
            frac_sig_single > fraction_threshold
        )

        # Prepare result dictionary
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
            "converged": converged,
        }

        # Optional console output
        if verbose:
            print("[check_convergence_ks]")
            if converged:
                print("PASS: Both KS checks below threshold.")
            else:
                print("WARNING: At least one KS check exceeded threshold.")
            print(
                f"  Multi-block: frac_significant={frac_sig_multi:.2%}  (blocks={multi_block_n_blocks}×{multi_block_size})"
            )
            print(
                f"  Single-block: frac_significant={frac_sig_single:.2%} (1×{single_block_size})"
            )
            print(f"  Threshold={fraction_threshold:.2%}, alpha={alpha}")

        return results

    def compute_gelman_rubin_rhat(self, n_block_size: int) -> np.ndarray:
        """
        Compute the Gelman–Rubin R-hat statistic for each parameter dimension
        using the last n_block_size samples of each chain.

        Parameters
        ----------
        n_block_size : int
            Number of final samples in each chain to consider.

        Returns
        -------
        rhat : np.ndarray, shape (dim,)
            Gelman–Rubin R-hat for each parameter dimension.

        Notes
        -----
        - Requires at least 2 chains.
        - Standard formula:
             Let m = number of chains, n = n_block_size (samples per chain),
                 chain_means[i, :] = mean of chain i’s last n_block_size samples
                 chain_vars[i, :]  = variance of chain i’s last n_block_size samples
             W = mean(chain_vars)
             B = n * var(chain_means)
             var_post = ((n-1)/n)*W + (1/n)*B
             R-hat = sqrt(var_post / W)
        """
        if self.x is None:
            raise ValueError("No chain data available. Run or load sampler first.")

        n_chains, n_steps, dim = self.x.shape
        if n_chains < 2:
            raise ValueError("Gelman–Rubin diagnostic requires at least 2 chains.")
        if n_block_size > n_steps:
            raise ValueError(
                f"Requested block size {n_block_size} exceeds chain length {n_steps}."
            )

        # Extract the last n_block_size samples per chain
        # shape (n_chains, n_block_size, dim)
        block = self.x[:, n_steps - n_block_size :, :]

        # Compute chain means & variances
        chain_means = np.mean(block, axis=1)  # shape (n_chains, dim)
        chain_vars = np.var(block, axis=1, ddof=1)  # shape (n_chains, dim)

        # Within-chain variance: W
        W = np.mean(chain_vars, axis=0)  # shape (dim,)

        # Between-chain variance: B
        B = n_block_size * np.var(chain_means, axis=0, ddof=1)  # shape (dim,)

        # Posterior variance estimate
        var_post = ((n_block_size - 1) / n_block_size) * W + (1.0 / n_block_size) * B

        rhat = np.sqrt(var_post / W)  # shape (dim,)
        return rhat

    def check_convergence_gelman_rubin(
        self, n_block_size: int, threshold: float = 1.1, verbose: bool = True
    ) -> dict:
        """
        Check Gelman–Rubin R-hat using the last n_block_size samples from each chain.

        Parameters
        ----------
        n_block_size : int
            Number of final samples in each chain to consider.
        threshold : float, optional
            Convergence criterion. If R-hat < threshold for all dimensions,
            we say it's converged (default=1.1).
        verbose : bool, optional
            If True, prints a short message (default=True).

        Returns
        -------
        results : dict
            {
              "rhat": np.ndarray of shape (dim,),
              "converged": bool
            }
        """
        rhat = self.compute_gelman_rubin_rhat(n_block_size)
        converged = np.all(rhat < threshold)

        if verbose:
            if converged:
                print(
                    f"[check_gelman_rubin_rhat_block]\nPASS: All R-hat < {threshold}."
                )
            else:
                print(
                    f"[check_gelman_rubin_rhat_block]\nWARNING: Some R-hat ≥ {threshold}."
                )
            print(f"  R-hat values: {rhat}")

        return {"rhat": rhat, "converged": converged}

    def plot_chains(self, burnin=0, parameter_indices=None, show_rate=True):
        """Trace plots of each dimension, optional acceptance-rate subplot."""
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
                axes[i].plot(self.x[c, :, param], label=f"Chain {c+1}")
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
            if self.rates:
                arr = np.array(self.rates)  # shape(blocks, n_chains)
                blocks = arr.shape[0]
                t = np.linspace(1, n_steps, blocks)
                for c in range(n_chains):
                    axr.plot(t, arr[:, c], marker="o", label=f"Chain {c+1}")
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
        self, burnin=0, parameter_indices=None, bins=50, smooth=True
    ):
        """Plot either histogram or KDE for each parameter dimension, chain by chain."""
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
                vals = self.x[c, burnin:, param]
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
        self, burnin=0, scaling=None, epsilon=1e-6
    ):
        """Recompute each chain's param from post-burnin samples (Haario approach)."""
        if self.x is None:
            raise ValueError("No chain data available.")
        for c in range(self.n_chains):
            x = self.x[c, burnin:]
            self.proposal_params[c] = self.update_proposal_covariance_from_samples(
                x, scaling, epsilon
            )

    def compute_empirical_covariance_whole_chain(
        self, burnin=0, pooled=False, n_pool=1
    ) -> Union[np.ndarray, list]:
        """
        If pooled=True, single covariance from all chains post-burnin.
        Otherwise, group in n_pool lumps. Return list of cov per group.
        """
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
        haario_adapt_factor=1.0,
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
        initial_states=init_states,
        n_steps_total=n_steps_total,
        burnin_period=burnin_period,
    )

    # Plot
    mh.plot_chains(burnin=burnin_period)
    mh.plot_empirical_distributions(burnin=burnin_period, smooth=False)

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
        adaptation_interval=100,
        freeze_adaptation=True,
        discard_burnin=False,
        n_pool=2,
        haario_adapt_factor=1.0,
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
    burnin_period = 2_000

    x = mh.scheduler(
        initial_states=init_states,
        n_steps_total=n_steps_total,
        burnin_period=burnin_period,
    )
    # x has shape (n_chains, n_steps_total, 2) or (n_chains, n_steps_total - burnin, 2)
    #   depending on discard_burnin (currently False => we keep burnin).

    # ------------------------------
    # 4) Convergence diagnosis
    # ------------------------------
    mh.check_acceptance_rates(burnin_period=burnin_period)
    results = mh.check_convergence_ks(
        multi_block_n_blocks=5,
        multi_block_size=100,
        single_block_size=None,
        alpha=0.1,
        fraction_threshold=0.5,
        verbose=True,
    )
    if results["converged"]:
        print("The sampler appears to have converged (KS-based check).")
    else:
        print("Non-convergence indicated by KS-based test.")
    res = mh.check_convergence_gelman_rubin(n_block_size=1000, threshold=1.1)
    if res["converged"]:
        print("Gelman-Rubin indicates convergence.")
    else:
        print("Chains may not be converged.")
    __import__("pdb").set_trace()

    # -------------------------------
    # 5) Plot chain traces & hist
    # -------------------------------
    mh.plot_chains(burnin=burnin_period)

    # Merge chains (post-burnin) for dimension-by-dimension hist
    x = x[:, burnin_period:, :].reshape(-1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(dim):
        ax = axes[i]
        x_1d = x[:, i]

        # Histogram of MCMC samples
        ax.hist(x_1d, bins=50, density=True, alpha=0.5, label="MCMC samples")

        # Overlay the *true* marginal PDF in red
        xs = np.linspace(0.01, 15.0, 300) * scale[i]
        # dimension i's scale
        scale_i = scale[0] if i == 0 else scale[1]
        pdf_vals = pdf_1d(xs, df=dof, scale=scale_i)

        ax.plot(xs, pdf_vals, "r-", label="True marginal PDF")
        ax.set_xlabel(f"Dimension {i}")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle(f"2D chi^2(ν={dof}) – Marginal Distributions", fontsize=14)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 6) 2D contour of joint PDF + scatter
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
        f"2D chi^2(ν={dof}): PDF Contour + Samples\n(scale0={scale[0]}, scale1={scale[1]})"
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #  case = "linear_regression"
    case = "chi2"

    if case == "linear_regression":
        test_linear_regression()
    elif case == "chi2":
        test_chi2()
