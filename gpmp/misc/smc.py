"""Sequential Monte Carlo (SMC) sampler implementation

This module provides the `ParticlesSet` and `SMC` classes for SMC
simulations, along with a `run_smc_sampling` function to execute full
sampling workflows. A test example using a 1D Gaussian mixture
illustrates the functionality.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Contributions:
    Julien Bect <julien.bect@centralesupelec.fr>, 2024
    Sébastien Petit, 2024
Copyright (c) 2022-2026 CentraleSupelec
License: GPLv3 (see LICENSE)

"""

import time, warnings
from dataclasses import dataclass, field
from numpy.random import default_rng
import scipy.stats as stats
from scipy.stats import qmc
from scipy.optimize import brentq
import gpmp.num as gnp
import gpmp.misc.knn_cov


@dataclass
class ParticlesSetConfig:
    initial_distribution_type: str = "randunif"
    resample_scheme: str = "multinomial"
    param_s_initial_value: float = 0.5
    param_s_upper_bound: float = 1e5
    param_s_lower_bound: float = 1e-3
    jitter_initial_value: float = 1e-16
    jitter_max_iterations: int = 10
    covariance_method: str = "normal"
    covariance_knn_n_random: int = 20
    covariance_knn_n_neighbors: int = 200


@dataclass
class SMCConfig:
    compute_next_logpdf_param_method: str = "p0"  # or "ess"
    mh_steps: int = 20
    mh_acceptation_rate_min: float = 0.15
    mh_acceptation_rate_max: float = 0.30
    mh_adjustment_factor: float = 1.4
    mh_adjustment_max_iterations: int = 50


class ParticlesSetError(BaseException):
    def __init__(self, param_s, lower, upper):
        message = (
            "ParticlesSet: scaling parameter param_s in MH step out of range "
            "(value: {}, lower bound: {}, upper bound: {}).".format(
                param_s, lower, upper
            )
        )
        super().__init__(message)


class ParticlesSet:
    """
    Defines a set of particles for SMC simulation.

    This class initializes, reweights, resamples, perturbs, and moves particles.
    Configuration is passed via a ParticlesSetConfig object (defaults are used if not provided).

    Parameters
    ----------
    box : array_like
        Domain bounds for initialization ([lower bounds, upper bounds]).
    n : int, optional
        Number of particles (default: 1000).
    config : ParticlesSetConfig, optional
        Configuration for initialization, resampling, and perturbation.
    rng : numpy.random.Generator, optional
        Random number generator (defaults to default_rng()).

    Attributes
    ----------
    n : int
        Number of particles.
    dim : int
        Dimension of the particles.
    x : ndarray
        Particle positions.
    logpx : ndarray
        Log-probabilities at current positions.
    w : ndarray
        Particle weights.
    param_s : float
        Scaling parameter for Gaussian perturbations.
    config : ParticlesSetConfig
        The configuration object.
    rng : numpy.random.Generator
        The random number generator.

    Methods
    -------
    particles_init(box, n, method)
        Initialize particles within the provided bounds.
    set_logpdf(logpdf_function)
        Set the log-density function.
    set_logpdf_with_parameter(logpdf_parameterized_function, param)
        Set the log-density using a parameterized function.
    reweight(update_logpx_and_w_pre=True)
        Update particle weights based on current log-probabilities.
    ess()
        Compute the effective sample size.
    resample(debug=False)
        Resample particles using the selected scheme.
    multinomial_resample(debug=False)
        Resample using multinomial sampling.
    residual_resample(debug=False)
        Resample using residual resampling.
    perturb()
        Perturb particles by adding scaled Gaussian noise.
    move()
        Execute a vectorized Metropolis-Hastings move and return the acceptance rate.
    """

    def __init__(
        self, box, n=1000, config: ParticlesSetConfig = None, rng=default_rng()
    ):
        """
        Initialize the ParticlesSet instance.
        """
        self.n = n  # Number of particles
        self.dim = len(box[0])
        self.rng = rng
        self.config = config if config is not None else ParticlesSetConfig()

        # Initialize particle parameters using config
        self.param_s = self.config.param_s_initial_value

        # Initialize the particles.
        self.x = None
        self.logpx = None
        self.w = None
        self.w_tmp = None
        self.particles_init(box, n, method=self.config.initial_distribution_type)
        self.logpdf_function = None

    def particles_init(self, box, n, method="randunif"):
        """Initialize particles within the given box.

        Parameters
        ----------
        box : array_like
            The domain box in which the particles are to be initialized.
        n : int
            Number of particles.
        method : str, optional
            Method for initializing particles. Currently, only
            'randunif' (uniform random) is supported. The option 'qmc'
            (quasi Monte-Carlo) will be supported in future versions.

        Returns
        -------
        tuple
            A tuple containing the positions, log-probabilities, and
            weights of the initialized particles.

        FIXME
        -----
        Implement more general initial densities

        """
        assert self.dim == len(
            box[0]
        ), "Box dimension does not match particles dimension"
        self.n = n

        # Initialize positions
        if method == "randunif":
            self.x = ParticlesSet.randunif(self.dim, self.n, box, self.rng)
        else:
            raise NotImplementedError(
                f"The method '{method}' is not supported. Currently, only 'randunif' is available."
            )

        # Initialize log-probabilities and weights
        self.logpx = gnp.zeros((n,))
        self.w_tmp = gnp.full((n,), 1 / n)
        self.w = gnp.full((n,), 1 / n)  # pre-update values of self.w

    def set_logpdf(self, logpdf_function):
        """
        Set the log-probability density function for the particles.

        Parameters
        ----------
        logpdf_function : callable
            Computes the log-probability density at given positions.
        """
        self.logpdf_function = logpdf_function

    def set_logpdf_with_parameter(self, logpdf_parameterized_function, param):
        """Set the log-probability density function for the particles
        with a prescribed parameter.

        Parameters
        ----------
        logpdf_function : callable
            Computes the log-probability density at given positions.
        param : ndarray
            Parameter for the logpdf_function (second argment of logpdf_function)
        """

        def logpdf(x):
            return logpdf_parameterized_function(x, param)

        self.logpdf_function = logpdf

    def reweight(self, update_logpx_and_w=True):
        logpx_new = self.logpdf_function(self.x).reshape(-1)
        self.w_tmp = self.w * gnp.exp(logpx_new - self.logpx)
        if update_logpx_and_w:
            self.logpx = logpx_new
            self.w = gnp.copy(self.w_tmp)

    def ess(self):
        """https://en.wikipedia.org/wiki/Effective_sample_size"""
        normalization = gnp.sum(self.w_tmp**2)
        if normalization == 0.0:
            return 0.0
        else:
            return gnp.sum(self.w_tmp) ** 2 / normalization

    def resample(self, debug=False):
        """
        Resample the particles based on the chosen resampling scheme.

        The resample method routes to either multinomial_resample or
        residual_resample according to self.resample_scheme.
        """
        if self.config.resample_scheme == "multinomial":
            self.multinomial_resample(debug=debug)
        elif self.config.resample_scheme == "residual":
            self.residual_resample(debug=debug)
        else:
            raise ValueError("Unknown resample scheme: {}".format(self.resample_scheme))

    def multinomial_resample(self, debug=False):
        """
        Resample using multinomial resampling.

        This method assigns offspring counts to particles according
        to a multinomial distribution.
        """
        x_resampled = gnp.empty(self.x.shape)
        logpx_resampled = gnp.empty((self.n,))
        normalization = gnp.sum(self.w_tmp)
        if normalization == 0.0:
            p = gnp.full((self.n,), 1 / self.n)
        else:
            p = self.w_tmp / gnp.sum(self.w_tmp)
        try:
            counts = ParticlesSet.multinomial_rvs(self.n, p, self.rng)
        except Exception:
            extype, value, tb = __import__("sys").exc_info()
            __import__("traceback").print_exc()
            __import__("pdb").post_mortem(tb)

        if debug:
            print(
                f"Multinomial resample: proportion discarded = {gnp.sum(counts==0) / self.n} "
            )

        i = 0
        j = 0
        while j < self.n:
            while counts[j] > 0:
                x_resampled[i] = self.x[j, :]
                logpx_resampled[i] = self.logpx[j]
                counts[j] = counts[j] - 1
                i += 1
            j += 1

        self.x = x_resampled
        self.logpx = logpx_resampled
        self.w_tmp = gnp.full((self.n,), 1 / self.n)
        self.w = gnp.full((self.n,), 1 / self.n)

    def residual_resample(self, debug=False):
        """
        Resample using residual resampling.

        This method reduces variance by first assigning a deterministic
        number of copies to each particle and then assigning the remainder
        via multinomial sampling.
        """
        N = self.n
        x_resampled = gnp.empty(self.x.shape)
        logpx_resampled = gnp.empty(self.logpx.shape)
        normalization = gnp.sum(self.w_tmp)
        if normalization == 0.0:
            p = gnp.full((self.n,), 1 / self.n)
        else:
            p = self.w_tmp / gnp.sum(self.w_tmp)

        # Deterministic assignment: floor of expected counts
        counts_det = gnp.asint(gnp.floor(N * p))
        N_det = int(gnp.sum(counts_det))

        # Compute residuals
        residuals = N * p - counts_det
        N_residual = N - N_det

        # Multinomial step on residuals
        if N_residual > 0:
            residuals = gnp.maximum(residuals, 0.0)
            total_residual = gnp.sum(residuals)
            if total_residual == 0.0:
                p_vals = gnp.full_like(residuals, 1.0 / len(residuals))
            else:
                p_vals = residuals / total_residual

            # Defensive check before multinomial draw
            if gnp.any(p_vals < 0) or gnp.any(p_vals > 1) or gnp.any(gnp.isnan(p_vals)):
                print("Residual resampling error: invalid p_vals.")
                __import__("pdb").set_trace()

            counts_res = ParticlesSet.multinomial_rvs(N_residual, p_vals, self.rng)
        else:
            counts_res = gnp.zeros_like(counts_det)

        # Total counts
        counts = counts_det + counts_res

        if debug:
            print(
                f"Residual resample: proportion discarded = {gnp.sum(counts==0) / self.n} "
            )

        i = 0
        j = 0
        while j < self.n:
            while counts[j] > 0:
                x_resampled[i] = self.x[j, :]
                logpx_resampled[i] = self.logpx[j]
                counts[j] = counts[j] - 1
                i += 1
            j += 1

        self.x = x_resampled
        self.logpx = logpx_resampled
        self.w_tmp = gnp.full((self.n,), 1 / self.n)
        self.w = gnp.full((self.n,), 1 / self.n)

    def perturb(self):
        """Perturb the particles by adding Gaussian noise.

        This method perturbs the current positions of the particles by
        applying a Gaussian random perturbation. The covariance matrix
        is computed from the particles' current positions and is
        scaled by the perturbation parameter `param_s`. The
        covariance matrix defines the spread of the Gaussian noise
        used to move the particles.

        """

        param_s_lower = self.config.param_s_lower_bound
        param_s_upper = self.config.param_s_upper_bound

        # Check if param_s is within bounds
        if self.param_s > param_s_upper or self.param_s < param_s_lower:
            raise ParticlesSetError(self.param_s, param_s_lower, param_s_upper)

        # Covariance matrix of the pertubation noise
        if self.config.covariance_method == "knn":
            base_cov = gpmp.misc.knn_cov.estimate_cov_matrix_knn(
                self.x,
                n_random=self.config.covariance_knn_n_random,
                n_neighbors=self.config.covariance_knn_n_neighbors,
            )  # shape (dim, dim)
        elif self.config.covariance_method == "normal":
            base_cov = gpmp.misc.knn_cov.estimate_cov_matrix(self.x)
        C = self.param_s * base_cov

        # Call ParticlesSet.multivariate_normal_rvs(C, self.n, self.rng)
        # with control on the possible degeneracy of C
        try:
            eps = ParticlesSet.multivariate_normal_rvs(C, self.n, self.rng)
            success = True
        except ValueError as e:
            # If the covariance matrix is not PSD, apply jittering to fix it
            print(f"Non-PSD covariance matrix encountered: {e}")
            success = False
            for i in range(
                self.config.jitter_max_iterations
            ):  # Try iterations of jittering
                jitter = self.config.jitter_initial_value * (10**i)
                C_jittered = C + jitter * np.eye(C.shape[0])  # Add jitter
                try:
                    eps = ParticlesSet.multivariate_normal_rvs(
                        C_jittered, self.n, self.rng
                    )
                    success = True
                    break
                except ValueError as inner_e:
                    print(f"Jittering attempt {i} failed: {inner_e}")

        if not success:
            raise RuntimeError(
                "Failed to generate samples after "
                + f"{self.config.jitter_max_iterations} jittering attempts. "
                + "Covariance matrix might still be non-PSD."
            )

        return self.x + eps.reshape(self.n, -1)

    def move(self):
        """
        Perform a Metropolis-Hastings step and compute the acceptation rate.

        This method perturbs the particles, computes the acceptation probabilities, and
        decides whether to move the particles to their new positions.

        Returns
        -------
        float
            Acceptation rate of the move.
        """
        # Perturb the particles
        y = self.perturb()
        logpy = self.logpdf_function(y).reshape(-1)

        # Determine which moves are accepted
        logrho = logpy - self.logpx
        u = ParticlesSet.rand((self.n,), self.rng)
        accept_mask = gnp.log(u) < logrho

        # Update
        self.x[accept_mask, :] = y[accept_mask, :]
        self.logpx[accept_mask] = logpy[accept_mask]

        # Compute the acceptance rate
        acceptance_rate = gnp.sum(accept_mask) / self.n

        return acceptance_rate

    @staticmethod
    def rand(size, rng):
        return rng.uniform(size=size)

    @staticmethod
    def multinomial_rvs(n, p, rng):
        return gnp.asarray(stats.multinomial.rvs(n=n, p=p, random_state=rng))

    @staticmethod
    def multivariate_normal_rvs(C, n, rng):
        return gnp.asarray(
            stats.multivariate_normal.rvs(cov=C, size=n, random_state=rng)
        )

    @staticmethod
    def randunif(dim, n, box, rng):
        return gnp.asarray(qmc.scale(rng.uniform(size=(n, dim)), box[0], box[1]))


class SMC:
    """
    Sequential Monte Carlo (SMC) sampler.

    Drives the SMC process using a set of particles, following the approach
    in Bect, J., Li, L., & Vazquez, E. (2017), "Bayesian subset simulation",
    SIAM/ASA Journal on Uncertainty Quantification, 5(1), 762-786.
    https://arxiv.org/abs/1601.02557

    Parameters
    ----------
    box : array_like
        Domain bounds for particle initialization ([lower bounds, upper bounds]).
    n : int, optional
        Number of particles (default: 1000).
    particles_config : ParticlesSetConfig, optional
        Configuration for the particle set; default settings are used if None.
    smc_config : SMCConfig, optional
        Configuration for SMC options (e.g., MH steps, thresholds); default settings are used if None.
    rng : numpy.random.Generator, optional
        Random number generator (default: default_rng()).

    Attributes
    ----------
    box : array_like
        Domain for particle initialization.
    n : int
        Number of particles.
    particles : ParticlesSet
        Manages particle operations (initialization, resampling, move, etc.).
    log : list
        List of diagnostic snapshots.
    stage : int
        Current stage of the SMC process.

    Methods
    -------
    step(logpdf_parameterized_function, logpdf_param)
        Execute a single SMC step.
    step_with_possible_restart(logpdf_parameterized_function, initial_logpdf_param, target_logpdf_param, min_ess_ratio, p0, debug=False)
        Execute an SMC step and restart if the effective sample size is too low.
    restart(logpdf_parameterized_function, initial_logpdf_param, target_logpdf_param, min_ess_ratio, p0, debug=False)
        Restart the SMC process with an updated logpdf parameter sequence.
    move_with_controlled_acceptation_rate(debug=False)
        Adjust particle moves to maintain the target acceptance rate.
    compute_next_logpdf_param_ess(...)
        Compute the next logpdf parameter based on effective sample size.
    compute_next_logpdf_param_p0(...)
        Compute the next logpdf parameter using a prescribed probability.
    compute_p_value(logpdf_function, new_logpdf_param, current_logpdf_param)
        Compute the average exponentiated difference in logpdf values.
    plot_state()
        Plot diagnostic data (logpdf parameters, ESS, acceptance rates over time).
    plot_particles()
        Display a matrix plot of particle positions.
    plot_empirical_distributions(parameter_indices=None, parameter_indices_pooled=None, bins=50)
        Plot histograms or KDEs for the particle distribution.
    """

    def __init__(
        self,
        box,
        n=2000,
        particles_config: ParticlesSetConfig = None,
        smc_config: SMCConfig = None,
        rng=default_rng(),
    ):
        self.box = box
        self.n = n
        self.particles_config = (
            particles_config if particles_config is not None else ParticlesSetConfig()
        )
        self.smc_config = smc_config if smc_config is not None else SMCConfig()
        self.particles = ParticlesSet(box, n, config=self.particles_config, rng=rng)

        # Set up next logpdf parameter method.
        method = self.smc_config.compute_next_logpdf_param_method
        if method == "p0":
            self.compute_next_logpdf_param = self.compute_next_logpdf_param_p0
        elif method == "ess":
            self.compute_next_logpdf_param = self.compute_next_logpdf_param_ess
        else:
            raise ValueError("compute_next_logpdf_param_method must be 'ess' or 'p0'.")

        # Logging: self.log will store full state snapshots.
        self.log = []
        self.stage = 0
        # log_data holds incremental values that can be be frozen with self.log_snapshot().
        self.log_data = {
            "current_logpdf_param": None,
            "ess": None,
            "target_logpdf_param": None,
            "restart_iteration": 0,
            "logpdf_param_sequence": [],
            "acceptation_rate_sequence": [],
            "execution_state": None,  # New field to store the current execution state.
        }

    def update_log(
        self, logpdf_param=None, ess=None, acceptation_rate=None, state=None
    ):
        """
        Update the incremental logging dictionary.
        Optionally record the execution state with extra context.
        """
        if logpdf_param is not None:
            self.log_data["current_logpdf_param"] = logpdf_param
        if ess is not None:
            self.log_data["ess"] = ess
        if acceptation_rate is not None:
            self.log_data["acceptation_rate_sequence"].append(acceptation_rate)
        if state is not None:
            # Append additional context information to the state string.
            self.log_data["execution_state"] = f"[Stage {self.stage}] {state}"

    def log_snapshot(self):
        """
        Freeze the current logging state into a snapshot and append it to self.log.
        The snapshot includes the current execution state.
        """
        snapshot = {
            "timestamp": time.time(),
            "stage": self.stage,
            "num_particles": self.n,
            "current_scaling_param": self.particles.param_s,
            "target_logpdf_param": self.log_data["target_logpdf_param"],
            "current_logpdf_param": self.log_data["current_logpdf_param"],
            "ess": self.log_data["ess"],
            "restart_iteration": self.log_data["restart_iteration"],
            "logpdf_param_sequence": self.log_data["logpdf_param_sequence"].copy(),
            "acceptation_rate_sequence": self.log_data[
                "acceptation_rate_sequence"
            ].copy(),
            "execution_state": self.log_data["execution_state"],
        }
        self.log.append(snapshot)
        # Reset the incremental acceptation rate sequence for the next stage.
        self.log_data["acceptation_rate_sequence"] = []

    def step(
        self, logpdf_parameterized_function, logpdf_param, debug=False, debug_plot=False
    ):
        """
        Perform a single step of the SMC process.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            A function that computes the log-probability density at
            given positions.
        logpdf_param: float
            Parameter value for the logpdf function (typically, a threshold).

        """
        # Set target density.
        self.update_log(state=f"Step start: set logpdf_param to {logpdf_param}")
        self.particles.set_logpdf_with_parameter(
            logpdf_parameterized_function, logpdf_param
        )

        # Reweight
        self.update_log(state=f"Reweight with logpdf_param = {logpdf_param}")
        self.particles.reweight()

        # Log current logpdf_param and ESS
        ess_value = self.particles.ess()
        self.update_log(logpdf_param=logpdf_param, ess=ess_value)

        # Resample
        self.update_log(state=f"Resample particles (ESS = {ess_value})")
        self.particles.resample(debug)

        # Move particles
        self.update_log(state="Move particles with controlled acceptation rate")
        self.move_with_controlled_acceptation_rate(debug)
        self.log_snapshot()

        for i in range(self.smc_config.mh_steps - 1):
            acceptation_rate = self.particles.move()
            self.update_log(
                acceptation_rate=acceptation_rate,
                state=f"Additional move {i+1}/{self.smc_config.mh_steps-1} with acceptation rate {acceptation_rate:.2f}",
            )

        # Log a snapshot of the full current state.
        self.log_snapshot()

        if debug_plot:
            self.plot_particles()

    def step_with_possible_restart(
        self,
        logpdf_parameterized_function,
        initial_logpdf_param,
        target_logpdf_param,
        min_ess_ratio,
        p0,
        debug=False,
    ):
        """Perform an SMC step with the possibility of restarting the process.

        This method checks if the effective sample size (ESS) is
        below a specified ratio, and if so, initiates a "restart". The
        restart process reinitializes particles and calculates a sequence of
        logpdf_params to target the desired distribution.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            A function that computes the log-probability density of a
            given position.
        initial_logpdf_param : float
            The starting logpdf_param value for the restart process.
        target_logpdf_param : float
            The desired target logpdf_param value for the log-probability
            density.
        min_ess_ratio : float
            The minimum acceptable ratio of ESS to the total number of
            particles. If the ESS falls below this ratio, a restart is
            initiated.
        p0 : float
            The prescribed probability used in the restart method to
            compute the new logpdf_param.
        debug : bool, optional
            If True, prints debug information during the
            process. Default is False.
        """
        # Logging
        self.stage += 1
        self.update_log(state="Starting step_with_possible_restart")
        self.log_data["current_logpdf_param"] = target_logpdf_param
        self.log_data["target_logpdf_param"] = target_logpdf_param
        self.log_snapshot()

        # Set logpdf with target param, compute ess
        self.particles.set_logpdf_with_parameter(
            logpdf_parameterized_function, target_logpdf_param
        )
        self.update_log(state="Computing initial ESS in step_with_possible_restart")
        self.particles.reweight(update_logpx_and_w=False)
        ess = self.particles.ess()
        self.update_log(ess=ess)

        # Restart?
        if ess / self.n < min_ess_ratio:
            self.update_log(
                state=f"ESS ratio ({ess/self.n:.2f}) below threshold ({min_ess_ratio}), initiating restart"
            )
            self.log_snapshot()
            self.restart(
                logpdf_parameterized_function,
                initial_logpdf_param,
                target_logpdf_param,
                min_ess_ratio,
                p0,
                debug=debug,
            )
        else:  # Normal step
            self.update_log(
                state="ESS acceptable, proceeding with resampling and moves"
            )
            self.log_snapshot()
            self.step(logpdf_parameterized_function, target_logpdf_param)

    def restart(
        self,
        logpdf_parameterized_function,
        initial_logpdf_param,
        target_logpdf_param,
        min_ess_ratio,
        p0,
        debug=False,
    ):
        """
        Perform a restart method in SMC.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            Parametric probability density
        initial_logpdf_param : float
            Starting param value.
        target_logpdf_param : float
            Target param value.
        min_ess_ratio : float
            Minimum ESS ratio
        p0 : float
            Prescribed probability
        debug : bool
            If True, print debug information.
        """
        if debug:
            print("---- (Re)starting SMC from initial parameter ----")
        self.update_log(state="Restarting: taking snapshot before restart")
        self.log_snapshot()

        if self.smc_config.compute_next_logpdf_param_method == "p0":
            threshold = p0
        elif self.smc_config.compute_next_logpdf_param_method == "ess":
            threshold = min_ess_ratio

        self.update_log(state="Reinitializing particles with initial distribution")
        self.particles.particles_init(
            self.box, self.n, method=self.particles_config.initial_distribution_type
        )

        # Is the initial threshold too difficult?
        self.particles.set_logpdf_with_parameter(
            logpdf_parameterized_function, initial_logpdf_param
        )
        self.particles.reweight(update_logpx_and_w=False)
        ess = self.particles.ess()
        if ess / self.n < min_ess_ratio:
            warnings.warn(
                f"ESS ratio {ess / self.n} below threshold={min_ess_ratio} at initialization.",
                RuntimeWarning,
            )

        current_logpdf_param = initial_logpdf_param
        self.log_data["logpdf_param_sequence"] = [initial_logpdf_param]

        while current_logpdf_param != target_logpdf_param:
            next_logpdf_param = self.compute_next_logpdf_param(
                logpdf_parameterized_function,
                current_logpdf_param,
                target_logpdf_param,
                threshold,
                debug=debug,
            )

            self.log_data["restart_iteration"] += 1
            self.log_data["logpdf_param_sequence"].append(next_logpdf_param)
            self.update_log(
                state=f"Restart loop iteration {self.log_data['restart_iteration']}: stepping with logpdf_param {next_logpdf_param}"
            )
            self.log_snapshot()

            self.step(logpdf_parameterized_function, next_logpdf_param, debug=debug)
            current_logpdf_param = next_logpdf_param

        self.log_data["logpdf_param_sequence"] = []
        self.log_data["restart_iteration"] = 0

    def move_with_controlled_acceptation_rate(self, debug=False):
        """
        Adjust the particles' movement to maintain the acceptation
        rate within specified bounds.  This method dynamically adjusts
        the scaling parameter based on the acceptation rate to ensure
        efficient exploration of the state space.

        """
        self.update_log(state="Entering move_with_controlled_acceptation_rate")
        iteration_counter = 0
        while iteration_counter < self.smc_config.mh_adjustment_max_iterations:
            iteration_counter += 1
            acceptation_rate = self.particles.move()
            self.update_log(
                acceptation_rate=acceptation_rate,
                state=f"Controlled move iteration {iteration_counter} with acceptation rate {acceptation_rate:.2f}",
            )
            if debug:
                print(f"Acceptation rate = {acceptation_rate:.2f}")

            if acceptation_rate < self.smc_config.mh_acceptation_rate_min:
                self.particles.param_s /= self.smc_config.mh_adjustment_factor
                self.update_log(
                    state=f"Acceptation rate low ({acceptation_rate:.2f}); decreasing param_s to {self.particles.param_s:.2e}"
                )
                continue
            if acceptation_rate > self.smc_config.mh_acceptation_rate_max:
                self.particles.param_s *= self.smc_config.mh_adjustment_factor
                self.update_log(
                    state=f"Acceptation rate high ({acceptation_rate:.2f}); increasing param_s to {self.particles.param_s:.2e}"
                )
                continue
            break

    def compute_next_logpdf_param_ess(
        self,
        logpdf_parameterized_function,
        current_logpdf_param,
        target_logpdf_param,
        eta0,
        debug=False,
    ):
        """
        Compute the next logpdf_param using a dichotomy method.

        This method is part of the restart strategy. It computes a
        logpdf_param for the parameter of the
        logpdf_parameterized_function, ensuring a controlled migration
        of particles to the next stage. The parameter p0 corresponds
        to the fraction of moved particles that will be in the support
        of the target density.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            Parametric log-probability density.
        current_logpdf_param : float
            Starting logpdf_param value.
        target_logpdf_param : float
            Target logpdf_param value.
        p0 : float
            Prescribed probability.
        debug : bool
            If True, print debug information.

        Returns
        -------
        float
            Next computed logpdf_param.

        """
        tolerance = 0.05
        param_easy = current_logpdf_param
        param_difficult = target_logpdf_param

        def compute_delta_eta(logpdf_param):
            self.particles.set_logpdf_with_parameter(
                logpdf_parameterized_function, logpdf_param
            )
            self.particles.reweight(update_logpx_and_w=False)
            eta = self.particles.ess() / self.particles.n
            if debug:
                print(
                    f"Search: eta = {eta:.2f} / eta0 = {eta0:.2f}, "
                    + f"test logpdf_param = {logpdf_param:.3e}, "
                    + f"current = {current_logpdf_param:.3e}, "
                    + f"target = {target_logpdf_param:.3e}"
                )
            return eta - eta0

        # Can we reach the target?
        if compute_delta_eta(target_logpdf_param) > 0:
            if debug:
                print(f"Target logpdf_param reached, current = {target_logpdf_param}.")
            return target_logpdf_param
        else:
            low = gnp.minimum(param_difficult, param_easy)
            high = gnp.maximum(param_difficult, param_easy)
            next_logpdf_param = brentq(compute_delta_eta, low, high, xtol=tolerance)

            return next_logpdf_param

    def compute_p_value(self, logpdf_function, new_logpdf_param, current_logpdf_param):
        """
        Compute the mean value of the exponentiated difference in
        log-probability densities between two logpdf_params.

        .. math::

            \\frac{1}{n} \\sum_{i=1}^{n} \\exp(logpdf_function(x_i, new_logpdf_param)
            - logpdf_function(x_i, current_logpdf_param))

        Parameters
        ----------
        logpdf_function : callable
            Function to compute log-probability density.
        new_logpdf_param : float
            The new logpdf_param value.
        current_logpdf_param : float
            The current logpdf_param value used as a reference.

        Returns
        -------
        float
            Computed mean value.

        """
        return gnp.mean(
            gnp.exp(
                logpdf_function(self.particles.x, new_logpdf_param)
                - logpdf_function(self.particles.x, current_logpdf_param)
            )
        )

    def compute_next_logpdf_param_p0(
        self,
        logpdf_parameterized_function,
        current_logpdf_param,
        target_logpdf_param,
        p0,
        debug=False,
    ):
        """
        Compute the next logpdf_param using a dichotomy method.

        This method is part of the restart strategy. It computes a
        logpdf_param for the parameter of the
        logpdf_parameterized_function, ensuring a controlled migration
        of particles to the next stage. The parameter p0 corresponds
        to the fraction of moved particles that will be in the support
        of the target density.

        Parameters
        ----------
        logpdf_parameterized_function : callable
            Parametric log-probability density.
        current_logpdf_param : float
            Starting logpdf_param value.
        target_logpdf_param : float
            Target logpdf_param value.
        p0 : float
            Prescribed probability.
        debug : bool
            If True, print debug information.

        Returns
        -------
        float
            Next computed logpdf_param.

        """
        tolerance = 0.05
        low = current_logpdf_param
        high = target_logpdf_param

        # Check if target_logpdf_param can be reached with p >= p0
        p_target = self.compute_p_value(
            logpdf_parameterized_function, target_logpdf_param, current_logpdf_param
        )
        if p_target >= p0:
            if debug:
                print("Target logpdf_param reached.")
            return target_logpdf_param

        while True:
            mid = (high + low) / 2
            p = self.compute_p_value(
                logpdf_parameterized_function, mid, current_logpdf_param
            )

            if debug:
                print(
                    f"Search: p = {p:.2f} / p0 = {p0:.2f}, "
                    + f"test logpdf_param = {mid}, "
                    + f"current = {current_logpdf_param}, "
                    + f"target = {target_logpdf_param}"
                )

            if abs(p - p0) < tolerance:
                break
            if p < p0:
                high = mid
            else:
                low = mid

        return mid

    def plot_state(self):
        """Plot the state of the SMC process over different stages.

        It includes visualizations of logpdf_params, effective sample
        size (ESS), and acceptation rates.
        """
        import matplotlib.pyplot as plt

        log_data = self.log

        def make_stairs(y):
            x_stairs = []
            y_stairs = []
            for i in range(len(y)):
                x_stairs.extend([i, i + 1])
                y_stairs.extend([y[i], y[i]])
            return x_stairs, y_stairs

        # Initializing lists to store data
        stages = []
        target_logpdf_params = []
        current_logpdf_params = []
        ess_values = []
        acceptation_rates = []
        execution_states = []  # Collect execution state info

        # Extracting and replicating data according to the length of 'acceptation_rate_sequence' in each log entry
        for entry in log_data:
            ar_seq = entry.get("acceptation_rate_sequence", [])
            if len(ar_seq) == 0:
                ar_seq = [0.0]
            n_ar = len(ar_seq)
            stages.extend([entry["stage"]] * n_ar)
            target_logpdf_params.extend([entry["target_logpdf_param"]] * n_ar)
            current_logpdf_params.extend([entry["current_logpdf_param"]] * n_ar)
            ess_values.extend([entry["ess"]] * n_ar)
            acceptation_rates.extend(ar_seq)
            execution_states.extend([entry["execution_state"]] * n_ar)

        # Plotting
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Time")
        ax1.set_ylabel("logpdf_param", color=color)
        t, target_stairs = make_stairs(target_logpdf_params)
        t, current_stairs = make_stairs(current_logpdf_params)
        ax1.plot(
            t,
            target_stairs,
            label="Target logpdf_param",
            color="red",
            linestyle="dashed",
        )
        ax1.plot(
            t,
            current_stairs,
            label="Current logpdf_param",
            color="red",
            linestyle="solid",
        )
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax * 1.2)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.legend(loc="upper left")

        # Adding vertical lines for stage changes
        last_stage = 0
        for idx, stage in enumerate(stages):
            if stage > last_stage:
                plt.axvline(x=idx, color="gray", linestyle="dashed")
                last_stage = stage

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("ESS", color=color)
        t, ess_stairs = make_stairs(ess_values)
        ax2.plot(t, ess_stairs, label="ESS", color=color)
        ax2.set_ylim(0.0, self.n)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.legend(loc="upper right")

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        color = "tab:green"
        ax3.set_ylabel("Acceptation Rate", color=color)
        ax3.plot(
            acceptation_rates, label="Acceptation Rate", color=color, linestyle="dotted"
        )
        ax3.set_ylim(0.0, 1.0)
        ax3.tick_params(axis="y", labelcolor=color)
        ax3.legend(loc="lower right")

        fig.tight_layout()
        plt.title("SMC Process State Over Stages")
        plt.show()

    def plot_particles(self):

        from gpmpcontrib.plot.visualization import plotmatrix

        plotmatrix(
            gnp.hstack((self.particles.x, self.particles.logpx.reshape(-1, 1))),
            self.particles.logpx,
        )

    def plot_empirical_distributions(
        self, parameter_indices=None, parameter_indices_pooled=None, bins=50
    ):
        """Plot histograms or KDEs for selected parameter dimensions.

        If `parameter_indices_pooled` is not None, plot multiple marginals on the same figure.
        """
        import matplotlib.pyplot as plt
        from itertools import cycle

        if self.particles.x is None:
            raise ValueError("No particles data.")
        n, dim = self.particles.x.shape

        color_cycler = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Individual plots
        if parameter_indices is not None:
            pidx = parameter_indices
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
                vals = self.particles.x[:, param]
                color = next(colors)
                ax.hist(
                    vals,
                    bins=bins,
                    density=True,
                    alpha=0.3,
                    color=color,
                    label=f"$\\theta_{{{param+1}}}$",
                )

                lo, hi = vals.min(), vals.max()
                xx = gnp.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 100)
                kde = stats.gaussian_kde(vals, bw_method="scott")
                ax.plot(xx, kde(xx), color=color)
                ax.set_xlabel(rf"$\theta_{{{param+1}}}$")
                ax.set_ylabel("Density")
                ax.legend()
            plt.tight_layout()
            plt.show()

        # Pooled: plot several marginals on the same axis
        if parameter_indices_pooled is not None:
            fig, ax = plt.subplots(figsize=(8, 3))
            colors = cycle(color_cycler)
            for param in parameter_indices_pooled:
                vals = self.particles.x[:, param]
                color = next(colors)
                ax.hist(
                    vals,
                    bins=bins,
                    density=True,
                    alpha=0.3,
                    color=color,
                    label=f"$\\theta_{{{param+1}}}$",
                )
                lo, hi = vals.min(), vals.max()
                xx = gnp.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 100)
                kde = stats.gaussian_kde(vals, bw_method="scott")
                ax.plot(xx, kde(xx), color=color)
            ax.set_xlabel(r"$\theta_i$")
            ax.set_ylabel("Density")
            ax.set_title("Marginal distributions")
            ax.legend()
            plt.tight_layout()
            plt.show()


def run_smc_sampling(
    logpdf_parameterized_function,
    initial_logpdf_param: float,
    target_logpdf_param: float,
    compute_next_logpdf_param_method,
    min_ess_ratio: float,
    p0: float = None,
    init_box: list = None,
    n_particles: int = 1000,
    mh_steps: int = 20,
    smc_config: SMCConfig = None,
    particles_config: ParticlesSetConfig = None,
    debug: bool = False,
    plot_particles: bool = False,
    plot_empirical_distributions: bool = False,
):
    """Run a full Sequential Monte Carlo (SMC) simulation.

    Parameters
    ----------
    logpdf_parameterized_function : callable
        Function of the form f(x, param) returning the log-density at x for a given param.
    initial_logpdf_param : float
        Initial value of the logpdf parameter.
    target_logpdf_param : float
        Final value of the logpdf parameter.
    compute_next_logpdf_param_method : str
        Method to compute the next logpdf parameter ('p0' or 'ess').
    min_ess_ratio : float
        Minimum acceptable ratio ESS / n. Below this threshold, a restart is triggered.
    p0 : float, optional
        Prescribed probability used if compute_next_logpdf_param_method is 'p0'.
    init_box : array_like
        Domain box for particle initialization.
    n_particles : int, optional
        Number of particles. Default is 1000.
    mh_steps : int, optional
        Number of MH steps for moving the particles. Default is 20.
    smc_config : SMCConfig, optional
        An instance of SMCConfig to set SMC options.
    particles_config : ParticlesSetConfig, optional
        An instance of ParticlesSetConfig to set particle options.
        (Takes precedence on mh_steps setting.)
    debug : bool, optional
        If True, print debug information during execution.
    plot_particles : bool, optional
        If True, display a matrix plot of the final particle distribution.
    plot_empirical_distributions : bool, optional
        If True, display the final particle distribution.

    Returns
    -------
    particles : ndarray
        Final particle positions.
    smc : SMC
        The SMC instance containing diagnostics and logs.

    """
    # Create the SMC instance using the configuration objects. If none are provided,
    # defaults are used based on the dataclass definitions.
    if particles_config is None:
        particles_config = ParticlesSetConfig(
            resample_scheme="residual", covariance_method="normal"
        )
    if smc_config is None:
        smc_config = SMCConfig(
            compute_next_logpdf_param_method=compute_next_logpdf_param_method,
            mh_steps=mh_steps,
        )

    smc = SMC(
        box=init_box,
        n=n_particles,
        particles_config=particles_config,
        smc_config=smc_config,
    )

    # Perform the SMC step with a possible restart.
    smc.step_with_possible_restart(
        logpdf_parameterized_function,
        initial_logpdf_param,
        target_logpdf_param,
        min_ess_ratio,
        p0,
        debug=debug,
    )

    # Optionally, plot the matrix plot of the particles distribution.
    if plot_particles:
        try:
            smc.plot_particles()
        except Exception as e:
            print("Plotting failed:", e)
    # Optionally, plot the particles distribution.
    if plot_empirical_distributions:
        try:
            smc.plot_empirical_distributions()
        except Exception as e:
            print("Plotting failed:", e)

    return smc.particles.x, smc


def log_indicator_density(f, threshold, log_px, tail="lower"):
    """Return logpdf(x) = log(1_{f(x) ? threshold} * p_X(x)) where ? depends on tail."""

    def logpdf(x):
        x = gnp.asarray(x)
        fx = gnp.asarray(f(x))
        logpx = log_px(x)
        if tail == "lower":
            return gnp.where(fx < threshold, logpx, gnp.asarray(-1e100))
        elif tail == "upper":
            return gnp.where(fx > threshold, logpx, gnp.asarray(-1e100))
        else:
            raise ValueError(f"Invalid tail argument: {tail}")

    return logpdf


def run_subset_simulation(
    f,
    thresholds,
    init_box,
    log_px,
    tail="upper",
    n_particles=1000,
    mh_steps=20,
    min_acceptation=0.15,
    max_acceptation=0.30,
    resample_scheme="residual",
    debug=False,
):
    """Estimate P(f(X) ? u_T), with ? = < or >, via Subset Simulation.

    Parameters
    ----------
    f : callable
        Function from R^d to R (performance or score function).
    thresholds : list of float
        Monotonic threshold sequence: decreasing for '<', increasing for '>'.
    init_box : list of [lower_bounds, upper_bounds]
        Sampling domain for initial distribution.
    log_px : callable
        Log-density of the base distribution p_X.
    tail : str
        Either 'lower' (f < u_i) or 'upper' (f > u_i).
    Returns
    -------
    p_estimate : float
        Final estimate of P(f(X) ? u_T).
    stage_probs : list of float
        Estimated conditional probabilities p_{u_i | u_{i-1}}.
    smc : SMC
        The SMC object with diagnostics.
    """
    if tail == "lower":
        assert thresholds[0] == float(
            "inf"
        ), "First threshold must be +8 for tail='lower'."
    elif tail == "upper":
        assert thresholds[0] == float(
            "-inf"
        ), "First threshold must be -8 for tail='upper'."
    else:
        raise ValueError(f"Invalid tail: {tail}")

    # Set up configs
    particles_config = ParticlesSetConfig(
        initial_distribution_type="randunif",
        resample_scheme=resample_scheme,
    )
    smc_config = SMCConfig(
        compute_next_logpdf_param_method="p0",  # not used
        mh_steps=mh_steps,
        mh_acceptation_rate_min=min_acceptation,
        mh_acceptation_rate_max=max_acceptation,
    )

    smc = SMC(
        init_box,
        n=n_particles,
        particles_config=particles_config,
        smc_config=smc_config,
    )

    # Initialize particles
    smc.particles.particles_init(init_box, n_particles)
    smc.log_data["target_logpdf_param"] = thresholds[1]

    stage_probs = gnp.empty(len(thresholds)-1)

    for k in range(1, len(thresholds)):
        uk = thresholds[k]
        uk_prev = thresholds[k - 1]
        if debug:
            print(f"\n[Stage {k}] Threshold u_k = {uk:.2f}")

        # Construct logpdf for current level
        logpdf_k = log_indicator_density(f, uk, log_px, tail=tail)
        smc.particles.set_logpdf(logpdf_k)

        # Reweight
        smc.particles.reweight()

        # Compute conditional probability p_{u_k | u_{k-1}}
        w_sum = gnp.sum(smc.particles.w)
        stage_probs[k-1] = w_sum

        if debug:
            print(f"    p_{{{uk:.2f} | {uk_prev:.2f}}} - {w_sum:.2f}")

        # Normalize weights
        smc.particles.w = smc.particles.w / w_sum

        # Resample and MH move
        smc.particles.resample(debug=debug)
        smc.move_with_controlled_acceptation_rate(debug=debug)
        for _ in range(mh_steps - 1):
            smc.particles.move()

        smc.stage += 1
        smc.log_snapshot()

    # Final estimate of tail probability
    p_estimate = float(gnp.prod(stage_probs))
    return p_estimate, stage_probs, smc


def test_run_smc_sampling_gaussian_mixture():
    import matplotlib.pyplot as plt
    from scipy import stats

    # Gaussian mixture parameters
    m1, s1, w1 = 0.0, 0.04, 0.3
    m2, s2, w2 = 1.0, 0.1, 0.7

    # Tempered log pdf: log p_T(x) = beta * log(p(x))
    def logpdf_mixture(x, beta):
        x = gnp.asarray(x)
        p = w1 * stats.norm.pdf(x, loc=m1, scale=s1) + w2 * stats.norm.pdf(
            x, loc=m2, scale=s2
        )
        p = gnp.maximum(p, 1e-300)
        return beta * gnp.log(p)

    # Domain: 1D in [-1, 2]
    init_box = [[-1], [2]]
    initial_logpdf_param = 0.01  # initial beta
    target_logpdf_param = 1.0  # target beta

    # SMC settings
    compute_next_logpdf_param_method = "ess"
    min_ess_ratio = 0.5

    # Run SMC sampling
    particles, smc_instance = run_smc_sampling(
        logpdf_mixture,
        initial_logpdf_param,
        target_logpdf_param,
        compute_next_logpdf_param_method,
        min_ess_ratio,
        init_box=init_box,
        n_particles=2000,
        debug=True,
        plot_particles=False,
        plot_empirical_distributions=False,
    )

    smc_instance.plot_empirical_distributions(parameter_indices_pooled=[0])

    print("Final particles shape:", particles.shape)
    print("Sample mean:", gnp.mean(particles))
    print("Sample variance:", gnp.var(particles))

    # Plot target density and histogram of particles
    x_vals = gnp.linspace(-0.5, 1.5, 600)
    target_density = lambda x_vals: w1 * stats.norm.pdf(
        x_vals, loc=m1, scale=s1
    ) + w2 * stats.norm.pdf(x_vals, loc=m2, scale=s2)
    plt.figure(figsize=(8, 3))
    plt.hist(particles, bins=100, density=True, histtype="step", label="SMC particles")
    plt.plot(x_vals, target_density(x_vals), "r--", label="Target density")
    #  plt.plot(particles, target_density(particles), 'b.')
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    smc_instance.plot_state()


def test_subset_sampling_gaussian_icdf():
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import numpy as np

    # Define f(x) = inverse CDF of standard normal
    def f(x):
        return norm.ppf(x[:, 0])

    # Quantile levels and corresponding thresholds
    q_levels = [0.0, 0.5, 0.9, 0.97, 0.99, 1 - 1e-3, 1 - 1e-4, 1 - 1e-5, 1 - 1e-6]
    thresholds = [float("-inf")] + list(norm.ppf(q_levels[1:]))

    # Domain: Uniform[0, 1]
    box = [[0.0], [1.0]]

    def log_px(x):
        inside = gnp.all((x >= 0.0) & (x <= 1.0), axis=1)
        return gnp.where(inside, 0.0, -1e100)

    # Run subset simulation
    p_hat, stage_probs, smc = run_subset_simulation(
        f=f,
        thresholds=thresholds,
        init_box=box,
        log_px=log_px,
        tail="upper",
        n_particles=10000,
        debug=True,
    )

    # Exact conditional and sequential probabilities
    exact_conditional_probs = []
    exact_sequential_probs = []
    for i in range(1, len(q_levels)):
        q_i = q_levels[i]
        q_prev = q_levels[i - 1]
        exact_conditional_probs.append((1 - q_i) / (1 - q_prev))
        exact_sequential_probs.append(1 - q_i)

    # Estimated sequential probabilities
    estimated_sequential_probs = np.cumprod(stage_probs)

    # Print results
    print("\nThresholds:                 ", [f"{u:.2f}" for u in thresholds[1:]])
    print("Estimated conditional probs:", [f"{p:.2e}" for p in stage_probs])
    print("Exact conditional probs:    ", [f"{p:.2e}" for p in exact_conditional_probs])
    print(
          "Estimated sequential probs: ", [f"{p:.2e}" for p in estimated_sequential_probs]
    )
    print("Exact sequential probs:     ", [f"{p:.2e}" for p in exact_sequential_probs])
    print(f"Estimated final probability: {p_hat:.3e}")
    print(f"Exact final probability:     {exact_sequential_probs[-1]:.3e}")

    # Plot conditional probabilities
    stages = list(range(1, len(thresholds)))
    plt.figure(figsize=(6, 4))
    plt.plot(stages, stage_probs, marker="o", label="Estimated p_{u_i | u_{i-1}}")
    plt.plot(
        stages,
        exact_conditional_probs,
        marker="x",
        linestyle="--",
        label="Exact p_{u_i | u_{i-1}}",
    )
    plt.xlabel("Stage (i)")
    plt.ylabel("p")
    plt.title("Conditional probabilities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot sequential tail probabilities
    plt.figure(figsize=(6, 4))
    plt.plot(
        stages, estimated_sequential_probs, marker="o", label="Estimated P(f(X) > u_i)"
    )
    plt.plot(
        stages,
        exact_sequential_probs,
        marker="x",
        linestyle="--",
        label="Exact P(f(X) > u_i)",
    )
    plt.xlabel("Stage (i)")
    plt.ylabel("P(f(X) > u_i)")
    plt.yscale("log")
    plt.title("Sequential tail probabilities")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    # Plot the inverse CDF: zoom on upper tail using 1 - x (log scale)
    x_vals = np.linspace(1e-8, 1 - 1e-8, 1000)
    y_vals = norm.ppf(x_vals)
    x_tail = 1 - x_vals  # Tail probability

    plt.figure(figsize=(6, 4))
    plt.semilogx(x_tail, y_vals, label=r"$f(x) = \Phi^{-1}(x)$")

    for i, q in enumerate(q_levels[1:], start=1):
        x_q = q
        y_q = thresholds[i]
        x_tail_q = 1 - x_q
        plt.axhline(y=y_q, color="gray", linestyle="dotted")
        plt.plot([x_tail_q], [y_q], "ro")
        plt.text(
            x_tail_q, y_q, f"$1-q={1 - q:.3g}$", fontsize=8, va="bottom", ha="right"
        )

    plt.xlabel(r"$1 - x$ (tail probability)")
    plt.ylabel(r"$\Phi^{-1}(x)$")
    plt.title("Zoom on right tail of standard gaussian inverse cdf")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Sample a Gaussian mixture")
    test_run_smc_sampling_gaussian_mixture()
    print("Subset sampling")
    test_subset_sampling_gaussian_icdf()
