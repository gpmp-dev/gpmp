"""Sequential Monte Carlo (SMC) sampler implementation

This module provides the `ParticlesSet` and `SMC` classes for SMC
simulations, along with a `run_smc_sampling` function to execute full
sampling workflows. A test example using a 1D Gaussian mixture
illustrates the functionality.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Contributions:
    Julien Bect <julien.bect@centralesupelec.fr>, 2024
    Sébastien Petit, 2024
Copyright (c) 2023, 2025 CentraleSupelec
License: GPLv3 (see LICENSE)

"""

import time, warnings
from numpy.random import default_rng
import scipy.stats as stats
from scipy.optimize import brentq
import gpmp.num as gnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
    A class representing a set of particles for Sequential Monte
    Carlo (SMC) simulation.

    This class provides elementary operations for initializing,
    reweighting, resampling, and moving particles.

    Parameters
    ----------
    box : array_like
        The domain box in which the particles are initialized.
    n : int, optional, default: 1000
        Number of particles.
    initial_distribution_type: str, optional, default: "randunif"
        Initial distribution for the particles.
    rng : numpy.random.Generator
        Random number generator.

    Attributes
    ----------
    n : int
        Number of particles.
    x : ndarray
        Current positions of the particles.
    logpx : ndarray
        Log-probabilities of the particles at their current positions.
    w : ndarray
        Weights of the particles.
    logpdf_function : callable
        Function to compute the log-probability density.
    param_s : float
        Scaling parameter for the perturbation step.
    resample_scheme : str
        Scheme for resampling ('multinomial' or 'residual').
    rng : numpy.random.Generator
        Random number generator.

    Methods
    -------
    particles_init(box, n)
        Initialize particles within the given box.
    set_logpdf(logpdf_function)
        Set the log-probability density function.
    reweight()
        Reweight the particles based on the log-probability density function.
    ess()
        Calculate the effective sample size (ESS) of the particles.
    resample()
        Resample the particles based on their weights.
    multinomial_resample()
        Resample using multinomial resampling.
    residual_resample()
        Resample using residual resampling.
    perturb()
        Perturb the particles by adding random noise.
    move()
        Perform a Metropolis-Hastings step and compute the acceptation rate.

    """

    def __init__(
        self, box, n=1000, initial_distribution_type="randunif", rng=default_rng()
    ):
        """
        Initialize the ParticlesSet instance.
        """
        self.n = n  # Number of particles
        self.dim = len(box[0])
        self.logpdf_function = None
        self.rng = rng

        # Dictionary to hold parameters for the particle set
        self.particles_set_params = {
            "initial_distribution_type": initial_distribution_type,
            "resample_scheme": "multinomial",
            "param_s_initial_value": 0.5,  # Initial scaling parameter for MH perturbation
            "param_s_upper_bound": 10**4,
            "param_s_lower_bound": 10 ** (-3),
            # Jitter added to pertubation covariance matrix when it's not PSD
            "jitter_initial_value": 1e-16,
            "jitter_max_iterations": 10,
        }
        self.param_s = self.particles_set_params["param_s_initial_value"]
        self.resample_scheme = self.particles_set_params["resample_scheme"]

        # Initialize the particles.  Returns a tuple containing the
        # positions, log-probabilities, and weights of the particles
        self.x = None
        self.logpx = None
        self.w = None
        self.w_pre = None
        self.particles_init(
            box, n, method=self.particles_set_params["initial_distribution_type"]
        )

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
        self.w = gnp.full((n,), 1 / n)
        self.w_pre = gnp.full((n,), 1 / n)  # pre-update values of self.w

    def set_logpdf(self, logpdf_function):
        """
        Set the log-probability density function for the particles.

        Parameters
        ----------
        logpdf_function : callable
            Computes the log-probability density at given positions.
        """
        self.logpdf_function = logpdf_function

    def reweight(self, update_logpx_and_w_pre=True):
        logpx_new = self.logpdf_function(self.x)
        self.w = self.w_pre * gnp.exp(logpx_new.reshape(-1) - self.logpx.reshape(-1))
        if update_logpx_and_w_pre:
            self.logpx = logpx_new
            self.w_pre = gnp.copy(self.w)

    def ess(self):
        """https://en.wikipedia.org/wiki/Effective_sample_size"""
        normalization = gnp.sum(self.w**2)
        if normalization == 0.0:
            return 0.0
        else:
            return gnp.sum(self.w) ** 2 / normalization

    def resample(self, debug=False):
        """
        Resample the particles based on the chosen resampling scheme.

        The resample method routes to either multinomial_resample or
        residual_resample according to self.resample_scheme.
        """
        if self.resample_scheme == "multinomial":
            self.multinomial_resample(debug=debug)
        elif self.resample_scheme == "residual":
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
        logpx_resampled = gnp.empty(self.logpx.shape)
        normalization = gnp.sum(self.w)
        if normalization == 0.0:
            p = gnp.full((self.n,), 1 / self.n)
        else:
            p = self.w / gnp.sum(self.w)
        try:
            counts = self.multinomial_rvs(self.n, p, self.rng)
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
                x_resampled = gnp.set_row_2d(x_resampled, i, self.x[j, :])
                logpx_resampled = gnp.set_elem_1d(logpx_resampled, i, self.logpx[j])
                counts = gnp.set_elem_1d(counts, j, counts[j] - 1)
                i += 1
            j += 1

        self.x = x_resampled
        self.logpx = logpx_resampled
        self.w = gnp.full((self.n,), 1 / self.n)
        self.w_pre = gnp.full((self.n,), 1 / self.n)

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
        normalization = gnp.sum(self.w)
        if normalization == 0.0:
            p = gnp.full((self.n,), 1 / self.n)
        else:
            p = self.w / gnp.sum(self.w)

        # Deterministic assignment: floor of expected counts
        counts_det = gnp.asint(gnp.floor(N * p))
        N_det = int(gnp.sum(counts_det))

        # Compute residuals
        residuals = N * p - counts_det
        N_residual = N - N_det

        # Multinomial step on residuals
        if N_residual > 0:
            try:
                p_vals = residuals / N_residual

                counts_res = self.multinomial_rvs(
                    N_residual, residuals / N_residual, self.rng
                )
            except Exception:
                extype, value, tb = __import__("sys").exc_info()
                __import__("traceback").print_exc()
                __import__("pdb").post_mortem(tb)
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
                x_resampled = gnp.set_row_2d(x_resampled, i, self.x[j, :])
                logpx_resampled = gnp.set_elem_1d(logpx_resampled, i, self.logpx[j])
                counts = gnp.set_elem_1d(counts, j, counts[j] - 1)
                i += 1
            j += 1

        self.x = x_resampled
        self.logpx = logpx_resampled
        self.w = gnp.full((self.n,), 1 / self.n)
        self.w_pre = gnp.full((self.n,), 1 / self.n)

    def perturb(self):
        """Perturb the particles by adding Gaussian noise.

        This method perturbs the current positions of the particles by
        applying a Gaussian random perturbation. The covariance matrix
        is computed from the particles' current positions and is
        scaled by the perturbation parameter `param_s`. This
        covariance matrix defines the spread of the Gaussian noise
        used to move the particles.

        """

        param_s_lower = self.particles_set_params["param_s_lower_bound"]
        param_s_upper = self.particles_set_params["param_s_upper_bound"]

        # Check if param_s is within bounds
        if self.param_s > param_s_upper or self.param_s < param_s_lower:
            raise ParticlesSetError(self.param_s, param_s_lower, param_s_upper)

        # Covariance matrix of the pertubation noise
        C = self.param_s * gnp.cov(self.x.reshape(self.n, -1).T)

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
                self.particles_set_params["jitter_max_iterations"]
            ):  # Try iterations of jittering
                jitter = self.particles_set_params["jitter_initial_value"] * (10**i)
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
                + f"{self.particles_set_params['jitter_max_iterations']} jittering attempts. "
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
        logpy = self.logpdf_function(y)

        # Compute acceptation probabilities
        rho = gnp.minimum(1, gnp.exp(logpy - self.logpx))

        accepted_moves = 0  # Counter for accepted moves
        for i in range(self.n):
            if ParticlesSet.rand(self.rng) < rho[i]:
                # Update the particle position and log probability if the move is accepted
                self.x = gnp.set_row_2d(self.x, i, y[i, :])
                self.logpx = gnp.set_elem_1d(self.logpx, i, logpy[i])
                accepted_moves += 1

        # Compute the acceptation rate
        acceptation_rate = accepted_moves / self.n

        return acceptation_rate

    @staticmethod
    def rand(rng):
        return rng.uniform()

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
        return gnp.asarray(stats.qmc.scale(rng.uniform(size=(n, dim)), box[0], box[1]))


class SMC:
    """Sequential Monte Carlo (SMC) sampler class.

    This class drives the SMC process using a set of particles,
    employing a strategy as described in
    Bect, J., Li, L., & Vazquez, E. (2017). "Bayesian subset simulation",
    SIAM/ASA Journal on Uncertainty Quantification, 5(1), 762-786.
    Available at: https://arxiv.org/abs/1601.02557

    Parameters
    ----------
    box : array_like
        The domain box for particle initialization.
    n : int, optional, default: 1000
        Number of particles.
    initial_distribution_type: str, optional, default: "randunif"
        Initial distribution for the particles.
    rng : numpy.random.Generator
        Random number generator.

    Attributes
    ----------
    box : array_like
        The domain box for particle initialization.
    n : int
        Number of particles.
    particles : ParticlesSet
        Instance of ParticlesSet class to manage the particles.

    Methods
    -------
    step(logpdf_parameterized_function, logpdf_param)
        Perform a single SMC step.
    move_with_controlled_acceptation_rate()
        Adjust the particles' movement to control the acceptation rate.

    """

    def __init__(
        self,
        box,
        n=2000,
        initial_distribution_type="randunif",
        compute_next_logpdf_param_method="p0",
        rng=default_rng(),
    ):
        """
        Initialize the SMC sampler.
        """
        self.box = box
        self.n = n
        self.initial_distribution_type = initial_distribution_type
        self.particles = ParticlesSet(box, n, initial_distribution_type, rng)

        # Dictionary to hold MH algorithm parameters
        self.mh_params = {
            "mh_steps": 20,
            "acceptation_rate_min": 0.2,
            "acceptation_rate_max": 0.4,
            "adjustment_factor": 1.4,
            "adjustment_max_iterations": 50,
        }

        # Next param method
        self.compute_next_logpdf_param_method = compute_next_logpdf_param_method
        if self.compute_next_logpdf_param_method == "p0":
            self.compute_next_logpdf_param = self.compute_next_logpdf_param_p0
        elif self.compute_next_logpdf_param_method == "ess":
            self.compute_next_logpdf_param = self.compute_next_logpdf_param_ess
        else:
            raise ValueError("compute_next_logpdf_param_method must be 'ess' or 'p0'.")
        # Logging
        self.log = []  # Store the state logs
        self.stage = 0
        self.logging_current_ess = None
        self.logging_current_logpdf_param = None
        self.logging_target_logpdf_param = None
        self.logging_restart_iteration = 0
        self.logging_logpdf_param_sequence = []  # Sequence of logpdf_params in restart
        self.logging_acceptation_rate_sequence = []

    def _log_data(
        self,
        logpdf_param=None,
        ess=None,
        acceptation_rate=None,
        log_current_state_and_reinitialize=False,
    ):
        """
        Helper function to log data during the SMC process. It logs both incremental data
        like logpdf_param, ESS, and acceptation rate, as well as the full state at the end of a stage.

        Parameters
        ----------
        logpdf_param : float, optional
            The current logpdf parameter value being used in the SMC step.
        ess : float, optional
            Effective sample size (ESS) of the current particle set.
        acceptation_rate : float, optional
            Acceptation rate of the particle move step.
        log_current_state_and_reinitialize : bool, optional
            If True, logs the full current state of the SMC process.
        """
        # Incremental data logging
        if logpdf_param is not None:
            self.logging_current_logpdf_param = logpdf_param
        if ess is not None:
            self.logging_current_ess = ess
        if acceptation_rate is not None:
            self.logging_acceptation_rate_sequence.append(acceptation_rate)

        # If log_current_state_and_reinitialize is True, log the full state
        if log_current_state_and_reinitialize:
            state = {
                "timestamp": time.time(),
                "stage": self.stage,
                "num_particles": self.n,
                "current_scaling_param": self.particles.param_s,
                "target_logpdf_param": self.logging_target_logpdf_param,
                "current_logpdf_param": self.logging_current_logpdf_param,
                "ess": self.logging_current_ess,
                "restart_iteration": self.logging_restart_iteration,
                "logpdf_param_sequence": self.logging_logpdf_param_sequence.copy(),
                "acceptation_rate_sequence": self.logging_acceptation_rate_sequence.copy(),
            }
            self.log.append(state)
            # Reinitialize acceptation_rate_sequence for the next stage
            self.logging_acceptation_rate_sequence = []

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

        # Set target density
        def logpdf(x):
            return logpdf_parameterized_function(x, logpdf_param)

        self.particles.set_logpdf(logpdf)

        # Reweight
        self.particles.reweight()

        # Log current logpdf_param and ESS
        self._log_data(logpdf_param=logpdf_param, ess=self.particles.ess())

        # Resample / move
        self.particles.resample(debug)
        self.move_with_controlled_acceptation_rate()

        for _ in range(self.mh_params["mh_steps"] - 1):
            # Additional moves if required
            acceptation_rate = self.particles.move()
            self._log_data(acceptation_rate=acceptation_rate)

        # Log state at the end of the step
        self._log_data(log_current_state_and_reinitialize=True)

        # Debug plot, if needed
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

        This method checks if the effective sample size (ESS) falls
        below a specified ratio, and if so, initiates a restart. The
        restart process reinitializes particles and recalculates
        logpdf_params to better target the desired distribution.

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
        self.logging_current_logpdf_param = target_logpdf_param
        self.logging_target_logpdf_param = target_logpdf_param

        # Set logpdf with target param
        def logpdf(x):
            return logpdf_parameterized_function(x, target_logpdf_param)

        self.particles.set_logpdf(logpdf)

        # reweight and compute ess
        self.particles.reweight(update_logpx_and_w_pre=False)
        ess = self.particles.ess()
        self.logging_current_ess = ess

        # restart?
        if ess / self.n < min_ess_ratio:
            self.restart(
                logpdf_parameterized_function,
                initial_logpdf_param,
                target_logpdf_param,
                min_ess_ratio,
                p0,
                debug=debug,
            )
            # Note: Logging will occur inside the restart method.

        else:  # Normal step
            # Reweight and update self.logpx
            self.particles.reweight(update_logpx_and_w_pre=True)
            # Resample
            self.particles.resample()
            # Move with control on acceptation rate
            self.move_with_controlled_acceptation_rate()
            # Additional moves if required
            for _ in range(self.mh_params["mh_steps"] - 1):
                acceptation_rate = self.particles.move()
                self.logging_acceptation_rate_sequence.append(acceptation_rate)
            # Logging
            self._log_data(log_current_state_and_reinitialize=True)

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
        p0 : float
            Prescribed probability
        debug : bool
            If True, print debug information.
        """
        if debug:
            print("---- (Re)starting SMC from initial parameter ----")

        self._log_data(log_current_state_and_reinitialize=True)

        if self.compute_next_logpdf_param_method == "p0":
            threshold = p0
        elif self.compute_next_logpdf_param_method == "ess":
            threshold = min_ess_ratio

        self.particles.particles_init(
            self.box, self.n, method=self.initial_distribution_type
        )

        # Is the initial threshold too difficult?
        def logpdf(x):
            return logpdf_parameterized_function(x, initial_logpdf_param)

        self.particles.set_logpdf(logpdf)
        self.particles.reweight(update_logpx_and_w_pre=False)
        ess = self.particles.ess()
        if ess / self.n < min_ess_ratio:
            warnings.warn(
                f"ESS ratio {ess / self.n} below threshold={min_ess_ratio} at initialization.",
                RuntimeWarning,
            )

        current_logpdf_param = initial_logpdf_param

        self.logging_logpdf_param_sequence = [initial_logpdf_param]

        while current_logpdf_param != target_logpdf_param:
            next_logpdf_param = self.compute_next_logpdf_param(
                logpdf_parameterized_function,
                current_logpdf_param,
                target_logpdf_param,
                threshold,
                debug=debug,
            )

            self.logging_restart_iteration += 1
            self.logging_logpdf_param_sequence.append(next_logpdf_param)

            self.step(logpdf_parameterized_function, next_logpdf_param, debug=debug)

            current_logpdf_param = next_logpdf_param

        # Logging reinitialization
        self.logging_logpdf_param_sequence = []
        self.logging_restart_iteration = 0

    def move_with_controlled_acceptation_rate(self, debug=False):
        """
        Adjust the particles' movement to maintain the acceptation
        rate within specified bounds.  This method dynamically adjusts
        the scaling parameter based on the acceptation rate to ensure
        efficient exploration of the state space.

        """
        iteration_counter = 0
        self.logging_acceptation_rate_sequence = []  # Logging
        while iteration_counter < self.mh_params["adjustment_max_iterations"]:
            iteration_counter += 1

            acceptation_rate = self.particles.move()

            # Logging
            self.logging_acceptation_rate_sequence.append(acceptation_rate)

            if debug:
                print(f"Acceptation rate = {acceptation_rate}")

            if acceptation_rate < self.mh_params["acceptation_rate_min"]:
                self.particles.param_s /= self.mh_params["adjustment_factor"]
                continue

            if acceptation_rate > self.mh_params["acceptation_rate_max"]:
                self.particles.param_s *= self.mh_params["adjustment_factor"]
                continue

            break

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
            def logpdf(x):
                return logpdf_parameterized_function(x, logpdf_param)

            self.particles.set_logpdf(logpdf)
            self.particles.reweight(update_logpx_and_w_pre=False)
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
        stage_changes = []  # To mark the stages where change occurs

        # Extracting and replicating data according to the length of 'acceptation_rate_sequence' in each log entry
        for idx, entry in enumerate(log_data):
            ar_length = len(entry["acceptation_rate_sequence"])
            if ar_length == 0:
                entry["acceptation_rate_sequence"] = [0.0]
                ar_length = 1

            stages.extend([entry["stage"]] * ar_length)
            target_logpdf_params.extend([entry["target_logpdf_param"]] * ar_length)
            current_logpdf_params.extend([entry["current_logpdf_param"]] * ar_length)
            ess_values.extend([entry["ess"]] * ar_length)
            acceptation_rates.extend(entry["acceptation_rate_sequence"])

        # Plotting
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Time")
        ax1.set_ylabel("logpdf_param", color=color)
        t, target_logpdf_params = make_stairs(target_logpdf_params)
        t, current_logpdf_params = make_stairs(current_logpdf_params)
        ax1.plot(
            t,
            target_logpdf_params,
            label="Target logpdf_param",
            color="red",
            linestyle="dashed",
        )
        ax1.plot(
            t,
            current_logpdf_params,
            label="Current logpdf_param",
            color="red",
            linestyle="solid",
        )
        (ymin, ymax) = ax1.get_ylim()
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
        t, ess_values = make_stairs(ess_values)
        ax2.plot(t, ess_values, label="ESS", color=color)
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

        # if parameter_indices_pooled is not None:
        #     pooled_vals = self.particles.x[:, parameter_indices_pooled].ravel()
        #     fig, ax = plt.subplots(figsize=(8, 3))
        #     color_cycler = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        #     colors = cycle(color_cycler)
        #     color = next(colors)
        #     ax.hist(
        #         pooled_vals,
        #         bins=bins,
        #         density=True,
        #         alpha=0.3,
        #         color=color,
        #         label="Histogram",
        #     )
        #     lo, hi = pooled_vals.min(), pooled_vals.max()
        #     xx = gnp.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 100)
        #     kde = stats.gaussian_kde(pooled_vals, bw_method='scott')
        #     ax.plot(xx, kde(xx), color=color, label="KDE")
        #     ax.set_xlabel(r"Pooled $\theta_i$")
        #     ax.set_ylabel("Density")
        #     ax.set_title("Marginal distributions")
        #     ax.legend()
        #     plt.tight_layout()
        #     plt.show()


def run_smc_sampling(
    logpdf_parameterized_function,
    initial_logpdf_param: float,
    target_logpdf_param: float,
    compute_next_logpdf_param_method,
    min_ess_ratio: float,
    p0: float = None,
    init_box: list = None,
    n_particles: int = 1000,
    debug: bool = False,
    plot_particles: bool = False,
    plot_empirical_distributions: bool = False,
):
    """
    Run a full Sequential Monte Carlo (SMC) simulation.

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
    smc = SMC(
        init_box,
        n=n_particles,
        compute_next_logpdf_param_method=compute_next_logpdf_param_method,
    )
    current_logpdf_param = initial_logpdf_param

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


def test_run_smc_sampling_gaussian_mixture():
    import matplotlib.pyplot as plt
    from scipy import stats

    # Gaussian mixture parameters
    m1, s1, w1 = 0.0, 0.05, 0.3
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
    initial_logpdf_param = 0.1  # initial beta
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
    x_vals = gnp.linspace(-1, 2, 300)
    target_density = w1 * stats.norm.pdf(
        x_vals, loc=m1, scale=s1
    ) + w2 * stats.norm.pdf(x_vals, loc=m2, scale=s2)
    plt.figure(figsize=(8, 3))
    plt.hist(particles, bins=100, density=True, histtype="step", label="SMC particles")
    plt.plot(x_vals, target_density, "r--", label="Target density")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_run_smc_sampling_gaussian_mixture()
