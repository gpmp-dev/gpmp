# gpmp/core/model.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Gaussian Process model class.
"""
import warnings
import gpmp.num as gnp

from . import kriging
from . import loo
from . import likelihood
from . import algebra
from . import fisher
from . import sample_paths as sample_paths
from . import utils


class Model:
    """Gaussian Process (GP) Model Class.

    This class implements a Gaussian Process (GP) model for function
    approximation.

    Attributes
    ----------
    mean : callable or None
        A function defining the mean of the Gaussian Process (GP),
        used when `self.meantype` is either "parameterized" or
        "linear_predictor". When `self.meantype` is "zero", this
        attribute is not used and should be set to `None`. The mean
        function is called as

        P = self.mean(x, meanparam),

        where `x` is an (n x d) array representing n data points in a
        d-dimensional space, and `meanparam` is an array of parameters
        for the mean function.
        The function returns a (n x q) matrix, where:

        - If `self.meantype` is "parameterized", `q` is 1, and the matrix
          represents the parameterized mean values at the points `x`.
        - If `self.meantype` is "linear_predictor", `q` is greater than or
          equal to 1, indicating the model uses a linear combination
          of q basis functions with linear_predictor coefficients. Each
          column of `P` corresponds to a different basis function
          evaluated at `x`. These basis functions are typically
          monomials, represented as :math:`x \\mapsto 1, x \\mapsto
          x, x \\mapsto x^2, \\ldots`.

    covariance : callable
        Returns the covariance of the GP. The function is called as

        K = self.covariance(x, y, self.covparam, pairwise),

        where x is (n x d) and y is either:
           - (m x d) array of points, or
           - None, meaning y := x (specialized “tt/ii” path).
        Pairwise indicates if an (n x m) covariance matrix
        (pairwise == False) or an (n x 1) vector (n == m, pairwise =
        True) should be returned

    meanparam : array_like, optional
        Parameter for the mean function, given as a one-dimensional
        array. This parameter is only used when meantype is set to
        'parameterized'.

    covparam : array_like, optional
        Parameter for the covariance function, given as a
        one-dimensional array. This parameter determines the
        characteristics of the covariance function, influencing
        aspects like length scale, variance, and smoothness of the GP.

    meantype : str, optional
        The type of mean used in the model. It can be:

        - 'zero': A zero mean function, implying the GP has a zero prior
          mean function. Then, self.mean is never called and should be set to None.
        - 'parameterized': A parameterized mean function with parameterized parameters. Useful
          when there's prior knowledge about the mean behavior of the
          function being modeled.
        - 'linear_predictor': A linearly parameterized mean function, corresponding to
          the case of "universal" or intrinsic kriging.

    Public API (methods)
    --------------------
    predict
        Posterior mean/variance at target points.
    loo
        Leave-one-out predictions via virtual cross-validation.
    negative_log_likelihood_zero_mean
        Negative log-likelihood with zero mean.
    negative_log_likelihood
        Negative log-likelihood with given mean.
    negative_log_restricted_likelihood
        REML criterion (linear_predictor).
    norm_k_sqrd_with_zero_mean
        RKHS norm z^T K^-1 z (zero-mean case).
    norm_k_sqrd
        RKHS norm (linear_predictor).
    k_inverses
        Returns z^T K^-1 z, K^-1 1, and K^-1 z.
    fisher_information
        Finite-difference Fisher information (SPD).
    fisher_information_cpd
        Fisher information in contrast space (CPD kernels).
    fisher_information_torch
        Fisher information via second-order differentiation.
    sample_paths
        Unconditional GP sample paths on xt.
    conditional_sample_paths
        Conditioning-by-kriging (zero/linear_predictor).
    conditional_sample_paths_parameterized_mean
        Conditioning with parameterized mean.

    Examples
    --------
    >>> import gpmp as gp
    >>> import gpmp.num as gnp
    >>> mean = lambda x, meanparam: (meanparam[0] + meanparam[1] * x)
    >>> def covariance(x, y, covparam, pairwise=False):
    ...     p = 0  # smoothness index for Matern nu=3/2
    ...     return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)
    >>> model = gp.core.Model(mean,
    ...     covariance,
    ...     meanparam=[0.5, 0.2],
    ...     covparam=[1.0, 0.1],
    ...     meantype="parameterized"
    >>> )
    >>> xi = gnp.array([0.0, 1.0, 2.0, 3.0, 5.0]).reshape(-1, 1)
    >>> zi = gnp.array([0.0, 1.2, 2.5, 4.2, 4.3])
    >>> xt = gnp.linspace(0.0, 5.0, 11).reshape(-1, 1)
    >>> zt_mean, zt_var = model.predict(xi, zi, xt)
    """

    def __init__(
        self,
        mean,
        covariance,
        meanparam=None,
        covparam=None,
        meantype="linear_predictor",
    ):
        """
        Parameters
        ----------
        mean : callable
            A function that returns the mean of the Gaussian Process (GP).
        covariance : callable
            A function that returns the covariance of the Gaussian Process (GP).
        meanparam : array_like, optional
            The parameters for the mean function, specified as a one-dimensional array of values.
        covparam : array_like, optional
            The parameters for the covariance function, specified as a one-dimensional array of values.
        meantype : str, optional
            Type of mean used in the model. It can be:
            'zero' - Zero mean function.
            'parameterized' - Known mean function with parameterized parameters.
            'linear_predictor' - Linearly parameterized mean function.
        """
        utils.validate_model_mean(meantype, mean, meanparam)
        self.meantype = meantype
        self.mean = mean
        self.meanparam = meanparam
        self.covparam = covparam
        self.covariance = covariance

    def __repr__(self):
        output = str("<gpmp.core.Model object> " + hex(id(self)))
        return output

    def __str__(self):
        if self.meantype == "zero":
            mean_desc = "Zero Mean"
        else:
            try:
                mean_desc = self.mean.__name__
            except AttributeError:
                mean_desc = str(self.mean)
        try:
            cov_desc = self.covariance.__name__
        except AttributeError:
            cov_desc = str(self.covariance)
        return (
            f"GP Model:\n"
            f"  Mean Type: {self.meantype}\n"
            f"  Mean Function: {mean_desc}\n"
            f"  Mean Parameters: {self.meanparam}\n"
            f"  Covariance Function: {cov_desc}\n"
            f"  Covariance Parameters: {self.covparam}"
        )

    # ------------------------------------------------------------------
    # Kriging predictors (delegating to gpmp.core.kriging)
    # ------------------------------------------------------------------
    def kriging_predictor_with_zero_mean(self, xi, xt, return_type=0):
        """Compute the kriging predictor with zero mean."""
        return kriging.kriging_predictor_with_zero_mean(self, xi, xt, return_type)

    def kriging_predictor(self, xi, xt, return_type=0):
        """Compute the kriging predictor with a non-zero mean.

        Parameters
        ----------
        xi : array_like, shape (n, d)
            Observation points.
        xt : array_like, shape (m, d)
            Prediction points.
        return_type : int, optional
            Indicator for posterior variance:
              -1: return None,
               0: return variance (default),
               1: return full covariance.

        Returns
        -------
        lambda_t : array_like, shape (n, m)
            Kriging weights.
        zt_posterior_variance : array_like
            Posterior variance (or covariance if return_type==1).
        """
        return kriging.kriging_predictor(self, xi, xt, return_type)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        xi,
        zi,
        xt,
        return_lambdas=False,
        zero_neg_variances=True,
        convert_in=True,
        convert_out=True,
    ):
        """Performs a prediction at target points xt given the data (xi, zi).

        Parameters
        ----------
        xi : ndarray or gnp.array of shape (ni, dim)
            Observation points in input space.
        zi : ndarray or gnp.array of shape (ni,) or (ni, 1)
            Observed values at the observation points.
        xt : ndarray or gnp.array of shape (nt, dim)
            Target points where predictions are to be made.
        return_lambdas : bool, optional
            Whether to return the kriging weights, by default False.
        zero_neg_variances : bool, optional
            Whether to replace negative posterior variances with zeros, by default True.
            Negative variances can occur due to numerical errors.
        convert_in : bool, optional
            Whether to convert input arrays to the backend's array type, by default True.
        convert_out : bool, optional
            Whether to convert output arrays to numpy arrays, by default True.

        Returns
        -------
        z_posterior_mean : gnp.array or ndarray of shape (nt,)
            Posterior mean predictions at target points.
        z_posterior_variance : gnp.array or ndarray of shape (nt,)
            Posterior variance estimates at target points.
        lambda_t : gnp.array or ndarray of shape (ni, nt), optional
            Kriging weights, only returned if return_lambdas=True.

        Notes
        -----
        From a Bayesian point of view, the outputs are respectively
        the posterior mean and variance of the Gaussian process given
        the data (xi, zi).

        Treatment of the mean based on 'meantype':
        1. "zero": The function uses the kriging predictor with zero mean.
        2. "linear_predictor": Uses the general / intrinsic kriging predictor.
        3. "parameterized": The zero-mean kriging predictor is used after
           centering zi around the parameterized mean. The mean is then added
           back to the posterior mean prediction. 'meanparam' should
           be provided for this type.
        """
        # Step 1: Prepare the data.
        xi, zi, xt = utils.ensure_shapes_and_type(
            xi=xi, zi=zi, xt=xt, convert=convert_in
        )
        # Step 2: Select the kriging predictor and adjust for mean.
        zi_centered, zt_prior_mean, lambda_t, zt_posterior_variance = (
            kriging.select_predictor(self, xi, zi, xt)
        )
        # Step 3: Postprocessing: ensure nonnegative variances.
        if gnp.any(zt_posterior_variance < 0.0):
            warnings.warn(
                "Negative variances detected. Consider using jitter.",
                RuntimeWarning,
            )
        if zero_neg_variances:
            zt_posterior_variance = gnp.maximum(zt_posterior_variance, 0.0)
        # Step 4: Compute the posterior mean.
        zt_posterior_mean = (
            gnp.einsum("i..., i...", lambda_t, zi_centered) + zt_prior_mean
        )
        # Optional: convert outputs to numpy arrays.
        if convert_out:
            zt_posterior_mean = gnp.to_np(zt_posterior_mean)
            zt_posterior_variance = gnp.to_np(zt_posterior_variance)
        if return_lambdas:
            return (zt_posterior_mean, zt_posterior_variance, lambda_t)
        return (zt_posterior_mean, zt_posterior_variance)

    def loo(self, xi, zi, convert_in=True, convert_out=False):
        """Compute the leave-one-out (LOO) prediction error.

        This method computes the LOO prediction error using the
        "virtual cross-validation" formula, which allows for
        computation of LOO predictions without looping on the data points.

        Parameters
        ----------
        xi : array_like, shape (n, d)
            Input data points used for fitting the GP model, where n
            is the number of points and d is the dimensionality.
        zi : array_like, shape (n, ) or (n, 1)
            Output (response) values corresponding to the input data points xi.
        convert_in : bool, optional
            Whether to convert input arrays to _gpmp_backend_ type or keep as-is.
        convert_out : bool, optional
            Whether to return numpy arrays or keep _gpmp_backend_ types.

        Returns
        -------
        zloo : array_like, shape (n, )
            Leave-one-out predictions for each data point in xi.
        sigma2loo : array_like, shape (n, )
            Variance of the leave-one-out predictions.
        eloo : array_like, shape (n, )
            Leave-one-out prediction errors for each data point in xi.
        """
        xi_, zi_, _ = utils.ensure_shapes_and_type(xi=xi, zi=zi, convert=convert_in)
        zloo, sigma2loo, eloo = loo.loo(self, xi_, zi_)
        if convert_out:
            zloo = gnp.to_np(zloo)
            sigma2loo = gnp.to_np(sigma2loo)
            eloo = gnp.to_np(eloo)
        return zloo, sigma2loo, eloo

    # ------------------------------------------------------------------
    # Likelihoods and algebraic norms (delegating to gpmp.core.likelihood/algebra)
    # ------------------------------------------------------------------
    def negative_log_likelihood_zero_mean(self, covparam, xi, zi):
        """Computes the negative log-likelihood of the Gaussian process model with zero
        mean.

        This function is specific to the zero-mean case, and the negative log-likelihood
        is computed based on the provided covariance function and parameters.

        Parameters
        ----------
        covparam : gnp.array
            Parameters for the covariance function. This array contains the hyperparameters
            required by the chosen covariance function.
        xi : ndarray(ni,d)
            Locations of the data points in the input space, where ni is the number of data points
            and d is the dimensionality of the input space.
        zi : ndarray(ni, )
            Observed values corresponding to each data point in xi.

        Returns
        -------
        nll : scalar
            Negative log-likelihood of the observed data given the model and covariance parameters.
        """
        return likelihood.negative_log_likelihood_zero_mean(self, covparam, xi, zi)

    def negative_log_likelihood(self, meanparam, covparam, xi, zi):
        """Computes the negative log-likelihood of the Gaussian process model with a
        given mean.

        This function computes the negative log-likelihood based on
        the provided mean function, covariance function, and their
        parameters.

        Parameters
        ----------
        meanparam : gnp.array
            Parameters for the mean function. This array contains the
            hyperparameters required by the chosen mean function.
        covparam : gnp.array
            Parameters for the covariance function. This array
            contains the hyperparameters required by the chosen
            covariance function.
        xi : ndarray(ni,d)
            Locations of the data points in the input space, where ni
            is the number of points and d is the dimensionality
            of the input space.
        zi : ndarray(ni, )
            Observed values corresponding to each data point in xi.

        Returns
        -------
        nll : scalar
            Negative log-likelihood of the observed data given the
            model, mean, and covariance parameters.
        """
        return likelihood.negative_log_likelihood(self, meanparam, covparam, xi, zi)

    def negative_log_restricted_likelihood(self, covparam, xi, zi):
        """Compute the negative log-restricted likelihood of the GP model.

        This method calculates the negative log-restricted likelihood,
        which is used for parameter estimation in the Gaussian Process
        model with a mean of type "linear predictor"

        Parameters
        ----------
        covparam : gnp.array
            Covariance parameters for the Gaussian Process.
        xi : array_like, shape (n, d)
            Input data points used for fitting the GP model, where n
            is the number of points and d is the dimensionality.
        zi : array_like, shape (n, )
            Output (response) values corresponding to the input data points xi.

        Returns
        -------
        L : float
            Negative log-restricted likelihood value.
        """
        return likelihood.negative_log_restricted_likelihood(self, covparam, xi, zi)

    def norm_k_sqrd_with_zero_mean(self, xi, zi, covparam):
        """Compute the squared norm of the residual vector with zero mean.

        This method calculates the squared norm of the residual vector
        (zi - mean(xi)) using the inverse of the covariance matrix K.

        Parameters
        ----------
        xi : array_like, shape (n, d)
            Input data points used for fitting the GP model, where n
            is the number of points and d is the dimensionality.
        zi : array_like, shape (n, )
            Output (response) values corresponding to the input data points xi.
        covparam : array_like
            Covariance parameters for the Gaussian Process.

        Returns
        -------
        norm_sqrd : float
            Squared norm of the residual vector.
        """
        return algebra.norm_k_sqrd_with_zero_mean(self, xi, zi, covparam)

    def k_inverses(self, xi, zi, covparam):
        """Compute various quantities involving the inverse of K.

        Specifically, this method calculates:
        - z^T K^-1 z
        - K^-1 1
        - K^-1 z

        Parameters
        ----------
        xi : array_like, shape (n, d)
            Input data points used for fitting the GP model.
        zi : array_like, shape (n, 1)
            Output (response) values corresponding to the input data points xi.
        covparam : array_like
            Covariance parameters for the Gaussian process.

        Returns
        -------
        zTKinvz : float
            z^T K^-1 z
        Kinv1 : array_like, shape (n, 1)
            K^-1 1
        Kinvz : array_like, shape (n, 1)
            K^-1 z
        """
        return algebra.k_inverses(self, xi, zi, covparam)

    def norm_k_sqrd(self, xi, zi, covparam):
        """Compute the squared norm of the residual vector after applying the contrast
        matrix W.

        This method calculates the squared norm of the residual vector
        (Wz) using the inverse of the covariance matrix (WKW), where W
        is a matrix of contrasts.

        Parameters
        ----------
        xi : ndarray, shape (ni, d)
            Input data points used for fitting the GP model, where ni
            is the number of points and d is the dimensionality.
        zi : ndarray, shape (ni, 1) or (ni, )
            Output (response) values corresponding to the input data points xi.
        covparam : array_like
            Covariance parameters for the Gaussian Process.

        Returns
        -------
        float
            The squared norm of the residual vector after applying the
            contrast matrix W: (Wz)' (WKW)^-1 Wz.
        """
        return algebra.norm_k_sqrd(self, xi, zi, covparam)

    # ------------------------------------------------------------------
    # Fisher information (delegating to gpmp.core.fisher)
    # ------------------------------------------------------------------
    def fisher_information(self, xi, covparam=None, epsilon=1e-3):
        """Compute the Fisher information matrix using
        finite-difference numerical differentiation (with respect to
        the covariance parameters).

        Parameters
        ----------
        xi : ndarray (n, d)
            Observation points where n is the number of points and d is the dimensionality.
        covparam : ndarray, optional
            Covariance parameters at which to compute Fisher information.
            If None, uses self.covparam.
        epsilon : float, optional
            Step size for finite-difference approximation. Default is 1e-3.

        Returns
        -------
        fisher_info : ndarray (p, p)
            Fisher Information matrix, where p is the number of covariance parameters.

        Notes
        -----
        1) The Fisher information for a zero-mean Gaussian vector with covariance K(θ) is given by:
             I_{ij}(θ) = 0.5 * Tr(K^{-1}(θ) * ∂K(θ)/∂θ_i * K^{-1}(θ) * ∂K(θ)/∂θ_j).
        2) dK/dθ_i is computed via central finite differences:
             dK_i = [K(θ + e_i) - K(θ - e_i)] / (2 * epsilon).
        3) This method can be computationally expensive for high-dimensional parameter spaces,
           but is robust when autodiff faces numerical instabilities.
        """
        return fisher.fisher_information(self, xi, covparam=covparam, epsilon=epsilon)

    def fisher_information_cpd(self, xi, covparam=None, epsilon=1e-3):
        """
        Fisher information for covariance parameters with CPD kernels.

        If the mean is of type "linear_predictor", the information is computed
        in contrast space with G = Wᵀ K W, where W spans Null(Pᵀ). Otherwise,
        the standard SPD formula with K is used.

        I_ij = 0.5 * Tr( M^{-1} ∂M/∂θ_i M^{-1} ∂M/∂θ_j ),
        with M = G (CPD case) or M = K (SPD case).

        Parameters
        ----------
        xi : (n, d) array
            Design points.
        covparam : (p,) array, optional
            Covariance parameters (default: self.covparam).
        epsilon : float, optional
            Step for central finite differences (default: 1e-3).

        Returns
        -------
        I : (p, p) array
            Fisher information matrix.
        """
        return fisher.fisher_information_cpd(
            self, xi, covparam=covparam, epsilon=epsilon
        )

    def fisher_information_torch(self, xi, covparam):
        """Compute Fisher Information matrix using second-order differentiation."""
        return fisher.fisher_information_torch(self, xi, covparam)

    # ------------------------------------------------------------------
    # Sampling (delegating to gpmp.core.sample_paths)
    # ------------------------------------------------------------------
    def sample_paths(self, xt, nb_paths, method="chol", check_result=True):
        """Generates nb_paths sample paths on xt from the zero-mean GP model GP(0, k),
        where k is the covariance specified by Model.covariance.

        Parameters
        ----------
        xt : ndarray, shape (nt, d)
            Input data points where the sample paths are to be
            generated, where nt is the number of points and d is the
            dimensionality.
        nb_paths : int
            Number of sample paths to generate.
        method : str, optional, default: 'chol'
            Method used for the factorization of the covariance
            matrix. Options are 'chol' for Cholesky decomposition and
            'svd' for singular value decomposition.
        check_result : bool, optional, default: True
            If True, checks if the Cholesky factorization is successful.

        Returns
        -------
        ndarray, shape (nt, nb_paths)
            Array containing the generated sample paths at the input points xt.
        """
        return sample_paths.sample_paths(
            self, xt, nb_paths, method=method, check_result=check_result
        )

    def conditional_sample_paths(
        self, ztsim, xi_ind, zi, xt_ind, lambda_t, convert_out=True
    ):
        """Generates conditional sample paths on xt from unconditional sample paths
        ztsim, using the matrix of kriging weights lambda_t, which is provided by
        kriging_predictor() or predict().

        Conditioning is done with respect to ni observations, located
        at the indices given by xi_ind in ztsim, with corresponding
        observed values zi. xt_ind specifies indices in ztsim
        corresponding to conditional simulation points.

        This method assumes the mean function is of type 'zero' or
        'linear_predictor'.

        NOTE: the function implements "conditioning by kriging" (see,
        e.g., Chiles and Delfiner, Geostatistics: Modeling Spatial
        Uncertainty, Wiley, 1999).

        Parameters
        ----------
        ztsim : ndarray, shape (n, nb_paths)
            Unconditional sample paths.
        zi : ndarray, shape (ni, 1) or (ni, )
            Observed values corresponding to the input data points xi.
        xi_ind : ndarray, shape (ni), dtype=int
            Indices of observed data points in ztsim.
        xt_ind : ndarray, shape (nt), dtype=int
            Indices of prediction data points in ztsim.
        lambda_t : ndarray, shape (ni, nt)
            Kriging weights.
        convert_out : bool, optional
            Whether to return numpy arrays or keep _gpmp_backend_ types.

        Returns
        -------
        ztsimc : ndarray, shape (nt, nb_paths)
            Conditional sample paths at the prediction data points xt.
        """
        return sample_paths.conditional_sample_paths(
            self, ztsim, xi_ind, zi, xt_ind, lambda_t, convert_out=convert_out
        )

    def conditional_sample_paths_parameterized_mean(
        self, ztsim, xi, xi_ind, zi, xt, xt_ind, lambda_t, convert_out=True
    ):
        """Generates conditional sample paths with a parameterized mean function.

        This method accommodates parameterized means, adjusting the
        unconditional sample paths 'ztsim' with respect to observed
        values 'zi' at 'xi' and prediction points 'xt', using kriging
        weights 'lambda_t'.

        Parameters
        ----------
        ztsim : ndarray
            Unconditional sample paths.
        xi : ndarray
            Input data points for observed values.
        xi_ind : ndarray, dtype=int
            Indices of observed data points in ztsim.
        zi : ndarray
            Observed values at the input data points xi.
        xt : ndarray
            Prediction data points.
        xt_ind : ndarray, dtype=int
            Indices of prediction data points in ztsim.
        lambda_t : ndarray
            Kriging weights matrix.
        convert_out : bool, optional
            Whether to return numpy arrays or keep _gpmp_backend_ types.

        Returns
        -------
        ztsimc : ndarray
            Conditional sample paths at prediction points xt, adjusted for a parameterized mean.
        """
        return sample_paths.conditional_sample_paths_parameterized_mean(
            self, ztsim, xi, xi_ind, zi, xt, xt_ind, lambda_t, convert_out=convert_out
        )

    # ------------------------------------------------------------------
    # Convenience wrappers to internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_shapes_and_type(*, xi=None, zi=None, xt=None, convert=True):
        """Validate and adjust shapes/types of input arrays."""
        return utils.ensure_shapes_and_type(xi=xi, zi=zi, xt=xt, convert=convert)

    @staticmethod
    def _validate_model_mean(meantype, mean, meanparam):
        """Validate model initialization inputs."""
        return utils.validate_model_mean(meantype, mean, meanparam)
