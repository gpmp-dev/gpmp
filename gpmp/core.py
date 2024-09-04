# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2024, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import warnings
import gpmp.num as gnp


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

        where x and y are (n x d) and (m x d) arrays of data points,
        and pairwise indicates if an (n x m) covariance matrix
        (pairwise == False) or an (n x 1) vector (n == m, pairwise =
        True) should be returned

    meanparam : array_like, optional
        Parameters for the mean function, given as a one-dimensional
        array. These parameters define the specific form and behavior
        of the mean function used in the GP model.

    covparam : array_like, optional
        Parameters for the covariance function, given as a
        one-dimensional array. These parameters determine the
        characteristics of the covariance function, influencing
        aspects like length scale, variance, and smoothness of the GP.

    meantype : str, optional
        The type of mean used in the model. It can be:

        - 'zero': A zero mean function, implying the GP has a zero prior
          mean function. Then, self.mean is never called and should be set to None.
        - 'parameterized': A parameterized mean function with parameterized parameters. Useful
          when there's prior knowledge about the mean behavior of the
          function being modeled.
        - 'linear_predictor': A linearly parameterized mean function with
          linear_predictor parameters, suitable for situations where the mean
          structure is to be learned from data.

    Methods
    -------
    kriging_predictor_with_zero_mean(xi, xt, return_type=0)
        Compute the kriging predictor assuming a zero mean
        function. Useful for models where the mean is assumed to be
        negligible.

    kriging_predictor(xi, xt, return_type=0)
        Compute the kriging predictor considering a non-zero mean
        function. This method is essential in practical applications.

    predict(xi, zi, xt, return_lambdas=False, zero_neg_variances=True, convert_in=True, convert_out=True)
        Performs prediction at target points `xt` given the data `(xi,
        zi)`. The treatment of the mean function is based on the
        `meantype` attribute.

    loo(xi, zi, convert_in=True, convert_out=False)
        Compute the leave-one-out (LOO) prediction error. This method
        is valuable for model validation and hyperparameter tuning.

    negative_log_likelihood_zero_mean(covparam, xi, zi)
        Computes the negative log-likelihood for a zero-mean GP model.

    negative_log_likelihood(meanparam, covparam, xi, zi)
        Computes the negative log-likelihood for a GP model with a given mean.

    negative_log_restricted_likelihood(covparam, xi, zi)
        Computes the negative log-restricted likelihood (REML)

    norm_k_sqrd_with_zero_mean(xi, zi, covparam)
        Computes the squared norm of the residual vector for a
        zero-mean GP model.

    norm_k_sqrd(xi, zi, covparam)
        Computes the squared norm of the residual vector after
        applying a contrast matrix.

    k_inverses(xi, zi, covparam)
        Calculates various quantities involving the inverse of the
        covariance matrix K.

    sample_paths(xt, nb_paths, method="chol", check_result=True)
        Generates sample paths from the GP model at specified input points.

    conditional_sample_paths(ztsim, xi_ind, zi, xt_ind, lambda_t)
        Generates conditional sample paths from unconditional sample
        paths using kriging weights.

    Examples
    --------
    >>> import gpmp as gp
    >>> import gpmp.num as gnp
    >>> mean = lambda x, meanparam: (meanparam[0] + meanparam[1] * x)
    >>> def covariance(x, y, covparam, pairwise=False):
    >>>     p = 0
    >>>     return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)
    >>> model = gp.core.Model(mean, covariance, meanparam=[0.5, 0.2], covparam=[1.0, 0.1])
    >>> xi = gnp.array([0.0, 1.0, 2.0, 3.0, 5.0]).reshape(-1, 1)
    >>> zi = gnp.array([0.0, 1.2, 2.5, 4.2, 4.3])
    >>> xt = gnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]).reshape(-1, 1)
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
            'linear_predictor' - Linearly parameterized mean function with linear_predictor parameters.
        """
        self.meantype = meantype

        if meantype not in ["zero", "parameterized", "linear_predictor"]:
            raise ValueError(
                "meantype must be one of 'zero', 'parameterized', or 'linear_predictor'"
            )

        if meantype == "zero" and mean is not None:
            raise ValueError("For meantype 'zero', mean must be None")

        if meantype in ["parameterized", "linear_predictor"] and not callable(mean):
            raise TypeError(
                "For meantype 'parameterized' or 'linear_predictor', mean must be a callable function"
            )

        self.meanparam = meanparam
        self.covparam = covparam
        self.mean = mean
        self.covariance = covariance

    def __repr__(self):
        output = str("<gpmp.core.Model object> " + hex(id(self)))
        return output

    def __str__(self):
        output = str("<gpmp.core.Model object>")
        return output

    def kriging_predictor_with_zero_mean(self, xi, xt, return_type=0):
        """Compute the kriging predictor with zero mean."""
        Kii = self.covariance(xi, xi, self.covparam)
        Kit = self.covariance(xi, xt, self.covparam)

        lambda_t, _ = gnp.cholesky_solve(Kii, Kit)

        if return_type == -1:
            zt_posterior_variance = None
        elif return_type == 0:
            zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=True)
            zt_posterior_variance = zt_prior_variance - gnp.einsum(
                "i..., i...", lambda_t, Kit
            )
        elif return_type == 1:
            zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=False)
            zt_posterior_variance = zt_prior_variance - gnp.matmul(lambda_t.T, Kit)

        return lambda_t, zt_posterior_variance

    def kriging_predictor(self, xi, xt, return_type=0):
        """Compute the kriging predictor with non-zero mean.

        Parameters
        ----------
        xi : ndarray(ni, d)
            Observation points
        xt : ndarray(nt, d)
            Prediction points
        return_type : -1, 0 or 1
            If -1, returned posterior variance is None. If 0
            (default), return the posterior variance at points xt.
            If 1, return the posterior covariance.

        Notes
        -----
        If return_type==1, then the covariance function k must be
        built so that k(xi, xi, covparam) returns the covariance
        matrix of observations, and k(xt, xt, covparam) returns the
        covariance matrix of the predictands. This means that the
        information of which are the observation points and which are
        the prediction points must be coded in xi / xt
        """
        # LHS
        Kii = self.covariance(xi, xi, self.covparam)
        Pi = self.mean(xi, self.meanparam)
        (ni, q) = Pi.shape
        # build [ [K P] ; [P' 0] ]
        LHS = gnp.vstack((gnp.hstack((Kii, Pi)), gnp.hstack((Pi.T, gnp.zeros((q, q))))))

        # RHS
        Kit = self.covariance(xi, xt, self.covparam)
        Pt = self.mean(xt, self.meanparam)
        RHS = gnp.vstack((Kit, Pt.T))

        # lambdamu_t = RHS^(-1) LHS
        lambdamu_t = gnp.solve(
            LHS, RHS, overwrite_a=True, overwrite_b=True, assume_a="sym"
        )

        lambda_t = lambdamu_t[0:ni, :]

        # posterior variance
        if return_type == -1:
            zt_posterior_variance = None
        elif return_type == 0:
            zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=True)
            zt_posterior_variance = zt_prior_variance - gnp.einsum(
                "i..., i...", lambdamu_t, RHS
            )
        elif return_type == 1:
            zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=False)
            zt_posterior_variance = zt_prior_variance - gnp.matmul(lambdamu_t.T, RHS)

        return lambda_t, zt_posterior_variance

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
            Observation points.
        zi : ndarray or gnp.array of shape (ni,) or (ni, 1)
            Observed values.
        xt : ndarray or gnp.array of shape (nt, dim)
            Prediction points.
        return_lambdas : bool, optional
            Set return_lambdas=True if lambdas should be returned, by default False.
        zero_neg_variances : bool, optional
            Whether to zero negative posterior variances (due to numerical errors), default=True.
        convert_in : bool, optional
            Whether to convert input arrays to _gpmp_backend_ type or keep as-is.
        convert_out : bool, optional
            Whether to return numpy arrays or keep _gpmp_backend_ types.

        Returns
        -------
        z_posterior_mean : gnp.array or ndarray
            1D array of shape nt representing the posterior mean.
        z_posterior_variance : gnp.array or ndarray
            1D array of shape nt representing the posterior variance.
        lambda_t : gnp.array(ni, nt), optional
            2D array of kriging weights, only returned if return_lambdas=True.

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

        Ensure to set the appropriate 'meantype' for the desired
        behavior. Supported types are 'zero', 'parameterized', and 'linear_predictor'.
        """
        xi_, zi_, xt_ = Model.ensure_shapes_and_type(
            xi=xi, zi=zi, xt=xt, convert=convert_in
        )

        # Decide which kriging predictor to use and if we need to adjust for mean
        zt_prior_mean_ = 0.0
        if self.meantype == "zero":
            lambda_t, zt_posterior_variance_ = self.kriging_predictor_with_zero_mean(
                xi_, xt_
            )
        elif self.meantype == "linear_predictor":
            lambda_t, zt_posterior_variance_ = self.kriging_predictor(xi_, xt_)
        elif self.meantype == "parameterized":
            lambda_t, zt_posterior_variance_ = self.kriging_predictor_with_zero_mean(
                xi_, xt_
            )

            if self.meanparam is None:
                raise ValueError(
                    "For meantype 'parameterized', meanparam *should not* be None."
                )
            zi_prior_mean_ = self.mean(xi_, self.meanparam).reshape(-1)
            zi_ = zi_ - zi_prior_mean_
            zt_prior_mean_ = self.mean(xt_, self.meanparam).reshape(-1)
        else:
            raise ValueError(
                f"Invalid mean_type {self.mean_type}. Supported types are 'zero', 'parameterized', and 'linear_predictor'."
            )

        if gnp.any(zt_posterior_variance_ < 0.0):
            warnings.warn(
                "In predict: negative variances detected. Consider using jitter.",
                RuntimeWarning,
            )
        if zero_neg_variances:
            zt_posterior_variance_ = gnp.maximum(zt_posterior_variance_, 0.0)

        # posterior mean
        zt_posterior_mean_ = gnp.einsum("i..., i...", lambda_t, zi_) + zt_prior_mean_

        # outputs
        if convert_out:
            zt_posterior_mean = gnp.to_np(zt_posterior_mean_)
            zt_posterior_variance = gnp.to_np(zt_posterior_variance_)
        else:
            zt_posterior_mean = zt_posterior_mean_
            zt_posterior_variance = zt_posterior_variance_

        if not return_lambdas:
            return (zt_posterior_mean, zt_posterior_variance)
        else:
            return (zt_posterior_mean, zt_posterior_variance, lambda_t)

    def loo(self, xi, zi, convert_in=True, convert_out=False):
        """Compute the leave-one-out (LOO) prediction error.

        This method computes the LOO prediction error using the
        "virtual cross-validation" formula, which allows for efficient
        computation of LOO predictions without re-fitting the model.

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

        Examples
        --------
        >>> xi = np.array([[1, 2], [3, 4], [5, 6]])
        >>> zi = np.array([1.2, 2.5, 4.2])
        >>> model = Model(mean, covariance, meanparam=[0.5, 0.2], covparam=[1.0, 0.1])
        >>> zloo, sigma2loo, eloo = model.loo(xi, zi)
        """
        xi_, zi_, _ = Model.ensure_shapes_and_type(xi=xi, zi=zi, convert=convert_in)

        if self.meantype == "zero":
            zloo, sigma2loo, eloo = self._loo_with_zero_mean(self.covparam, xi_, zi_)
        elif self.meantype == "parameterized":
            zloo, sigma2loo, eloo = self._loo_with_parameterized_mean(
                self.meanparam, self.covparam, xi_, zi_
            )
        elif self.meantype == "linear_predictor":
            zloo, sigma2loo, eloo = self._loo_with_linear_predictor_mean(
                self.meanparam, self.covparam, xi_, zi_
            )
        else:
            raise ValueError(f"Unknown mean type: {self.meantype}")

        if convert_out:
            zloo = gnp.to_np(zloo)
            sigma2loo = gnp.to_np(sigma2loo)
            eloo = gnp.to_np(eloo)

        return zloo, sigma2loo, eloo

    def _loo_with_zero_mean(self, covparam, xi, zi):
        """Compute LOO prediction error for zero mean."""
        K = self.covariance(xi, xi, covparam)  # shape (n, n)

        # Use the "virtual cross-validation" formula
        Kinv = gnp.cholesky_inv(K)

        # e_loo,i  = 1 / Kinv_i,i ( Kinv  z )_i
        Kinvzi = gnp.matmul(Kinv, zi)  # shape (n, )
        Kinvdiag = gnp.diag(Kinv)  # shape (n, )
        eloo = Kinvzi / Kinvdiag  # shape (n, )

        # sigma2_loo,i = 1 / Kinv_i,i
        sigma2loo = 1.0 / Kinvdiag  # shape (n, )

        # zloo_i = z_i - e_loo,i
        zloo = zi - eloo  # shape (n, )

        return zloo, sigma2loo, eloo

    def _loo_with_parameterized_mean(self, meanparam, covparam, xi, zi):
        """Compute LOO prediction error for parameterized mean."""
        zi_prior_mean = self.mean(xi, meanparam).reshape(-1)
        centered_zi = zi - zi_prior_mean
        zloo_centered, sigma2loo, eloo_centered = self._loo_with_zero_mean(
            covparam, xi, centered_zi
        )
        zloo = zloo_centered + zi_prior_mean
        return zloo, sigma2loo, eloo_centered

    def _loo_with_linear_predictor_mean(self, meanparam, covparam, xi, zi):
        """Compute LOO prediction error for linear_predictor mean."""
        K = self.covariance(xi, xi, covparam)
        P = self.mean(xi, meanparam)

        # Use the "virtual cross-validation" formula
        # Qinv = K^-1 - K^-1P (Pt K^-1 P)^-1 Pt K^-1
        Kinv = gnp.inv(K)
        KinvP = gnp.matmul(Kinv, P)
        PtKinvP = gnp.einsum("ki, kj->ij", P, KinvP)
        R = gnp.solve(PtKinvP, KinvP.T)
        Qinv = Kinv - gnp.matmul(KinvP, R)

        # e_loo,i  = 1 / Q_i,i ( Qinv  z )_i
        Qinvzi = gnp.matmul(Qinv, zi)  # shape (n, )
        Qinvdiag = gnp.diag(Qinv)  # shape (n, )
        eloo = Qinvzi / Qinvdiag  # shape (n, )

        # sigma2_loo,i = 1 / Qinv_i,i
        sigma2loo = 1.0 / Qinvdiag  # shape (n, )

        # z_loo
        zloo = zi - eloo  # shape (n, )

        return zloo, sigma2loo, eloo

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
        K = self.covariance(xi, xi, covparam)
        n = K.shape[0]

        try:
            Kinv_zi, C = gnp.cholesky_solve(K, zi)
        except RuntimeError:
            if gnp._gpmp_backend_ == "jax" or gnp._gpmp_backend_ == "numpy":
                return gnp.inf
            elif gnp._gpmp_backend_ == "torch":
                inf_tensor = gnp.tensor(float("inf"), requires_grad=True)
                return inf_tensor  # returns inf with None gradient

        norm2 = gnp.einsum("i..., i...", zi, Kinv_zi)
        ldetK = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))

        L = 0.5 * (n * gnp.log(2.0 * gnp.pi) + ldetK + norm2)

        return L.reshape(())

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
            is the number of data points and d is the dimensionality
            of the input space.
        zi : ndarray(ni, )
            Observed values corresponding to each data point in xi.

        Returns
        -------
        nll : scalar
            Negative log-likelihood of the observed data given the
            model, mean, and covariance parameters.
        """
        zi_prior_mean = self.mean(xi, meanparam).reshape(-1)
        centered_zi = zi - zi_prior_mean

        # Call the zero mean version with the centered observations
        return self.negative_log_likelihood_zero_mean(covparam, xi, centered_zi)

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

        Examples
        --------
        >>> xi = np.array([[1, 2], [3, 4], [5, 6]])
        >>> zi = np.array([1.2, 2.5, 4.2])
        >>> model = Model(mean, covariance, meanparam=[0.5, 0.2], covparam=[1.0, 0.1])
        >>> covparam = np.array([1.0, 0.1])
        >>> L = model.negative_log_restricted_likelihood(covparam, xi, zi)
        """
        K = self.covariance(xi, xi, covparam)
        P = self.mean(xi, self.meanparam)
        n, q = P.shape

        # Compute a matrix of contrasts
        [Q, R] = gnp.qr(P, "complete")
        W = Q[:, q:n]

        # Contrasts (n-q) x 1
        Wzi = gnp.matmul(W.T, zi)

        # Compute G = W' * (K * W), the covariance matrix of contrasts
        G = gnp.matmul(W.T, gnp.matmul(K, W))

        # Cholesky factorization: G = U' * U, with upper-triangular U
        # Compute G^(-1) * (W' zi)
        try:
            WKWinv_Wzi, C = gnp.cholesky_solve(G, Wzi)
        except RuntimeError:
            if gnp._gpmp_backend_ == "jax" or gnp._gpmp_backend_ == "numpy":
                return gnp.inf
            elif gnp._gpmp_backend_ == "torch":
                # Use LinAlgError instead of raising RuntimeError for linalg operations
                # https://github.com/pytorch/pytorch/issues/64785
                # https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
                # extype, value, tb = __import__("sys").exc_info()
                # __import__("traceback").print_exc()
                # __import__("pdb").post_mortem(tb)
                inf_tensor = gnp.tensor(float("inf"), requires_grad=True)
                return inf_tensor  # returns inf with None gradient

        # Compute norm2 = (W' zi)' * G^(-1) * (W' zi)
        norm2 = gnp.einsum("i..., i...", Wzi, WKWinv_Wzi)

        # Compute log(det(G)) using the Cholesky factorization
        ldetWKW = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))

        L = 0.5 * ((n - q) * gnp.log(2.0 * gnp.pi) + ldetWKW + norm2)

        return L.reshape(())

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

        Examples
        --------
        >>> xi = np.array([[1, 2], [3, 4], [5, 6]])
        >>> zi = np.array([1.2, 2.5, 4.2])
        >>> model = Model(mean, covariance, meanparam=[0.5, 0.2], covparam=[1.0, 0.1])
        >>> covparam = np.array([1.0, 0.1])
        >>> norm_sqrd = model.norm_k_sqrd_with_zero_mean(xi, zi, covparam)
        """
        K = self.covariance(xi, xi, covparam)
        Kinv_zi, _ = gnp.cholesky_solve(K, zi)
        norm_sqrd = gnp.einsum("i..., i...", zi, Kinv_zi)

        return norm_sqrd

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

        Examples
        --------
        >>> xi = np.array([[1, 2], [3, 4], [5, 6]])
        >>> zi = np.array([[1.2], [2.5], [4.2]])
        >>> model = Model(mean, covariance, meanparam=[0.5, 0.2], covparam=[1.0, 0.1])
        >>> covparam = np.array([1.0, 0.1])
        >>> zTKinvz, Kinv1, Kinvz = model.compute_k_inverses(xi, zi, covparam)
        """
        K = self.covariance(xi, xi, covparam)
        ones_vector = gnp.ones(zi.shape)

        Kinv = gnp.cholesky_inv(K)
        Kinv_zi = gnp.einsum("...i, i... ", Kinv, zi)
        Kinv_1 = gnp.einsum("...i, i... ", Kinv, ones_vector)

        zTKinvz = gnp.einsum("i..., i...", zi, Kinv_zi)

        return zTKinvz, Kinv_1, Kinv_zi

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
        K = self.covariance(xi, xi, covparam)
        P = self.mean(xi, self.meanparam)
        n, q = P.shape

        # Compute a matrix of contrasts
        [Q, R] = gnp.qr(P, "complete")
        W = Q[:, q:n]

        # Contrasts (n-q) x 1
        Wzi = gnp.matmul(W.T, zi)

        # Compute G = W' * (K * W), the covariance matrix of contrasts
        G = gnp.matmul(W.T, gnp.matmul(K, W))

        # Compute G^(-1) * (W' zi)
        WKWinv_Wzi, _ = gnp.cholesky_solve(G, Wzi)

        # Compute norm_2 = (W' zi)' * G^(-1) * (W' zi)
        norm_sqrd = gnp.einsum("i..., i...", Wzi, WKWinv_Wzi)

        return norm_sqrd

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

        Examples
        --------
        >>> xt = np.array([[1, 2], [3, 4], [5, 6]])
        >>> model = Model(mean, covariance, meanparam=[0.5, 0.2], covparam=[1.0, 0.1])
        >>> nb_paths = 10
        >>> sample_paths = model.sample_paths(xt, nb_paths)
        """
        xt_ = gnp.asarray(xt)

        K = self.covariance(xt_, xt_, self.covparam)

        # Factorization of the covariance matrix
        if method == "chol":
            C = gnp.cholesky(K)
            if check_result:
                if gnp.isnan(C).any():
                    raise AssertionError(
                        "In sample_paths: Cholesky factorization failed. Consider using jitter or the sdv switch."
                    )
        elif method == "svd":
            u, s, vt = gnp.svd(K, full_matrices=True, hermitian=True)
            C = gnp.matmul(u * gnp.sqrt(s), vt)

        # Generates samplepaths
        zsim = gnp.matmul(C, gnp.randn(K.shape[0], nb_paths))

        return zsim

    def conditional_sample_paths(self, ztsim, xi_ind, zi, xt_ind, lambda_t, convert_out=True):
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
        xi_ind : ndarray, shape (ni, 1), dtype=int
            Indices of observed data points in ztsim.
        xt_ind : ndarray, shape (nt, 1), dtype=int
            Indices of prediction data points in ztsim.
        lambda_t : ndarray, shape (ni, nt)
            Kriging weights.
        convert_out : bool, optional
            Whether to return numpy arrays or keep _gpmp_backend_ types.

        Returns
        -------
        ztsimc : ndarray, shape (nt, nb_paths)
            Conditional sample paths at the prediction data points xt.

        Examples
        --------
        >>> ztsim = np.random.randn(10, 5)
        >>> zi = np.array([[1], [2], [3]])
        >>> xi_ind = np.array([[0], [3], [7]])
        >>> xt_ind = np.array([[1], [2], [4], [5], [6], [8], [9]])
        >>> lambda_t = np.random.randn(3, 7)
        >>> ztsimc = model.conditional_sample_paths(ztsim, xi_ind, zi, xt_ind, lambda_t)
        """
        zi_ = gnp.asarray(zi).reshape(-1, 1)
        ztsim_ = gnp.asarray(ztsim)

        delta = zi_ - ztsim_[xi_ind, :]

        ztsimc = ztsim_[xt_ind, :] + gnp.einsum("ij,ik->jk", lambda_t, delta)

        if convert_out:
            ztsimc = gnp.to_np(ztsimc)

        return ztsimc

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
        xi_, zi_, xt_ = Model.ensure_shapes_and_type(xi=xi, zi=zi, xt=xt)
        ztsim_ = gnp.asarray(ztsim)

        zi_prior_mean_ = self.mean(xi_, self.meanparam).reshape(-1)
        zi_centered_ = zi_ - zi_prior_mean_

        zt_prior_mean_ = self.mean(xt_, self.meanparam).reshape(-1, 1)

        delta = zi_centered_.reshape(-1, 1) - ztsim_[xi_ind, :]

        ztsimc = (
            ztsim_[xt_ind, :]
            + gnp.einsum("ij,ik->jk", lambda_t, delta)
            + zt_prior_mean_
        )

        if convert_out:
            ztsimc = gnp.to_np(ztsimc)

        return ztsimc

    @staticmethod
    def ensure_shapes_and_type(xi=None, zi=None, xt=None, convert=True):
        """Ensure correct shapes and types for provided arrays.

        Parameters
        ----------
        xi : ndarray or gnp.array(ni, dim), optional
            Observation points.
        zi : ndarray or gnp.array(ni,) or gnp.array(ni, 1), optional
            Observed values.
        xt : ndarray or gnp.array(nt, dim), optional
            Prediction points.
        convert : bool, optional
            Whether to convert input arrays to _gpmp_backend_ type or keep as-is.

        Returns
        -------
        xi, zi, xt : tuple
            Tuples containing arrays with ensured shapes and types.
        """

        if xi is not None:
            assert len(xi.shape) == 2, "xi should be a 2D array"

        if zi is not None:
            if len(zi.shape) == 2:
                assert (
                    zi.shape[1] == 1
                ), "zi should only have one column if it's a 2D array"
                zi = zi.reshape(-1)  # reshapes (ni, 1) to (ni,)
            else:
                assert len(zi.shape) == 1, "zi should either be 1D or a 2D column array"

        if xt is not None:
            assert len(xt.shape) == 2, "xt should be a 2D array"

        if xi is not None and zi is not None:
            assert (
                xi.shape[0] == zi.shape[0]
            ), "Number of rows in xi should be equal to the number of rows in zi"
        if xi is not None and xt is not None:
            assert (
                xi.shape[1] == xt.shape[1]
            ), "xi and xt should have the same number of columns"

        if convert:
            if xi is not None:
                xi = gnp.asarray(xi)
            if zi is not None:
                zi = gnp.asarray(zi)
            if xt is not None:
                xt = gnp.asarray(xt)

        return xi, zi, xt
