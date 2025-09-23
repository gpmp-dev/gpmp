# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2025, CentraleSupelec
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

    Methods
    -------
    kriging_predictor_with_zero_mean(xi, xt, return_type=0)
        Compute the kriging predictor assuming a zero mean
        function.

    kriging_predictor(xi, xt, return_type=0)
        Compute the kriging predictor considering a non-zero mean
        function.

    predict(xi, zi, xt, return_lambdas=False, zero_neg_variances=True,
            convert_in=True, convert_out=True)
        Performs prediction at target points `xt` given the data `(xi,
        zi)`. The treatment of the mean function is based on the
        `meantype` attribute.

    loo(xi, zi, convert_in=True, convert_out=False)
        Compute the leave-one-out (LOO) prediction error, for model
        validation and hyperparameter tuning.

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

    fisher_information(xi, covparam)
        Computes Fisher information matrix wrt covariance parameters

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
            'linear_predictor' - Linearly parameterized mean function.
        """
        self._validate_model_mean(meantype, mean, meanparam)
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

    def kriging_predictor_with_zero_mean(self, xi, xt, return_type=0):
        """Compute the kriging predictor with zero mean."""
        Kii = self.covariance(xi, xi, self.covparam)
        Kit = self.covariance(xi, xt, self.covparam)

        lambda_t, _ = gnp.cholesky_solve(Kii, Kit)

        zt_posterior_variance = self._compute_posterior_variance(
            xt, lambda_t, Kit, return_type
        )

        return lambda_t, zt_posterior_variance

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

        # lambdamu_t = LHS^(-1) RHS
        lambdamu_t = gnp.solve(
            LHS, RHS, overwrite_a=True, overwrite_b=False, assume_a="sym"
        )
        lambda_t = lambdamu_t[0:ni, :]

        zt_posterior_variance = self._compute_posterior_variance(
            xt, lambdamu_t, RHS, return_type
        )

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

        Ensure to set the appropriate 'meantype' for the desired behavior.
        """
        # Step 1: Prepare the data.
        xi, zi, xt = self._prepare_data(xi, zi, xt, convert_in)
        # Step 2: Select the kriging predictor and adjust for mean.
        zi_centered, zt_prior_mean, lambda_t, zt_posterior_variance = (
            self._select_predictor(xi, zi, xt)
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
        xi_, zi_, _ = Model._ensure_shapes_and_type(xi=xi, zi=zi, convert=convert_in)

        if self.meantype == "zero":
            zloo, sigma2loo, eloo = self._loo_with_zero_mean(self.covparam, xi_, zi_)
        elif self.meantype == "parameterized":
            zloo, sigma2loo, eloo = self._loo_with_parameterized_mean(
                self.meanparam, self.covparam, xi_, zi_
            )
        elif self.meantype == "linear_predictor":
            zloo, sigma2loo, eloo = self._loo_with_linear_predictor_mean_cpd(
                self.meanparam, self.covparam, xi_, zi_
            )
        else:
            raise ValueError(f"Unknown mean type: {self.meantype}")

        if convert_out:
            zloo = gnp.to_np(zloo)
            sigma2loo = gnp.to_np(sigma2loo)
            eloo = gnp.to_np(eloo)

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
            return self._return_inf()
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
        """
        K = self.covariance(xi, xi, covparam)
        P = self.mean(xi, self.meanparam)
        W = self._compute_contrast_matrix(P)
        Wzi = gnp.matmul(W.T, zi)
        G = self._compute_contrast_covariance(W, K)
        try:
            WKWinv_Wzi, C = gnp.cholesky_solve(G, Wzi)
        except RuntimeError:
            return self._return_inf()
        norm2 = gnp.einsum("i..., i...", Wzi, WKWinv_Wzi)
        ldetWKW = 2.0 * gnp.sum(gnp.log(gnp.diag(C)))
        n, q = P.shape
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

        """
        K = self.covariance(xi, xi, covparam)
        ones_vector = gnp.ones(zi.shape)
        Kinv = gnp.cholesky_inv(K)
        Kinv_zi = gnp.einsum("...i, i...", Kinv, zi)
        Kinv_1 = gnp.einsum("...i, i...", Kinv, ones_vector)
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
        W = self._compute_contrast_matrix(P)
        # Contrasts (n-q) x 1
        Wzi = gnp.matmul(W.T, zi)
        # Compute G = W' * (K * W), the covariance matrix of contrasts
        G = self._compute_contrast_covariance(W, K)
        # Compute G^(-1) * (W' zi)
        WKWinv_Wzi, _ = gnp.cholesky_solve(G, Wzi)
        # Compute norm_2 = (W' zi)' * G^(-1) * (W' zi)
        norm_sqrd = gnp.einsum("i..., i...", Wzi, WKWinv_Wzi)
        return norm_sqrd

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
        if covparam is None:
            covparam = self.covparam
        else:
            covparam = gnp.asarray(covparam)

        param_dim = covparam.shape[0]
        fisher_info = gnp.empty((param_dim, param_dim))

        # 1) K and its inverse
        K = self.covariance(xi, xi, covparam)
        try:
            K_inv = gnp.inv(K)
        except:
            raise RuntimeError(
                "Covariance matrix not invertible; adjust hyperparameters or add jitter."
            )

        # 2) Partial derivatives of K w.r.t. each parameter (finite differences)
        dK = []
        for i in range(param_dim):

            def f(tmp_val):
                param_copy = gnp.copy(covparam)
                param_copy = gnp.set_elem_1d(param_copy, i, tmp_val)
                return self.covariance(xi, xi, param_copy)

            dK_i = gnp.derivative_finite_diff(f, covparam[i], epsilon)
            dK.append(dK_i)

        # 3) Fisher Information matrix entries
        for i in range(param_dim):
            for j in range(i, param_dim):
                # 0.5 * trace(K_inv * dK[i] * K_inv * dK[j])
                term = 0.5 * gnp.trace(K_inv @ dK[i] @ K_inv @ dK[j])
                fisher_info = gnp.set_elem_2d(fisher_info, i, j, term)
                fisher_info = gnp.set_elem_2d(fisher_info, j, i, term)

        return fisher_info

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
        if covparam is None:
            covparam = self.covparam
        covparam = gnp.asarray(covparam)
        p = covparam.shape[0]

        K = self.covariance(xi, xi, covparam)
        # Build contrasts
        P = (
            self.mean(xi, self.meanparam)
            if self.meantype == "linear_predictor"
            else None
        )
        if P is not None:
            Q, _ = gnp.qr(P, mode="complete")
            W = Q[:, P.shape[1] :]
            G = gnp.matmul(W.T, gnp.matmul(K, W))
            Cg = gnp.cholesky(G)
        else:
            Ck = gnp.cholesky(K)  # SPD case

        # finite differences for dK_i (central)
        dK = []
        for i in range(p):

            def f(tmp):
                theta = gnp.copy(covparam)
                theta = gnp.set_elem_1d(theta, i, tmp)
                return self.covariance(xi, xi, theta)

            dK_i = gnp.derivative_finite_diff(f, covparam[i], epsilon)
            dK.append(dK_i)

        I = gnp.empty((p, p))
        if P is not None:  # CPD / linear_predictor
            dG = [gnp.matmul(W.T, gnp.matmul(dK_i, W)) for dK_i in dK]

            # Helper: solve G^{-1} A via Cholesky
            def Gsolve(A):
                X, _ = gnp.cholesky_solve(G, A)
                return X

            for i in range(p):
                Gi = Gsolve(dG[i])
                for j in range(i, p):
                    term = 0.5 * gnp.trace(gnp.matmul(Gi, Gsolve(dG[j])))
                    I = gnp.set_elem_2d(I, i, j, term)
                    I = gnp.set_elem_2d(I, j, i, term)
        else:  # SPD case

            def Ksolve(A):
                X, _ = gnp.cholesky_solve(K, A)
                return X

            Ai = [Ksolve(dK_i) for dK_i in dK]
            for i in range(p):
                for j in range(i, p):
                    term = 0.5 * gnp.trace(gnp.matmul(Ai[i], Ai[j]))
                    I = gnp.set_elem_2d(I, i, j, term)
                    I = gnp.set_elem_2d(I, j, i, term)
        return I

    def fisher_information_torch(self, xi, covparam):
        """Compute Fisher Information matrix using second-order differentiation."""
        xi_tensor = gnp.asarray(xi)

        def log_det_cov(params):
            K = self.covariance(xi_tensor, xi_tensor, params)
            L = gnp.cholesky(K)
            return 2.0 * gnp.sum(gnp.log(gnp.diag(L)))

        sodf = gnp.SecondOrderDifferentiableFunction(log_det_cov)
        sodf.evaluate(covparam)
        sodf.gradient()
        fisher_info = 0.5 * sodf.hessian()

        return fisher_info

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
        zi_ = gnp.asarray(zi).reshape(-1, 1)
        ztsim_ = gnp.asarray(ztsim)
        xi_ind = gnp.asarray(xi_ind).reshape(-1)
        xt_ind = gnp.asarray(xt_ind).reshape(-1)

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
        xi_, zi_, xt_ = Model._ensure_shapes_and_type(xi=xi, zi=zi, xt=xt)
        ztsim_ = gnp.asarray(ztsim)
        xi_ind = gnp.asarray(xi_ind).reshape(-1)
        xt_ind = gnp.asarray(xt_ind).reshape(-1)

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
    def _ensure_shapes_and_type(xi=None, zi=None, xt=None, convert=True):
        """Validate and adjust shapes/types of input arrays.

        Parameters
        ----------
        xi : array_like, optional
            Observation points (n, dim).
        zi : array_like, optional
            Observed values (n,) or (n, 1).
        xt : array_like, optional
            Prediction points (m, dim).
        convert : bool, optional
            Convert arrays to backend type (default True).

        Returns
        -------
        tuple
            (xi, zi, xt) with proper shapes and types.
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

    def _validate_model_mean(self, meantype, mean, meanparam):
        """Validate model initialization inputs."""
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

    def _return_inf(self):
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

    def _prepare_data(self, xi, zi, xt, convert_in):
        """Convert inputs to the proper shapes and types."""
        return Model._ensure_shapes_and_type(xi=xi, zi=zi, xt=xt, convert=convert_in)

    def _select_predictor(self, xi, zi, xt):
        """
        Select the appropriate kriging predictor based on meantype.

        Returns
        -------
            zi_centered : centered observed values.
            zt_prior_mean : prior mean adjustment for zt.
            lambda_t : kriging weights.
            zt_posterior_variance : posterior variance.

        """
        # Default: no mean adjustment.
        zt_prior_mean = 0.0
        zi_centered = zi

        if self.meantype == "zero":
            lambda_t, zt_posterior_variance = self.kriging_predictor_with_zero_mean(
                xi, xt
            )
        elif self.meantype == "linear_predictor":
            lambda_t, zt_posterior_variance = self.kriging_predictor(xi, xt)
        elif self.meantype == "parameterized":
            if self.meanparam is None:
                raise ValueError(
                    "For meantype 'parameterized', meanparam should not be None."
                )
            # Use the zero-mean predictor but center the data.
            lambda_t, zt_posterior_variance = self.kriging_predictor_with_zero_mean(
                xi, xt
            )
            zi_prior_mean = self.mean(xi, self.meanparam).reshape(-1)
            zi_centered = zi - zi_prior_mean
            zt_prior_mean = self.mean(xt, self.meanparam).reshape(-1)
        else:
            raise ValueError(
                f"Invalid meantype {self.meantype}. Supported types are 'zero', 'parameterized', and 'linear_predictor'."
            )

        return zi_centered, zt_prior_mean, lambda_t, zt_posterior_variance

    def _compute_posterior_variance(self, xt, lambdamu_t, RHS, return_type=0):
        """Compute posterior variance based on return type."""
        if return_type == -1:
            return None
        elif return_type == 0:
            zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=True)
            return zt_prior_variance - gnp.einsum("i..., i...", lambdamu_t, RHS)
        elif return_type == 1:
            zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=False)
            return zt_prior_variance - gnp.matmul(lambdamu_t.T, RHS)

    def _loo_with_zero_mean(self, covparam, xi, zi):
        """Compute LOO prediction error for zero mean.

        LOO predictions based on the "virtual cross-validation" formula
        """
        K = self.covariance(xi, xi, covparam)
        # K^{-1} z and the Cholesky C
        Kinv_zi, C = gnp.cholesky_solve(K, zi)  # returns (K^{-1} z, C)

        # diag(K^{-1}) via triangular inverse
        Kinvdiag = _diag_kinv_from_chol(C)

        # e_loo,i  = 1 / Kinv_i,i ( Kinv  z )_i
        eloo = Kinv_zi / Kinvdiag
        # sigma2_loo,i = 1 / Kinv_i,i
        sigma2loo = 1.0 / Kinvdiag
        # zloo_i = z_i - e_loo,i
        zloo = zi - eloo
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

    def _loo_with_linear_predictor_mean_cpd(self, meanparam, covparam, xi, zi):
        """Compute LOO prediction error for linear_predictor mean. CPD-safe version."""
        K = self.covariance(xi, xi, covparam)
        P = self.mean(xi, meanparam)
        Q, _R = gnp.qr(P, mode="complete")
        W = Q[:, P.shape[1] :]  # (n, n-q)
        G = gnp.matmul(W.T, gnp.matmul(K, W))  # (n-q, n-q), SPD
        # Solve S = G^{-1} W^T  (n-q x n)
        S, _ = gnp.cholesky_solve(G, W.T)
        # Qinv z = W S z
        Qinvzi = gnp.matmul(W, gnp.matmul(S, zi))
        # diag(Qinv) = row-wise inner products w_i^T (G^{-1} w_i)
        # which equals sum_r W[i,r] * S[r,i]
        Qinvdiag = gnp.sum(W * S.T, axis=1)
        eloo = Qinvzi / Qinvdiag
        sigma2loo = 1.0 / Qinvdiag
        zloo = zi - eloo
        return zloo, sigma2loo, eloo

    def _compute_contrast_matrix(self, P):
        """Compute a matrix of contrasts from design matrix P."""
        n, q = P.shape
        [Q, R] = gnp.qr(P, "complete")
        return Q[:, q:n]

    def _compute_contrast_covariance(self, W, K):
        """Compute covariance matrix of contrasts G = W' * (K * W)."""
        return gnp.matmul(W.T, gnp.matmul(K, W))

    def _kriging_predictor_alternative(self, xi, xt, return_type=0):
        Kii = self.covariance(xi, xi, self.covparam)
        Pi = self.mean(xi, self.meanparam)
        (ni, q) = Pi.shape
        Kit = self.covariance(xi, xt, self.covparam)
        Pt = self.mean(xt, self.meanparam)
        LHS = gnp.vstack((gnp.hstack((Kii, Pi)), gnp.hstack((Pi.T, gnp.zeros((q, q))))))
        RHS = gnp.vstack((Kit, Pt.T))
        try:
            lambdamu_t = gnp.solve(
                LHS, RHS, overwrite_a=True, overwrite_b=False, assume_a="sym"
            )
        except Exception:
            # Fallback: nullspace/contrast route (SPD on Wᵀ K W)
            return self._kriging_predictor_nullspace(xi, xt, return_type)
        lambda_t = lambdamu_t[0:ni, :]
        zt_posterior_variance = self._compute_posterior_variance(
            xt, lambdamu_t, RHS, return_type
        )
        return lambda_t, zt_posterior_variance

    def _kriging_predictor_nullspace(self, xi, xt, return_type=0):
        # CPD-safe universal kriging using contrasts
        K = self.covariance(xi, xi, self.covparam)
        P = self.mean(xi, self.meanparam)
        n, q = P.shape
        Kit = self.covariance(xi, xt, self.covparam)
        Pt = self.mean(xt, self.meanparam)  # (m, q)
        Q, R = gnp.qr(P, mode="complete")
        Q1, W = Q[:, :q], Q[:, q:]
        Rq = R[:q, :q]
        KW = gnp.matmul(K, W)
        G = gnp.matmul(W.T, KW)  # SPD
        alpha, _ = gnp.cholesky_solve(G, gnp.matmul(W.T, Kit))
        beta = gnp.solve(Rq.T, Pt.T, assume_a="sym")
        lambda_t = gnp.matmul(W, alpha) + gnp.matmul(Q1, beta)
        if return_type == -1:
            zt_posterior_variance = None
        elif return_type == 0:
            v0 = self.covariance(xt, xt, self.covparam, pairwise=True)
            RHS = gnp.vstack((Kit, Pt.T))
            LM = gnp.vstack((lambda_t, beta))
            zt_posterior_variance = v0 - gnp.einsum("i..., i...", LM, RHS)
        elif return_type == 1:
            V0 = self.covariance(xt, xt, self.covparam, pairwise=False)
            RHS = gnp.vstack((Kit, Pt.T))
            LM = gnp.vstack((lambda_t, beta))
            zt_posterior_variance = V0 - gnp.matmul(LM.T, RHS)
        else:
            raise ValueError("return_type must be in {-1,0,1}")
        return lambda_t, zt_posterior_variance

    def _diag_kinv_from_chol(C, lower=True):
        """Return diag(K^{-1}) from Cholesky factor C (K = C Cᵀ if lower else Cᵀ C)."""
        n = C.shape[0]
        I = gnp.eye(n)
        Tinv = gnp.solve_triangular(C, I, lower=lower)  # C^{-1}
        # If lower: K^{-1} = C^{-T} C^{-1} ⇒ diag = column-wise sum of squares of C^{-1}
        # If upper: K^{-1} = C^{-1} C^{-T} ⇒ diag = row-wise   sum of squares of C^{-1}
        return gnp.sum(Tinv * Tinv, axis=0) if lower else gnp.sum(Tinv * Tinv, axis=1)
