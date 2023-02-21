# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import warnings
import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg


class Model:
    """
    Model manager class

    Attributes
    ----------
    mean :
    covariance :
    meanparam :
    covparam :

    Methods
    -------
    """

    def __init__(self, mean, covariance, meanparam=None, covparam=None):
        self.mean = mean
        self.covariance = covariance

        self.meanparam = meanparam
        self.covparam = covparam

    def __repr__(self):
        output = str("<gpmp.core.Model object> " + hex(id(self)))
        return output

    def __str__(self):
        output = str("<gpmp.core.Model object>")
        return output

    def kriging_predictor_with_zero_mean(self, xi, xt, return_type=0):
        """Compute the kriging predictor with zero mean"""
        Kii = self.covariance(xi, xi, self.covparam)
        Kit = self.covariance(xi, xt, self.covparam)

        lambda_t = linalg.solve(
            Kii, Kit, sym_pos=True, overwrite_a=True, overwrite_b=True
        )

        if return_type == -1:
            zt_posterior_variance = None
        elif return_type == 0:
            zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=True)
            zt_posterior_variance = zt_prior_variance - jnp.einsum(
                "i..., i...", lambda_t, Kit
            )
        elif return_type == 1:
            zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=False)
            zt_posterior_variance = zt_prior_variance - jnp.matmul(lambda_t.T, Kit)

        return lambda_t, zt_posterior_variance

    def kriging_predictor(self, xi, xt, return_type=0):
        """Compute the kriging predictor with non-zero mean

        Parameters
        ----------
        xi : ndarray(ni, d)
            Observation points
        xt : ndarray(nt, d)
            Prediction points
        return_type : -1, 0 or 1
            If -1, returned posterior variance is None. If 0
            (default), return the posterior variance at points xt. If
            1, return the posterior covariance.

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
        LHS = jnp.vstack(
            (jnp.hstack((Kii, Pi)), jnp.hstack((Pi.transpose(), jnp.zeros((q, q)))))
        )

        # RHS
        Kit = self.covariance(xi, xt, self.covparam)
        Pt = self.mean(xt, self.meanparam)
        RHS = jnp.vstack((Kit, Pt.transpose()))

        # lambdamu_t = RHS^(-1) LHS
        lambdamu_t = linalg.solve(LHS, RHS, overwrite_a=True, overwrite_b=True)

        lambda_t = lambdamu_t[0:ni, :]

        # posterior variance
        if return_type == -1:
            zt_posterior_variance = None
        elif return_type == 0:
            zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=True)
            zt_posterior_variance = zt_prior_variance - jnp.einsum(
                "i..., i...", lambdamu_t, RHS
            )
        elif return_type == 1:
            zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=False)
            zt_posterior_variance = zt_prior_variance - jnp.matmul(lambdamu_t.T, RHS)

        return lambda_t, zt_posterior_variance

    def predict(self, xi, zi, xt, return_lambdas=False, zero_neg_variances=True):
        """Performs a prediction at target points xt given the data (xi, zi).

        Parameters
        ----------
        xi : ndarray(ni,dim)
            observation points
        zi : ndarray(ni,1)
            observed values
        xt : ndarray(nt,dim)
            prediction points
        return_lambdas : bool, optional
            Set return_lambdas=True if lambdas should be returned, by default False
        zero_neg_variances : bool, optional
            Whether to zero negative posterior variances (due to numerical errors), default=True

        Returns
        -------
        z_posterior_mean : ndarray
            2d array of shape nt x 1
        z_posterior variance : ndarray
            2d array of shape nt x 1

        Notes
        -----
        From a Bayesian point of view, the outputs are
        respectively the posterior mean and variance of the
        Gaussian process given the data (xi, zi).
        """

        if self.mean is None:
            lambda_t, zt_posterior_variance = self.kriging_predictor_with_zero_mean(
                xi, xt
            )
        else:
            lambda_t, zt_posterior_variance = self.kriging_predictor(xi, xt)

        if jnp.any(zt_posterior_variance < 0):
            warnings.warn(
                "In predict: negative variances detected. Consider using jitter.", RuntimeWarning
            )
        if zero_neg_variances:
            zt_posterior_variance = jnp.maximum(zt_posterior_variance, 0)

        # posterior mean
        zt_posterior_mean = jnp.einsum("i..., i...", lambda_t, zi)

        # outputs
        if not return_lambdas:
            return (zt_posterior_mean, zt_posterior_variance)
        else:
            return (zt_posterior_mean, zt_posterior_variance, lambda_t)

    def loo_with_zero_mean(self, xi, zi):

        n = xi.shape[0]
        K = self.covariance(xi, xi, self.covparam)

        # Use the "virtual cross-validation" formula
        C, lower = linalg.cho_factor(K)
        Kinv = linalg.cho_solve((C, lower), jnp.eye(n))

        # e_loo,i  = 1 / Kinv_i,i ( Kinv  z )_i
        Kinvzi = jnp.matmul(Kinv, zi)
        Kinvdiag = jnp.diag(Kinv)
        eloo = Kinvzi / Kinvdiag

        # sigma2_loo,i = 1 / Kinv_i,i
        sigma2loo = 1 / Kinvdiag

        # zloo_i = z_i - e_loo,i
        zloo = zi - eloo

        return zloo, sigma2loo, eloo

    def loo(self, xi, zi):

        n = xi.shape[0]
        K = self.covariance(xi, xi, self.covparam)
        P = self.mean(xi, self.meanparam)

        # Use the "virtual cross-validation" formula
        # Qinv = K^-1 - K^-1P (Pt K^-1 P)^-1 Pt K^-1
        C, lower = linalg.cho_factor(K)
        Kinv = linalg.cho_solve((C, lower), jnp.eye(n))
        KinvP = linalg.cho_solve((C, lower), P)

        PtKinvP = jnp.einsum("ki, kj->ij", P, KinvP)

        R = linalg.solve(PtKinvP, KinvP.transpose())
        Qinv = Kinv - jnp.matmul(KinvP, R)

        # e_loo,i  = 1 / Q_i,i ( Qinv  z )_i
        Qinvzi = jnp.matmul(Qinv, zi)
        Qinvdiag = jnp.diag(Qinv)
        eloo = Qinvzi / Qinvdiag

        # sigma2_loo,i = 1 / Qinv_i,i
        sigma2loo = 1 / Qinvdiag

        # z_loo
        zloo = zi - eloo

        # __import__("pdb").set_trace()

        return zloo, sigma2loo, eloo

    def negative_log_likelihood(self, xi, zi, covparam):
        """Computes the negative log-likelihood of the model

        Parameters
        ----------
        xi : ndarray(ni,d)
            points
        zi : ndarray(ni,1)
            values
        covparam : _type_
            _description_

        Returns
        -------
        nll : scalar
            negative log likelihood
        """
        K = self.covariance(xi, xi, covparam)
        n = K.shape[0]

        C, lower = linalg.cho_factor(K)

        ldetK = 2 * jnp.sum(jnp.log(jnp.diag(C)))
        Kinv_zi = linalg.cho_solve((C, lower), zi)
        norm2 = jnp.einsum("i..., i...", zi, Kinv_zi)

        L = 1 / 2 * (n * jnp.log(2 * jnp.pi) + ldetK + norm2)

        return L.reshape(())

    def negative_log_restricted_likelihood(self, xi, zi, covparam):
        """Computes the negative log- restricted likelihood of the model

        Parameters
        ----------
        xi : ndarray(ni,d)
            points
        zi : ndarray(ni,1)
            values
        covparam : _type_
            _description_

        Returns
        -------
        nll : scalar
            negative log likelihood
        """
        K = self.covariance(xi, xi, covparam)
        P = self.mean(xi, self.meanparam)
        Pshape = P.shape
        n, q = Pshape

        # Compute a matrix of contrasts
        [Q, R] = jnp.linalg.qr(P, "complete")
        W = Q[:, q:n]

        # Contrasts (n-q) x 1
        Wzi = W.T.dot(zi)

        # Compute G = W' * (K * W), the covariance matrix of contrasts
        G = W.T.dot(K.dot(W))

        # Cholesky factorization: G = U' * U, with upper-triangular U
        C, lower = linalg.cho_factor(G)

        # Compute log(det(G)) using the Cholesky factorization
        ldetWKW = 2 * jnp.sum(jnp.log(jnp.diag(C)))

        # Compute norm2 = (W' zi)' * G^(-1) * (W' zi)
        WKWinv_Wzi = linalg.cho_solve((C, lower), Wzi)
        norm2 = jnp.einsum("i..., i...", Wzi, WKWinv_Wzi)

        L = 1 / 2 * ((n - q) * jnp.log(2 * jnp.pi) + ldetWKW + norm2)

        return L.reshape(())

    def norm_k_sqrd_with_zero_mean(self, xi, zi, covparam):
        """

        Parameters
        ----------
        xi : ndarray(ni, d)
            points
        zi : ndarray(ni, 1)
            values
        covparam : _type_
            _description_

        Returns
        -------
        _type_
            z' K^-1 z
        """
        K = self.covariance(xi, xi, covparam)
        C, lower = linalg.cho_factor(K)
        Kinv_zi = linalg.cho_solve((C, lower), zi)
        norm_sqrd = jnp.einsum("i..., i...", zi, Kinv_zi)
        return norm_sqrd

    def norm_k_sqrd(self, xi, zi, covparam):
        """

        Parameters
        ----------
        xi : ndarray(ni, d)
            _description_
        zi : ndarray(ni, 1)
            _description_
        covparam : _type_
            _description_

        Returns
        -------
        _type_
            (Wz)' (WKW)^-1 Wz where W is a matrix of contrasts
        """
        K = self.covariance(xi, xi, covparam)
        P = self.mean(xi, self.meanparam)
        n, q = P.shape

        # Compute a matrix of contrasts
        [Q, R] = jnp.linalg.qr(P, "complete")
        W = Q[:, q:n]

        # Contrasts (n-q) x 1
        Wzi = W.T.dot(zi)

        # Compute G = W' * (K * W), the covariance matrix of contrasts
        G = W.T.dot(K.dot(W))

        # Cholesky factorization: G = U' * U, with upper-triangular U
        C, lower = linalg.cho_factor(G)

        # Compute norm_2 = (W' zi)' * G^(-1) * (W' zi)
        WKWinv_Wzi = linalg.cho_solve((C, lower), Wzi)
        norm_sqrd = jnp.einsum("i..., i...", Wzi, WKWinv_Wzi)

        return norm_sqrd

    def sample_paths(self, xt, nb_paths, check_result=True, method="chol"):
        """Generates nb_paths sample paths on xt from the GP model GP(0, k),
        where k is the covariance specified by Model.covariance

        Parameters
        ----------
        xt : ndarray(nt, 1)
            _description_
        nb_paths : int
            _description_

        Returns
        -------
        _type_
            _description_

        """
        K = self.covariance(xt, xt, self.covparam)

        # Factorization of the covariance matrix
        if method == "chol":
            (C, lower) = linalg.cho_factor(K)
            if check_result:
                if jnp.isnan(C).any():
                    raise AssertionError(
                        "In sample_paths: Cholesky factorization failed. Consider using jitter or the sdv switch."
                    )
        elif method == "svd":
            u, s, vt = jnp.linalg.svd(K, full_matrices=True, hermitian=True)
            C = jnp.matmul(u * jnp.sqrt(s), vt)

        # Generates samplepaths
        key = jax.random.PRNGKey(0)
        zsim = jnp.einsum(
            "ki,kj->ij", C, jax.random.normal(key, shape=(K.shape[0], nb_paths))
        )

        return zsim

    def conditional_sample_paths(self, ztsim, xi_ind, zi, xt_ind, lambda_t):
        """Generates conditionnal sample paths on xt from unconditional
        sample paths ztsim, using the matrix of kriging weights
        lambda_t, which is provided by kriging_predictor() or predict().

        Conditioning is done with respect to ni observations, located
        at the indices given by xi_ind in ztsim, with corresponding
        observed values zi. xt_ind specifies indices in ztsim
        corresponding to conditional simulation points.

        NOTE: the function implements "conditioning by kriging" (see,
        e.g., Chiles and Delfiner, Geostatistics: Modeling Spatial
        Uncertainty, Wiley, 1999).

        Parameters
        ----------
        ztsim : ndarray(n, nb_paths)
            Unconditional sample paths
        zi : ndarray(ni, 1)
            Observed values
        xi_ind : ndarray(ni, 1, dtype=int)
            Observed indices in ztsim
        xt_ind : ndarray(nt, 1, dtype=int)
            Prediction indices in ztsim
        lambda_t : ndarray(ni, nt)
            Kriging weights

        Returns
        -------
        ztsimc : ndarray(nt, nb_paths)
            Conditional sample paths at xt

        """

        d = zi.reshape((-1, 1)) - ztsim[xi_ind, :]

        ztsimc = ztsim[xt_ind, :] + jnp.einsum("ij,ik->jk", lambda_t, d)

        return ztsimc
