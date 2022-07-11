## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
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
        output = str("<gpmp.core.Model object> "+hex(id(self)))
        return output

    def __str__(self):
        output = str("<gpmp.core.Model object>")
        return output

    def kriging_predictor_with_zero_mean(self, xi, xt):
        Kii = self.covariance(xi, xi, self.covparam)
        Kit = self.covariance(xi, xt, self.covparam)
        lambda_t = linalg.solve(Kii, Kit, sym_pos=True, overwrite_a=True, overwrite_b=True)

        zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=True)
        zt_posterior_variance = zt_prior_variance - jnp.einsum('i..., i...', lambda_t, Kit)

        return lambda_t, zt_posterior_variance

    def kriging_predictor(self, xi, xt):
        # LHS
        Kii = self.covariance(xi, xi, self.covparam)
        Pi = self.mean(xi, self.meanparam)
        (ni, q) = Pi.shape
        # build [ [K P] ; [P' 0] ]
        LHS = jnp.vstack((\
                    jnp.hstack((Kii, Pi)),
                    jnp.hstack((Pi.transpose(), jnp.zeros((q, q))))
                    ))
        
        # RHS
        Kit = self.covariance(xi, xt, self.covparam)
        Pt = self.mean(xt, self.meanparam)
        RHS = jnp.vstack((Kit, Pt.transpose()))

        # lambdamu_t = RHS^(-1) LHS
        lambdamu_t = linalg.solve(LHS, RHS, overwrite_a=True, overwrite_b=True)

        lambda_t = lambdamu_t[0:ni, :]

        zt_prior_variance = self.covariance(xt, None, self.covparam, pairwise=True)
        zt_posterior_variance = zt_prior_variance - jnp.einsum('i..., i...', lambdamu_t, RHS)

        return lambda_t, zt_posterior_variance
        
    def predict(self, xi, zi, xt, return_lambdas=False):
        """Performs a prediction at target points xt given the data (xi, zi).

        Parameters
        ----------
        xi : ndarray(ni,dim)
            _description_
        zi : ndarray(ni,1)
            _description_
        xt : ndarray(nt,dim)
            _description_
        return_lambdas : bool, optional
            Set return_lambdas=True if lambdas should be returned, by default False

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
            lambda_t, zt_posterior_variance = self.kriging_predictor_with_zero_mean(xi, xt)
        else:
            lambda_t, zt_posterior_variance = self.kriging_predictor(xi, xt)
        
        # posterior mean
        zt_posterior_mean = jnp.einsum('i..., i...', lambda_t, zi)

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

        PtKinvP = jnp.einsum('ki, kj->ij', P, KinvP)

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
        xi : _type_
            _description_
        zi : _type_
            _description_
        covparam : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        K = self.covariance(xi, xi, covparam)
        n = K.shape[0]

        C, lower = linalg.cho_factor(K)

        ldetK = 2 * jnp.sum(jnp.log(jnp.diag(C)))
        Kinv_zi = linalg.cho_solve((C, lower), zi)
        norm2 = jnp.einsum('i..., i...', zi, Kinv_zi)

        L = 1 / 2 * (n * jnp.log(2 * jnp.pi) + ldetK + norm2)

        return L.reshape(())
    
    def negative_log_restricted_likelihood(self, xi, zi, covparam):
        """Computes the negative log- restricted likelihood of the model

        Parameters
        ----------
        xi : _type_
            _description_
        zi : _type_
            _description_
        covparam : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        K = self.covariance(xi, xi, covparam)
        P = self.mean(xi, self.meanparam)
        Pshape = P.shape
        n, q = Pshape

        # Compute a matrix of contrasts
        [Q, R] = jnp.linalg.qr(P, 'complete')
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
        norm2 = jnp.einsum('i..., i...', Wzi, WKWinv_Wzi)

        L = 1 / 2 * ((n - q) * jnp.log(2 * jnp.pi) + ldetWKW + norm2)
        
        return L.reshape(())

    def make_ml_criterion(self, xi, zi):
        """maximum likelihood criterion

        Parameters
        ----------
        xi : _type_
            _description_
        zi : _type_
            _description_

        Returns
        -------
        _type_
            maximum likelihood criterion
        _type_
            maximum likelihood criterion's gradient
        """
        nll = jax.jit(lambda covparam: self.negative_log_likelihood(xi, zi, covparam))
        dnll = jax.grad(nll)
        return nll, dnll

    def make_reml_criterion(self, xi, zi):
        """restricted maximum likelihood criterion

        Parameters
        ----------
        xi : _type_
            _description_
        zi : _type_
            _description_

        Returns
        -------
        _type_
            restricted maximum likelihood criterion
        _type_
            restricted maximum likelihood criterion's gradient
        """
        nlrel = jax.jit(lambda covparam: self.negative_log_restricted_likelihood(xi, zi, covparam))
        dnlrel = jax.grad(nlrel)

        return nlrel, dnlrel

    def norm_k_sqrd_with_zero_mean(self, xi, zi, covparam):
        """

        Parameters
        ----------
        xi : _type_
            _description_
        zi : _type_
            _description_
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
        norm_sqrd = jnp.einsum('i..., i...', zi, Kinv_zi)
        return norm_sqrd

    def norm_k_sqrd(self, xi, zi, covparam):
        """

        Parameters
        ----------
        xi : _type_
            _description_
        zi : _type_
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
        [Q, R] = jnp.linalg.qr(P, 'complete')
        W = Q[:, q:n]

        # Contrasts (n-q) x 1
        Wzi = W.T.dot(zi)

        # Compute G = W' * (K * W), the covariance matrix of contrasts
        G = W.T.dot(K.dot(W))

        # Cholesky factorization: G = U' * U, with upper-triangular U
        C, lower = linalg.cho_factor(G)

        # Compute norm_2 = (W' zi)' * G^(-1) * (W' zi)
        WKWinv_Wzi = linalg.cho_solve((C, lower), Wzi)
        norm_sqrd = jnp.einsum('i..., i...', Wzi, WKWinv_Wzi)

        return norm_sqrd

    def sample_paths(self, xt, nb_paths):
        """Generates m sample paths on xt

        Parameters
        ----------
        xt : _type_
            _description_
        nb_paths : int
            _description_

        Returns
        -------
        _type_
            _description_
        """
        K = self.covariance(xt, xt, self.covparam)

        # Cholesky factorization of the covariance matrix
        (C, lower) = linalg.cho_factor(K)

        # Generates samplepaths
        key = jax.random.PRNGKey(0)
        zsim = jnp.einsum('ki,kj->ij', C,
                          jax.random.normal(key, shape=(K.shape[0], nb_paths)))

        return zsim

    def conditional_sample_paths(self,
                                 ztsim,
                                 lambda_t,
                                 zi,
                                 xi_ind,
                                 noisesim=None):
        """Generates m conditional sample paths on xt

        Parameters
        ----------
        ztsim : _type_
            _description_
        lambda_t : _type_
            _description_
        zi : _type_
            _description_
        xi_ind : _type_
            _description_
        noisesim : bool, optional
            _description_, by default None

        Returns
        -------
        ztsimc : _type_
            _description_
        """

        # dealing with noisy observations?
        noisy = False if noisesim is None else True

        if noisy:
            d = zi.reshape((-1, 1)) - ztsim[xi_ind, :] - noisesim
        else:
            d = zi.reshape((-1, 1)) - ztsim[xi_ind, :]

        ztsimc = ztsim + jnp.einsum('ij,ik->jk', lambda_t, d)

        return ztsimc
