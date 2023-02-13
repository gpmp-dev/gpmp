'''GP conditional sample paths

Copyright (c) 2022, CentraleSupelec
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
License: GPLv3 (see LICENSE)
'''
import math
import numpy as np
import jax.numpy as jnp
import gpmp as gp

## -- dataset


def generate_data():
    dim = 1
    box = [[-1], [1]]
    
    nt = 200
    # build a regular grid and evaluate the test function
    xt_ = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gp.misc.testfunctions.twobumps(xt_)

    # extend xt so that the second column is the noise variance
    xt = np.hstack((xt_, np.zeros((nt, 1))))

    # observations
    noise_std_func = lambda x : 0.1 + (x + 1)**2
    
    xi1 = gp.misc.designs.regulargrid(dim, 30, box)
    xi2 = gp.misc.designs.regulargrid(dim, 50, [[0], [1]])
    xi_ = np.vstack((xi1, xi2))
    ni = xi_.shape[0]
        
    noise_std = noise_std_func(xi_)
    xi = np.hstack((xi_, noise_std**2))
    
    # Evaluation results with heteroscedastic Gaussian noise
    u = np.random.normal(size=(xi.shape[0], 1))
    zi = gp.misc.testfunctions.twobumps(xi_).reshape((-1, 1)) + noise_std * u

    return xt, zt, xi, zi

xt, zt, xi, zi = generate_data()

## -- model specification


def kernel_ii_or_tt(x, param, pairwise=False):
    """Covariance of the observations at points given by x
    Parameters
    ----------
    x : ndarray(n, d + 1)
        Data points in dimension d. The last column is the noise
        variance at the location.
    param : ndarray(1 + d)
        sigma2 and range parameters

    """
    # parameters
    p = 2
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1])
    noise_variance = jnp.exp(param[2])

    if pairwise:
        # return a vector of covariances between pretictands
        K = sigma2 * jnp.ones((x.shape[0], )) + x[:, -1]  # nx x 0
    else:
        # return a covariance matrix between observations
        xs = gp.kernel.scale(x[:, :-1], invrho)
        K = gp.kernel.distance(xs, xs)  # nx x nx
        K = sigma2 * gp.kernel.maternp_kernel(p, K) \
            + jnp.diag(x[:, -1])

    return K


def kernel_it(x, y, param, pairwise=False):
    """Covariance between observations and prediction points
    """
    p = 2
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1])

    xs = gp.kernel.scale(x[:, :-1], invrho)
    ys = gp.kernel.scale(y[:, :-1], invrho)
    if pairwise:
        # return a vector of covariances
        K = gp.kernel.distance_pairwise(xs, ys) # nx x 0
    else:
        # return a covariance matrix
        K = gp.kernel.distance(xs, ys)  # nx x ny

    K = sigma2 * gp.kernel.maternp_kernel(p, K)
    return K


def kernel(x, y, param, pairwise=False):

    if y is x or y is None:
        return kernel_ii_or_tt(x, param, pairwise)
    else:
        return kernel_it(x, y, param, pairwise)


def constant_mean(x, param):
    return jnp.ones((x.shape[0], 1))


mean = constant_mean

meanparam = None
covparam = jnp.array([math.log(0.5**2), math.log(1 / .7)])

model = gp.core.Model(mean, kernel, meanparam, covparam)

## -- prediction

zpm, zpv, lambda_t = model.predict(xi, zi, xt, return_lambdas=True)

zpv = np.maximum(zpv, 0)  # zeroes negative variances

## -- visualization

fig = gp.misc.plotutils.Figure(isinteractive=True)
fig.plot(xt[:, 0], zt, 'C0', linestyle=(0, (5, 5)), linewidth=1)
fig.plot(xi[:, 0], zi, 'rs')
fig.plotgp(xt[:, 0], zpm, zpv)
fig.xlabel('x')
fig.ylabel('z')
fig.title('data and prediction')
fig.show()


## -- generate sample paths

ni = xi.shape[0]
nt = xt.shape[0]
xixt = np.vstack((xi, xt))
xi_ind = np.arange(ni)
xt_ind = np.arange(nt) + ni

n_samplepaths = 3
zsim = model.sample_paths(xixt, n_samplepaths)

fig = gp.misc.plotutils.Figure(isinteractive=True)
fig.plot(xixt[xt_ind, 0], zsim[xt_ind])
fig.ax.set_prop_cycle(None) # reset color cycle
fig.plot(xixt[xi_ind, 0], zsim[xi_ind], 'o')
fig.title('Unconditional sample paths with simulated noisy observations')
fig.show()

## -- conditional sample paths

zpm, zpv, lambda_t = model.predict(xi, zi, xt, return_lambdas=True)
zpsim = model.conditional_sample_paths(zsim, xi_ind, zi, xt_ind, lambda_t)

# ## -- visualization

fig = gp.misc.plotutils.Figure(isinteractive=True)
fig.plot(xixt[xt_ind, 0], zt, 'C2', linewidth=1)
fig.plot(xixt[xt_ind, 0], zpsim, 'C0', linewidth=1)
fig.plot(xi[:, 0], zi, 'rs')
fig.plotgp(xt[:, 0], zpm, zpv)
fig.show()
