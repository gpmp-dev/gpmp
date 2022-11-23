'''GP regression in 1D, with noisy evaluations

This example repeats example02 with a noisy dataset.

----
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)
----
This example is based on the file stk_example_kb01.m from the STK at
https://github.com/stk-kriging/stk/ 
by Julien Bect and Emmanuel Vazquez, released under the GPLv3 license.

Original copyright notice:

   Copyright (c) 2015, 2016, 2018 CentraleSupelec
   Copyright (c) 2011-2014 SUPELEC
----

'''
import math
import numpy as np
import jax.numpy as jnp
import gpmp as gp

## -- dataset


def generate_data(noise_std):
    '''
    Data generation
    (xt, zt): target
    (xi, zi): input dataset
    '''
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gp.misc.testfunctions.twobumps(xt)

    ind = [10, 45, 100, 130, 130, 130, 131, 132, 133, 133, 133, 134, 160]
    xi = xt[ind]
    zi = zt[ind] + noise_std * np.random.randn(len(ind))

    return xt, zt, xi, zi


noise_std = 1e-1
xt, zt, xi, zi = generate_data(noise_std)

## -- model specification


def zero_mean(x, param):
    return None


def constant_mean(x, param):
    return jnp.ones((x.shape[0], 1))


def kernel_ii(x, param, pairwise=False):
    """Covariance of the observations at points given by x
    """
    # parameters
    p = 2
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1])
    noise_variance = jnp.exp(param[2])

    if pairwise:
        # return a vector of covariances
        K = sigma2 * jnp.ones((x.shape[0], ))  # nx x 0
    else:
        # return a covariance matrix
        xs = gp.kernel.scale(x, invrho)
        K = gp.kernel.distance(xs, xs)  # nx x nx
        K = sigma2 * gp.kernel.maternp_kernel(p, K) \
            + noise_variance * jnp.eye(K.shape[0])

    return K


def kernel_it(x, y, param, pairwise=False):
    """Covariance between observations and prediction points
    """
    p = 2
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1])

    xs = gp.kernel.scale(x, invrho)
    ys = gp.kernel.scale(y, invrho)
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
        return kernel_ii(x, param, pairwise)
    else:
        return kernel_it(x, y, param, pairwise)


mean = constant_mean
meanparam = None

covparam = jnp.array([
    math.log(0.5**2),  # log(sigma2)
    math.log(1 / .7),  # log(1/rho)
    2 * math.log(noise_std)])  # log(noise_variance)

model = gp.core.Model(mean, kernel, meanparam, covparam)

## -- prediction

(zpm, zpv) = model.predict(xi, zi, xt)

zpv = np.maximum(zpv, 0)  # zeroes negative variances

## -- visualization

fig = gp.misc.plotutils.Figure(isinteractive=True)
fig.plot(xt, zt, 'C0', linestyle=(0, (5, 5)), linewidth=1)
fig.plot(xi, zi, 'rs')
fig.plotgp(xt, zpm, zpv)
fig.xlabel('x')
fig.ylabel('z')
fig.show()
