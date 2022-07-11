'''GP interpolation in 1D, with noiseless data

This example shows how to compute GP interpolation with unknown mean
(aka ordinary / intrinsic kriging) on a one-dimensional noiseless dataset.

A Matérn covariance function is used for the Gaussian Process (GP)
prior.  The parameters of this covariance function are assumed to be
known (i.e., no parameter estimation is performed here).

The kriging predictor / posterior mean of the GP, interpolates the
data

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


#-- dataset


def generate_data():
    '''
    Data generation
    (xt, zt): target
    (xi, zi): input dataset
    '''
    # simply use
    # xt = np.expand_dims(np.linspace(-1, 1, nt), axis=1)
    # or build the regular grid using gp.misc.designs.regulargrid as follows
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gp.misc.testfunctions.twobumps(xt)

    # ind = np.arange(nt)
    # np.random.shuffle(ind)
    # ni = 5
    # ind = ind[0:ni]
    ind = [10, 45, 100, 130, 160]
    xi = xt[ind]
    zi = zt[ind]

    return xt, zt, xi, zi


xt, zt, xi, zi = generate_data()

#-- model specification


def zero_mean(x, param):
    return None


def constant_mean(x, param):
    return jnp.ones((x.shape[0], 1))


mean = constant_mean


def kernel(x, y, param, pairwise=False):

    p = 2
    sigma2 = jnp.exp(param[0])
    invrho = jnp.exp(param[1])
    nugget = 100 * gp.eps

    if y is x or y is None:
        if pairwise:
            K = sigma2 * jnp.ones((x.shape[0], ))  # nx x 0
        else:
            xs = gp.kernel.scale(x, invrho)
            K = gp.kernel.distance(xs, xs)  # nx x nx
            K = sigma2 * gp.kernel.maternp_kernel(p, K) \
                + nugget * jnp.eye(K.shape[0])
    else:
        xs = gp.kernel.scale(x, invrho)
        ys = gp.kernel.scale(y, invrho)
        if pairwise:
            K = gp.kernel.distance_pairwise(xs, ys) # nx x 0
        else:
            K = gp.kernel.distance(xs, ys)  # nx x ny
            
        K = sigma2 * gp.kernel.maternp_kernel(p, K)

    return K


meanparam = None
covparam = jnp.array([math.log(0.5**2),    # log(sigma2)
                      math.log(1 / .7)])   # log(1/rho)

model = gp.core.Model(mean, kernel, meanparam, covparam)

#-- prediction

(zpm, zpv) = model.predict(xi, zi, xt)

zpv = np.maximum(zpv, 0)  # zeroes negative variances

#-- visualization

fig = gp.misc.plotutils.Figure(isinteractive=True)
fig.plot(xt, zt, 'C0', linewidth=0.5)
fig.plotdata(xi, zi)
fig.plotgp(xt, zpm, zpv)
fig.show()
