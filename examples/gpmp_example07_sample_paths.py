'''GP conditional sample paths

Copyright (c) 2022, CentraleSupelec
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
License: GPLv3 (see LICENSE)
'''
import math
import numpy as np
import jax.numpy as jnp
import gpmp as gp

# dataset


def generate_data():
    nt = 200
    xt = np.expand_dims(np.linspace(-1, 1, nt), axis=1)
    zt = gp.misc.testfunctions.twobumps(xt)

    ind = [10, 45, 100, 130, 160]
    xi = xt[ind]
    zi = zt[ind]

    return xt, zt, xi, zi, ind


xt, zt, xi, zi, xi_ind = generate_data()

# model specification


def kernel(x, y, covparamm, pairwise=False):
    p = 1
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


def constant_mean(x, param):
    return jnp.ones((x.shape[0], 1))


mean = constant_mean

meanparam = None
covparam = jnp.array([math.log(0.5**2), math.log(1 / .7)])

model = gp.core.Model(mean, kernel, meanparam, covparam)

# generate sample paths
n_samplepaths = 6
zsim = model.sample_paths(xt, n_samplepaths)
fig = gp.misc.plotutils.Figure(isinteractive=True)
fig.plot(xt, zsim)

# prediction
zpm, zpv, lambda_t = model.predict(xi, zi, xt, return_lambdas=True)

zpv = np.maximum(zpv, 0)  # zeroes negative variances

# conditional sample paths
zpsim = model.conditional_sample_paths(zsim, lambda_t, zi, xi_ind)

# visualization
fig = gp.misc.plotutils.Figure(isinteractive=True)
fig.plot(xt, zt, 'C2', linewidth=0.5)
fig.plot(xt, zpsim, 'C0', linewidth=0.5)
fig.plot(xi, zi, 'rs')
fig.plotgp(xt, zpm, zpv)
fig.show()
