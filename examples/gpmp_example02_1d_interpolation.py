'''Plot and optimize the restricted negative log-likelihood

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)

'''
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp


## -- dataset


def generate_data():
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

    ni = 6
    xi = gp.misc.designs.ldrandunif(dim, ni, box)
    zi = gp.misc.testfunctions.twobumps(xi)
   
    return xt, zt, xi, zi


xt, zt, xi, zi = generate_data()

## -- model specification


def constant_mean(x, param):
    return np.ones((x.shape[0], 1))


def kernel(x, y, covparam, pairwise=False):
    p = 3
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


meanparam = None
covparam0 = None
model = gp.core.Model(constant_mean, kernel, meanparam, covparam0)

## -- automatic selection of parameters using REML

model, info = gp.kernel.select_parameters_with_reml(model, xi, zi, return_info=True)

gp.misc.modeldiagnosis.diag(model, info, xi, zi)

plot_likelihood = True
if plot_likelihood:
    gp.misc.modeldiagnosis.plot_likelihood_sigma_rho(model, info)

## -- prediction

(zpm, zpv) = model.predict(xi, zi, xt)

fig = gp.misc.plotutils.Figure(isinteractive=True)
fig.plot(xt, zt, 'k', linewidth=1, linestyle=(0, (5, 5)))
fig.plotdata(xi, zi)
fig.plotgp(xt, zpm, zpv, colorscheme='simple')
fig.xlabel('$x$')
fig.ylabel('$z$')
fig.title('Posterior GP with parameters selected by ReML')
fig.show()
