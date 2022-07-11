'''Plot and optimize the restricted negative log-likelihood

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)

'''
import math
import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp

# -- dataset


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

    ni = 5
    xi = gp.misc.designs.ldrandunif(dim, ni, box)
    zi = gp.misc.testfunctions.twobumps(xi)

    
    return xt, zt, xi, zi


xt, zt, xi, zi = generate_data()

# -- model specification


def constant_mean(x, param):
    return np.ones((x.shape[0], 1))


def kernel(x, y, covparam, pairwise=False):
    p = 3
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


meanparam = None
covparam0 = None

model = gp.core.Model(constant_mean, kernel, meanparam, covparam0)

# -- automatic selection of parameters using REML

covparam0 = gp.kernel.anisotropic_parameters_initial_guess(model, xi, zi)

nlrl, dnlrl = model.make_reml_criterion(xi, zi)

covparam_reml = gp.kernel.autoselect_parameters(covparam0, nlrl, dnlrl)

model.covparam = covparam_reml

gp.kernel.print_sigma_rho(covparam_reml)


# -- plot likelihood profile

n = 200
sigma = np.logspace(-0.9, 1.0, n)
rho = np.logspace(-1.4, 0.4, n)

sigma_mesh, rho_mesh = np.meshgrid(sigma, rho)

nlrl_values = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        covparam = np.array(
            [math.log(sigma_mesh[i, j]**2), math.log(1 / rho_mesh[i, j])])
        nlrl_values[i, j] = nlrl(covparam)
        if np.isnan(nlrl_values[i, j]):
            nlrl_values[i, j] = 0

plt.contourf(np.log10(sigma_mesh), np.log10(rho_mesh),
             np.log10(nlrl_values - np.min(nlrl_values)))
plt.plot(0.5*np.log10(np.exp(covparam_reml[0])),
         - np.log10(np.exp(covparam_reml[1])),
         'ro')
plt.plot(0.5*np.log10(np.exp(covparam0[0])),
         - np.log10(np.exp(covparam0[1])),
         'bo')
plt.xlabel('sigma (log10)')
plt.ylabel('rho (log10)')
plt.title('log10 of the negative log restricted likelihood')
plt.colorbar()
plt.show()

# -- prediction

(zpm, zpv) = model.predict(xi, zi, xt)

zpv = np.maximum(zpv, 0)  # zeroes negative variances

fig = gp.misc.plotutils.Figure(isinteractive=True)
fig.plot(xt, zt, 'k', linewidth=0.5)
fig.plotdata(xi, zi)
fig.plotgp(xt, zpm, zpv, colorscheme='simple')
fig.xlabel('$x$')
fig.ylabel('$z$')
fig.title('Posterior GP with parameters selected by ReML')
fig.show()
