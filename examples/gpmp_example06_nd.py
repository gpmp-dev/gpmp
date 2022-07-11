'''Prediction of some classical test functions in dimension > 2

An anisotropic Matern covariance function is used for the Gaussian
Process (GP) prior. The parameters of this covariance function
(variance and ranges) are estimated using the Restricted Maximum
Likelihood (ReML).

The mean function of the GP prior is assumed to be constant and
unknown.

The function is sampled on a space-filling Latin Hypercube design, and
the data is assumed to be noiseless.

----
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)
'''

import numpy as np
import matplotlib.pyplot as plt
import gpmp as gp

# -- choose test case
testcase = 2
if testcase == 1:
    f = gp.misc.testfunctions.hartmann4
    dim = 4
    box = [[0] * 4, [1.0] * 4]
    ni = 40
    xi = gp.misc.designs.ldrandunif(dim, ni, box)
    nt = 1000
    xt = gp.misc.designs.ldrandunif(dim, nt, box)

if testcase == 2:
    f = gp.misc.testfunctions.hartmann6
    dim = 6
    box = [[0] * 6, [1.0] * 6]
    ni = 500
    xi = gp.misc.designs.ldrandunif(dim, ni, box)
    nt = 1000
    xt = gp.misc.designs.ldrandunif(dim, nt, box)

elif testcase == 3:
    f = gp.misc.testfunctions.borehole
    dim = 8
    box = [[0.05, 100,   63070,  990,  63.1, 700, 1120, 9855],
           [0.15, 50000, 115600, 1110, 116,  820, 1680, 12045]]
    ni = 30
    xi = gp.misc.designs.maximinldlhs(dim, ni, box)
    nt = 1000
    xt = gp.misc.designs.ldrandunif(dim, nt, box)

elif testcase == 4:
    f = gp.misc.testfunctions.detpep8d
    dim = 8
    box = [[0] * 8, [1.0] * 8]
    ni = 60
    xi = gp.misc.designs.maximinldlhs(dim, ni, box)
    nt = 1000
    xt = gp.misc.designs.ldrandunif(dim, nt, box)

# -- compute the function

zi = f(xi)
zt = f(xt)

# -- model specification


def constant_mean(x, param):
    return np.ones((x.shape[0], 1))


def kernel(x, y, covparam, pairwise=False):
    p = 10
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


model = gp.core.Model(constant_mean, kernel)

# -- parameter selection

covparam0 = gp.kernel.anisotropic_parameters_initial_guess(model, xi, zi)

nlrl, dnlrl = model.make_reml_criterion(xi, zi)
covparam_reml = gp.kernel.autoselect_parameters(covparam0, nlrl, dnlrl)

model.covparam = covparam_reml

gp.kernel.print_sigma_rho(covparam_reml)

# -- prediction

(zpm, zpv) = model.predict(xi, zi, xt)
zpv = np.maximum(zpv, 0)

# -- visualization

# predictions vs truth
plt.plot(zt, zpm, 'ko')
(xmin, xmax), (ymin, ymax) = plt.xlim(), plt.ylim()
xmin = min(xmin, ymin)
xmax = max(xmax, ymax)
plt.plot([xmin, xmax], [xmin, xmax], '--')
plt.show()

# LOO predictions
zloom, zloov, eloo = model.loo(xi, zi)
gp.misc.plotutils.plot_loo(zi, zloom, zloov)

# cross sections
gp.misc.plotutils.crosssections(model, xi, zi, box, [0, 1], np.arange(dim))
