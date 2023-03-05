"""GP Conditional Sample Paths

This script constructs a GP model using a Matérn kernel and a constant
mean function. The script generates sample paths from the GP prior,
and then generates conditional sample paths given the observed data.

Copyright (c) 2022-2023, CentraleSupelec
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
License: GPLv3 (see LICENSE)

"""

import math
import numpy as np
import gpmp.num as gnp
import gpmp as gp
import gpmp.misc.plotutils as plotutils


def generate_data():
    nt = 200
    xt = np.linspace(-1, 1, nt).reshape(-1, 1)
    zt = gp.misc.testfunctions.twobumps(xt)

    ind = [10, 45, 100, 130, 155]
    xi = xt[ind]
    zi = zt[ind]

    return xt, zt, xi, zi, ind


def kernel(x, y, covparam, pairwise=False):
    p = 2
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


def constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))


def visualization(xt, zt, zpsim, xi, zi, zpm, zpv):
    fig = plotutils.Figure(isinteractive=True)
    fig.plot(xt, zt, 'C2', linewidth=1, label='truth')
    fig.plot(xt, zpsim[:, 1], 'C0', linewidth=1, label='posterior sample paths')
    fig.plot(xt, zpsim[:, 1:], 'C0', linewidth=1)
    fig.plot(xi, zi, 'rs')
    fig.plotgp(xt, zpm, zpv)
    fig.title('Conditional sample paths')
    fig.legend()
    fig.show()


def main():
    xt, zt, xi, zi, xi_ind = generate_data()

    mean = constant_mean
    meanparam = None
    covparam = gnp.array([math.log(0.5 ** 2), math.log(1 / .7)])
    model = gp.core.Model(mean, kernel, meanparam, covparam)

    n_samplepaths = 6
    zsim = model.sample_paths(xt, n_samplepaths, method='chol')

    zpm, zpv, lambda_t = model.predict(xi, zi, xt, return_lambdas=True)
    zpsim = model.conditional_sample_paths(zsim, xi_ind, zi, gnp.arange(xt.shape[0]), lambda_t)

    visualization(xt, zt, zpsim, xi, zi, zpm, zpv)


if __name__ == "__main__":
    main()
