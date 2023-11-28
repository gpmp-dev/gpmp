"""
GP interpolation in 1D, with noiseless data

This example demonstrates how to compute GP interpolation with unknown mean
(aka ordinary / intrinsic kriging) on a one-dimensional noiseless dataset.

A Mat'ern covariance function is used for the Gaussian Process (GP) prior.
The parameters of this covariance function are assumed to be known
(i.e., no parameter estimation is performed here).

The kriging predictor / posterior mean of the GP, interpolates the data.

----
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)
----
This example is based on the file stk_example_kb01.m from the STK at
https://github.com/stk-kriging/stk/
by Julien Bect and Emmanuel Vazquez, released under the GPLv3 license.

Original copyright notice:

   Copyright (c) 2015, 2016, 2018 CentraleSupelec
   Copyright (c) 2011-2014 SUPELEC
"""

import math
import numpy as np
import gpmp.num as gnp
import gpmp as gp


def generate_data():
    """
    Data generation
    (xt, zt): target
    (xi, zi): input dataset
    """
    # Build (xt, zt)
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gp.misc.testfunctions.twobumps(xt)

    shuffle = True
    if shuffle:
        ni = 5
        ind = np.random.choice(nt, ni, replace=False)
    else:
        ind = [10, 45, 100, 130, 160]
    xi = xt[ind]
    zi = zt[ind]

    return xt, zt, xi, zi


def zero_mean(x, param):
    return None


def constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))


def kernel_ii_or_tt(x, param, pairwise=False):
    """Covariance between observations or predictands at x."""
    p = 2
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1]
    nugget = 100 * gnp.eps

    if pairwise:
        # return a vector of covariances
        K = sigma2 * gnp.ones((x.shape[0], ))  # nx x 0
    else:
        # return a covariance matrix
        K = gnp.scaled_distance(loginvrho, x, x)  # nx x nx
        K = sigma2 * gp.kernel.maternp_kernel(p, K) + nugget * gnp.eye(K.shape[0])

    return K


def kernel_it(x, y, param, pairwise=False):
    """Covariance between observations and prediction points."""
    p = 2
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1]

    if pairwise:
        # return a vector of covariances
        K = gnp.scaled_distance_elementwise(loginvrho, x, y)  # nx x 0
    else:
        # return a covariance matrix
        K = gnp.scaled_distance(loginvrho, x, y)  # nx x ny

    K = sigma2 * gp.kernel.maternp_kernel(p, K)
    return K


def kernel(x, y, param, pairwise=False):
    if y is x or y is None:
        return kernel_ii_or_tt(x, param, pairwise)
    else:
        return kernel_it(x, y, param, pairwise)


def visualize(xt, zt, xi, zi, zpm, zpv):
    fig = gp.misc.plotutils.Figure(isinteractive=True)
    fig.plot(xt, zt, 'C0', linestyle=(0, (5, 5)), linewidth=1.0)
    fig.plotdata(xi, zi)
    fig.plotgp(xt, zpm, zpv)
    fig.xylabels('x', 'z')
    fig.show(grid=True, legend=True, legend_fontsize=9)


def main():
    xt, zt, xi, zi = generate_data()
    mean = constant_mean
    meanparam = None

    covparam = gnp.array([math.log(0.5**2),    # log(sigma2)
                          math.log(1 / .7)])   # log(1/rho)

    model = gp.core.Model(mean, kernel, meanparam, covparam)

    zpm, zpv = model.predict(xi, zi, xt)
    visualize(xt, zt, xi, zi, zpm, zpv)


if __name__ == "__main__":
    main()
