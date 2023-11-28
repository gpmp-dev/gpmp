"""
GP Conditional Sample Paths

This script demonstrates Gaussian Process (GP) regression with conditional sample paths
on heteroscedastic noisy data. It generates synthetic data using a predefined test function (two bumps)
and adds heteroscedastic Gaussian noise to the observed points. The script then constructs a GP
model using a Matern kernel and a constant mean function. It generates sample paths from the GP
prior, and then generates conditional sample paths given the observed data.

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
    xi_union = np.vstack((xi1, xi2))
    ni = xi_union.shape[0]

    # Calculate the noise standard deviation for each observation point
    noise_std = noise_std_func(xi_union)

    # Add the noise variance (noise_std**2) as the last column of xi
    xi = np.hstack((xi_union, noise_std**2))

    # Evaluation results with heteroscedastic Gaussian noise
    u = np.random.normal(size=(xi.shape[0], 1))
    zi = gp.misc.testfunctions.twobumps(xi_union).reshape((-1, 1)) + noise_std * u

    return xt, zt, xi, zi


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
    p = 2
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1]
    noise_variance = x[:, -1]  # Extract noise variance from the last column of x
    noise_variance = gnp.asarray(noise_variance)

    if pairwise:
        # return a vector of covariances
        K = sigma2 * gnp.ones((x.shape[0], )) + x[:, -1]
    else:
        # return a covariance matrix between observations
        K = gnp.scaled_distance(loginvrho, x[:, :-1], x[:, :-1])
        K = sigma2 * gp.kernel.maternp_kernel(p, K) + gnp.diag(noise_variance) 

    return K


def kernel_it(x, y, param, pairwise=False):
    """Covariance between observations and prediction points
    """
    p = 2
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1]

    if pairwise:
        # return a vector of covariances
        K = gnp.scaled_distance_elementwise(loginvrho, x[:, :-1], y[:, :-1])
    else:
        # return a covariance matrix
        K = gnp.scaled_distance(loginvrho, x[:, :-1], y[:, :-1])

    K = sigma2 * gp.kernel.maternp_kernel(p, K)
    return K


def kernel(x, y, param, pairwise=False):
    if y is x or y is None:
        return kernel_ii_or_tt(x, param, pairwise)
    else:
        return kernel_it(x, y, param, pairwise)


def constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))


def visualize(xt, zt, xi, zi, zpm, zpv, zsim, zpsim):
    fig = plotutils.Figure(isinteractive=True)
    fig.plot(xt[:, 0], zt, 'C0', linestyle=(0, (5, 5)), linewidth=1)
    fig.plot(xi[:, 0], zi, 'rs')
    fig.plotgp(xt[:, 0], zpm, zpv)
    fig.xylabels('x', 'z')
    fig.title('Data and Prediction, no sample paths')
    fig.show(grid=True)

    ni = xi.shape[0]
    nt = xt.shape[0]
    xixt = np.vstack((xi, xt))
    xi_ind = np.arange(ni)
    xt_ind = np.arange(nt) + ni

    fig = plotutils.Figure(isinteractive=True)
    fig.plot(xixt[xt_ind, 0], zsim[xt_ind])
    fig.ax.set_prop_cycle(None)
    fig.plot(xixt[xi_ind, 0], zsim[xi_ind], 'o')
    fig.title('Unconditional sample paths, with simulated noisy observations for each sample path')
    fig.show(grid=True)

    fig = plotutils.Figure(isinteractive=True)
    fig.plot(xixt[xt_ind, 0], zt, 'C2', linewidth=1)
    fig.plot(xixt[xt_ind, 0], zpsim, 'C0', linewidth=1)
    fig.plot(xi[:, 0], zi, 'rs')
    fig.plotgp(xt[:, 0], zpm, zpv)
    fig.title('Conditional Sample Paths')
    fig.show(grid=True)


def main():
    xt, zt, xi, zi = generate_data()

    mean = constant_mean
    meanparam = None
    covparam = gnp.array([math.log(0.5**2), math.log(1 / .7)])
    model = gp.core.Model(mean, kernel, meanparam, covparam)

    zpm, zpv, lambda_t = model.predict(xi, zi, xt, return_lambdas=True)

    ni = xi.shape[0]
    nt = xt.shape[0]
    xixt = np.vstack((xi, xt))
    xi_ind = np.arange(ni)
    xt_ind = np.arange(nt) + ni

    n_samplepaths = 3
    zsim = model.sample_paths(xixt, n_samplepaths, method='svd')
    zpsim = model.conditional_sample_paths(zsim, xi_ind, zi, xt_ind, lambda_t)

    visualize(xt, zt, xi, zi, zpm, zpv, zsim, zpsim)


if __name__ == "__main__":
    main()
