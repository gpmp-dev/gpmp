'''GP regression in 1D, with noisy evaluations

This example repeats example02 with a noisy dataset.

----
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2024, CentraleSupelec
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
import gpmp.num as gnp
import gpmp as gp

def generate_data(noise_std):
    """
    Generates a noisy dataset

    Args:
        noise_std (float): Standard deviation of noise.

    Returns:
        tuple: (xt, zt, xi, zi) - target and input datasets.
    """
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gp.misc.testfunctions.twobumps(xt)

    ind = [10, 45, 100, 130, 130, 130, 131, 132, 133, 133, 133, 134, 160]
    xi = xt[ind]
    zi = zt[ind] + noise_std * np.random.randn(len(ind))

    return xt, zt, xi, zi


def constant_mean(x, _):
    return gnp.ones((x.shape[0], 1))


def kernel_ii_or_tt(x, param, pairwise=False):
    p = 2
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1]
    noise_variance = gnp.exp(param[2])

    if pairwise:
        K = sigma2 * gnp.ones((x.shape[0], ))
    else:
        K = gnp.scaled_distance(loginvrho, x, x)
        K = sigma2 * gp.kernel.maternp_kernel(p, K) + noise_variance * gnp.eye(K.shape[0])

    return K


def kernel_it(x, y, param, pairwise=False):
    p = 2
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1]

    if pairwise:
        K = gnp.scaled_distance_elementwise(loginvrho, x, y)
    else:
        K = gnp.scaled_distance(loginvrho, x, y)

    K = sigma2 * gp.kernel.maternp_kernel(p, K)
    return K


def kernel(x, y, param, pairwise=False):
    if y is x or y is None:
        return kernel_ii_or_tt(x, param, pairwise)
    else:
        return kernel_it(x, y, param, pairwise)


def main():
    noise_std = 1e-1
    xt, zt, xi, zi = generate_data(noise_std)

    mean = constant_mean
    meanparam = None

    covparam = gnp.array([
        math.log(0.5**2),
        math.log(1 / .7),
        2 * math.log(noise_std)])

    model = gp.core.Model(mean, kernel, meanparam, covparam)

    (zpm, zpv) = model.predict(xi, zi, xt)

    return xt, zt, xi, zi, zpm, zpv


def visualize(xt, zt, xi, zi, zpm, zpv):
    fig = gp.misc.plotutils.Figure(isinteractive=True)
    fig.plot(xt, zt, 'C0', linestyle=(0, (5, 5)), linewidth=1)
    fig.plot(xi, zi, 'rs')
    fig.plotgp(xt, zpm, zpv)
    fig.xylabels('x', 'z')
    fig.title('GP regression with noisy observations')
    fig.show(grid=True)


if __name__ == "__main__":
    xt, zt, xi, zi, zpm, zpv = main()
    visualize(xt, zt, xi, zi, zpm, zpv)
