"""
Gaussian processes in 2D

An anisotropic Matern covariance function is used for the Gaussian
Process (GP) prior. The parameters of this covariance function
(variance and ranges) are estimated using the Restricted Maximum
Likelihood (ReML) method.

The mean function of the GP prior is assumed to be constant and
unknown.

The function is sampled on a space-filling Latin Hypercube design, and
the data is assumed to be noiseless.

----
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2024, CentraleSupelec
License: GPLv3 (see LICENSE)
----
This example is based on the file stk_example_kb03.m from the STK at
https://github.com/stk-kriging/stk/ 
by Julien Bect and Emmanuel Vazquez, released under the GPLv3 license.

Original copyright notice:

   Copyright (c) 2015, 2016, 2018 CentraleSupelec
   Copyright (c) 2011-2014 SUPELEC
----
"""

import numpy as np
import gpmp.num as gnp
import gpmp as gp
import matplotlib.pyplot as plt


# Test function selection
def select_test_function(case_num):
    if case_num == 1:
        f = gp.misc.testfunctions.braninhoo
        dim = 2
        box = [[-5, 0], [10, 15]]
        ni = 20
    elif case_num == 2:
        f = gp.misc.testfunctions.wave
        dim = 2
        box = [[-1, -1], [1, 1]]
        ni = 40
    return f, dim, box, ni


def create_model():
    def constant_mean(x, param):
        return gnp.ones((x.shape[0], 1))

    def kernel(x, y, covparam, pairwise=False):
        p = 6
        return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)

    return gp.core.Model(constant_mean, kernel)


def main():
    case_num = 2
    f, dim, box, ni = select_test_function(case_num)

    # Compute the function on a 80 x 80 regular grid
    nt = [80, 80]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = f(xt)

    design_type = 'ld'
    if design_type == 'lhs':
        xi = gp.misc.designs.maximinlhs(dim, ni, box)
    elif design_type == 'ld':
        xi = gp.misc.designs.ldrandunif(dim, ni, box)
    zi = f(xi)

    model = create_model()

    # Parameter selection
    covparam0 = gp.kernel.anisotropic_parameters_initial_guess(model, xi, zi)
    nlrl, dnlrl = gp.kernel.make_selection_criterion_with_gradient(
        model,
        gp.kernel.negative_log_restricted_likelihood,
        xi,
        zi)
    covparam_reml, info = gp.kernel.autoselect_parameters(covparam0, nlrl, dnlrl, info=True)
    model.covparam = gnp.asarray(covparam_reml)
    gp.misc.modeldiagnosis.diag(model, info, xi, zi)

    # Prediction
    (zpm, zpv) = model.predict(xi, zi, xt)

    # Visualization
    cmap = plt.get_cmap('PiYG')
    contour_lines = 30

    fig, axes = plt.subplots(nrows=2, ncols=2)
    data = [zt, zpm, np.abs(zpm - zt), np.sqrt(zpv)]
    titles = [
        'function to be approximated',
        f'approximation from {ni} points',
        'true approx error',
        'posterior std'
    ]

    for ax, z, title in zip(axes.flat, data, titles):
        cs = ax.contourf(xt[:, 0].reshape(nt),
                         xt[:, 1].reshape(nt),
                         z.reshape(nt),
                         levels=contour_lines,
                         cmap=cmap)
        ax.plot(xi[:, 0], xi[:, 1], 'ro', label='data')
        ax.set_title(title)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.legend()
        fig.colorbar(cs, ax=ax, shrink=0.9)

    plt.show()

    # Predictions vs truth
    plt.figure()
    plt.plot(zt, zpm, 'ko')
    (xmin, xmax), (ymin, ymax) = plt.xlim(), plt.ylim()
    xmin = min(xmin, ymin)
    xmax = max(xmax, ymax)
    plt.plot([xmin, xmax], [xmin, xmax], '--')
    plt.xlabel('true values')
    plt.ylabel('predictions')
    plt.show()

    # LOO predictions
    zloom, zloov, eloo = model.loo(xi, zi)
    gp.misc.plotutils.plot_loo(zi, zloom, zloov)

    gp.misc.plotutils.crosssections(model, xi, zi, box, [0, 20], [0, 1])

if __name__ == '__main__':
    main()
