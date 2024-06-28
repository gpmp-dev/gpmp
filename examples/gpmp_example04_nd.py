"""Prediction of some classical test functions in dimension > 2

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
Copyright (c) 2022-2024, CentraleSupelec
License: GPLv3 (see LICENSE)
"""
import gpmp.num as gnp
import gpmp as gp
import matplotlib.pyplot as plt


def choose_test_case(problem):
    if problem == 1:
        problem_name = "Hartmann4"
        f = gp.misc.testfunctions.hartmann4
        dim = 4
        box = [[0.0] * 4, [1.0] * 4]
        ni = 40
        xi = gp.misc.designs.ldrandunif(dim, ni, box)
        nt = 1000
        xt = gp.misc.designs.ldrandunif(dim, nt, box)

    elif problem == 2:
        problem_name = "Hartmann6"
        f = gp.misc.testfunctions.hartmann6
        dim = 6
        box = [[0.0] * 6, [1.0] * 6]
        ni = 500
        xi = gp.misc.designs.ldrandunif(dim, ni, box)
        nt = 1000
        xt = gp.misc.designs.ldrandunif(dim, nt, box)

    elif problem == 3:
        problem_name = "Borehole"
        f = gp.misc.testfunctions.borehole
        dim = 8
        box = [
            [0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0],
            [0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0],
        ]
        ni = 30
        xi = gp.misc.designs.maximinldlhs(dim, ni, box)
        nt = 1000
        xt = gp.misc.designs.ldrandunif(dim, nt, box)

    elif problem == 4:
        problem_name = "detpep8d"
        f = gp.misc.testfunctions.detpep8d
        dim = 8
        box = [[0.0] * 8, [1.0] * 8]
        ni = 60
        xi = gp.misc.designs.maximinldlhs(dim, ni, box)
        nt = 1000
        xt = gp.misc.designs.ldrandunif(dim, nt, box)

    return problem_name, f, dim, box, ni, xi, nt, xt


def constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))


def kernel(x, y, covparam, pairwise=False):
    p = 10
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


def visualize_predictions(problem_name, zt, zpm):
    plt.figure()
    plt.plot(zt, zpm, "ko")
    (xmin, xmax), (ymin, ymax) = plt.xlim(), plt.ylim()
    xmin = min(xmin, ymin)
    xmax = max(xmax, ymax)
    plt.plot([xmin, xmax], [xmin, xmax], "--")
    plt.title(problem_name)
    plt.show()


def main():
    problem = 1
    problem_name, f, dim, box, ni, xi, nt, xt = choose_test_case(problem)

    zi = f(xi)
    zt = f(xt)

    model = gp.core.Model(constant_mean, kernel)

    model, info = gp.kernel.select_parameters_with_reml(model, xi, zi, info=True)
    gp.misc.modeldiagnosis.diag(model, info, xi, zi)

    (zpm, zpv) = model.predict(xi, zi, xt)

    visualize_predictions(problem_name, zt, zpm)

    zloom, zloov, eloo = model.loo(xi, zi)
    gp.misc.plotutils.plot_loo(zi, zloom, zloov)

    gp.misc.plotutils.crosssections(model, xi, zi, box, [0, 1], list(range(dim)))

    print('')
    
    gp.misc.modeldiagnosis.perf(
        model,
        xi,
        zi,
        loo=True,
        loo_res=(zloom, zloov, eloo),
        xtzt=(xt, zt),
        zpmzpv=(zpm, zpv),
    )


if __name__ == "__main__":
    main()
