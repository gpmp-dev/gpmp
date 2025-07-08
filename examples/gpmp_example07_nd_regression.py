"""GP regression in nd, with noisy evaluations

This example repeats example04 with observation noise added. A custom kernel
is built, whose parameters are selected by ReML.

----
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
License: GPLv3 (see LICENSE)
----

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
        ni = 80
        xi = gp.misc.designs.ldrandunif(dim, ni, box)
        nt = 1000
        xt = gp.misc.designs.ldrandunif(dim, nt, box)
        noise_std = 0.2

    elif problem == 2:
        problem_name = "Hartmann6"
        f = gp.misc.testfunctions.hartmann6
        dim = 6
        box = [[0.0] * 6, [1.0] * 6]
        ni = 200
        xi = gp.misc.designs.ldrandunif(dim, ni, box)
        nt = 1000
        xt = gp.misc.designs.ldrandunif(dim, nt, box)
        noise_std = 0.1

    elif problem == 3:
        problem_name = "Borehole"
        f = gp.misc.testfunctions.borehole
        dim = 8
        box = [
            [0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0],
            [0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0],
        ]
        ni = 50
        xi = gp.misc.designs.maximinldlhs(dim, ni, box)
        nt = 1000
        xt = gp.misc.designs.ldrandunif(dim, nt, box)
        noise_std = 10.0

    elif problem == 4:
        problem_name = "detpep8d"
        f = gp.misc.testfunctions.detpep8d
        dim = 8
        box = [[0.0] * 8, [1.0] * 8]
        ni = 120
        xi = gp.misc.designs.maximinldlhs(dim, ni, box)
        nt = 1000
        xt = gp.misc.designs.ldrandunif(dim, nt, box)
        noise_std = 5.0

    return (
        problem_name,
        f,
        dim,
        box,
        ni,
        gnp.asarray(xi),
        nt,
        gnp.asarray(xt),
        noise_std,
    )


def visualize_predictions(problem_name, zt, zpm):
    plt.figure()
    plt.plot(zt, zpm, "ko")
    (xmin, xmax), (ymin, ymax) = plt.xlim(), plt.ylim()
    xmin = min(xmin, ymin)
    xmax = max(xmax, ymax)
    plt.plot([xmin, xmax], [xmin, xmax], "--")
    plt.title(problem_name)
    plt.show()


def constant_mean(x, _):
    return gnp.ones((x.shape[0], 1))


def kernel_ii_or_tt(x, param, pairwise=False):
    p = 2
    sigma2 = gnp.exp(param[0])
    loginvrho = param[2:]
    noise_variance = gnp.exp(param[1])

    if pairwise:
        K = sigma2 * gnp.ones((x.shape[0],))
    else:
        K = gnp.scaled_distance(loginvrho, x, x)
        K = sigma2 * gp.kernel.maternp_kernel(p, K) + noise_variance * gnp.eye(
            K.shape[0]
        )

    return K


def kernel_it(x, y, param, pairwise=False):
    p = 2
    sigma2 = gnp.exp(param[0])
    loginvrho = param[2:]

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


problem = 4
problem_name, f, dim, box, ni, xi, nt, xt, noise_std = choose_test_case(problem)
zi = gnp.asarray(f(xi)) + noise_std * gnp.randn(ni)
zt = gnp.asarray(f(xt))

mean = constant_mean
meanparam = None

covparam0 = gnp.concatenate(
    (
        gnp.array([gnp.log(gnp.var(zi))]),
        gnp.array([2 * gnp.log(0.1) + gnp.log(gnp.var(zi))]),
        -gnp.log(gnp.std(xi, axis=0)).flatten(),
    )
)

model = gp.core.Model(mean, kernel, meanparam, covparam0)


# selection criterion
selection_criterion = gp.kernel.negative_log_restricted_likelihood
nlrl, dnlrl, nlrl_nograd = gp.kernel.make_selection_criterion_with_gradient(
    model, selection_criterion, xi, zi
)

# optimize parameters
covparam_reml, info = gp.kernel.autoselect_parameters(
    covparam0, nlrl, dnlrl, silent=False, info=True
)

model.covparam = gnp.asarray(covparam_reml)
info["covparam0"] = covparam0
info["covparam"] = covparam_reml
info["selection_criterion"] = nlrl

gp.misc.modeldiagnosis.diag(
    model, info, xi, zi, model_type="linear_mean_matern_anisotropic_noisy"
)

(zpm, zpv) = model.predict(xi, zi, xt)

visualize_predictions(problem_name, zt, zpm)
