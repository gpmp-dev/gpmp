"""
Plot and optimize the restricted negative log-likelihood

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2024, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import gpmp.num as gnp
import gpmp as gp
import matplotlib.pyplot as plt


def generate_data():
    """
    Data generation.

    Returns
    -------
    tuple
        (xt, zt): target data
        (xi, zi): input dataset
    """
    c = 1.0
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gp.misc.testfunctions.twobumps(xt) + c

    ni = 7
    xi = gp.misc.designs.ldrandunif(dim, ni, box)
    zi = gp.misc.testfunctions.twobumps(xi) + c

    return xt, zt, xi, zi


def constant_mean(x, param):
    return param * gnp.ones((x.shape[0], 1))


def kernel(x, y, covparam, pairwise=False):
    p = 3
    return gp.kernel.maternp_covariance(x, y, p, covparam, pairwise)


def visualize_results(xt, zt, xi, zi, zpm, zpv):
    """
    Visualize the results using gp.misc.plotutils (a matplotlib wrapper).

    Parameters
    ----------
    xt : numpy.ndarray
        Target x values
    zt : numpy.ndarray
        Target z values
    xi : numpy.ndarray
        Input x values
    zi : numpy.ndarray
        Input z values
    zpm : numpy.ndarray
        Posterior mean
    zpv : numpy.ndarray
        Posterior variance
    """
    fig = gp.misc.plotutils.Figure(isinteractive=True)
    fig.plot(xt, zt, "k", linewidth=1, linestyle=(0, (5, 5)))
    fig.plotdata(xi, zi)
    fig.plotgp(xt, zpm, zpv, colorscheme="bw")
    fig.xylabels("$x$", "$z$")
    fig.title("Posterior GP with parameters selected by ML")
    fig.show(xlim=[-1.0, 1.0])


def main():
    xt, zt, xi, zi = generate_data()

    meanparam0 = None
    covparam0 = None
    model = gp.core.Model(
        constant_mean, kernel, meanparam0, covparam0, meantype="parameterized"
    )

    # Parameter initial guess
    (
        meanparam0,
        covparam0,
    ) = gp.kernel.anisotropic_parameters_initial_guess_constant_mean(model, xi, zi)

    param0 = gnp.concatenate((meanparam0, covparam0))

    # selection criterion
    nll, dnll = gp.kernel.make_selection_criterion_with_gradient(
        model,
        gp.kernel.negative_log_likelihood,
        xi,
        zi,
        parameterized_mean=True,
        meanparam_len=1,
    )

    param_ml, info = gp.kernel.autoselect_parameters(
        param0, nll, dnll, silent=False, info=True
    )

    model.meanparam = gnp.asarray(param_ml[0])
    model.covparam = gnp.asarray(param_ml[1:])

    info["covparam0"] = param0[1:]
    info["covparam"] = param_ml[1:]
    info["selection_criterion"] = nll

    gp.misc.modeldiagnosis.diag(model, info, xi, zi)

    # Prediction
    zpm, zpv = model.predict(xi, zi, xt)

    # Visualization
    print("\nVisualization")
    print("-------------")
    visualize_results(xt, zt, xi, zi, zpm, zpv)

    zloom, zloov, eloo = model.loo(xi, zi)
    gp.misc.plotutils.plot_loo(zi, zloom, zloov)

    return model


if __name__ == "__main__":
    model = main()
