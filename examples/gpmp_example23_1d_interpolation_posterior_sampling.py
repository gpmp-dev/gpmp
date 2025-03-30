"""
Demonstrates ReMAP-based GP parameter selection, posterior sampling, 
and visualization of a 1D Gaussian process model.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
License: GPLv3 (see LICENSE)
"""

import gpmp.num as gnp
import gpmp as gp
from gpmp.misc.param_posterior import sample_from_selection_criterion
import matplotlib.pyplot as plt
from matplotlib import interactive


def generate_data():
    """
    Data generation.

    Returns
    -------
    tuple
        (xt, zt): target data
        (xi, zi): input dataset
    """
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gp.misc.testfunctions.twobumps(xt)

    ni = 6
    xi = gp.misc.designs.ldrandunif(dim, ni, box)
    zi = gp.misc.testfunctions.twobumps(xi)

    return xt, zt, xi, zi


def constant_mean(x, param):
    return gnp.ones((x.shape[0], 1))


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
    fig.plotgp(xt, zpm, zpv, colorscheme="simple")
    fig.xylabels("$x$", "$z$")
    fig.title("Posterior GP with parameters selected by ReMAP")
    fig.show(grid=True, xlim=[-1.0, 1.0], legend=True, legend_fontsize=9)


def main():
    xt, zt, xi, zi = generate_data()

    model = gp.core.Model(constant_mean, kernel)

    # Automatic selection of parameters using ReMAP
    model, info = gp.kernel.select_parameters_with_remap(model, xi, zi, info=True)
    gp.misc.modeldiagnosis.diag(model, info, xi, zi)

    # Prediction
    zpm, zpv = model.predict(xi, zi, xt)

    samples, mh = sample_from_selection_criterion(
        info,
        n_steps_total=10_000,
        burnin_period=5_000,
        n_chains=2,
        show_progress=True,
    )

    # Visualization
    print("\nVisualization")
    print("-------------")
    interactive(True)
    plot_likelihood_cross_sections = True
    plot_likelihood_2d_profile = True
    if plot_likelihood_cross_sections:
        gp.misc.modeldiagnosis.plot_selection_criterion_crossections(
            info=info, delta=0.6, param_names=["sigma^2 (log)", "rho (log)"]
        )
    if plot_likelihood_2d_profile:
        gp.misc.modeldiagnosis.plot_selection_criterion_sigma_rho(
            model, info, criterion_name="log posterior"
        )

    plt.scatter(
        gnp.log10(gnp.exp(samples[0, :, 0] / 2)),
        gnp.log10(gnp.exp(-samples[0, :, 1])),
        alpha=0.2,
    )

    visualize_results(xt, zt, xi, zi, zpm, zpv)


if __name__ == "__main__":
    main()
