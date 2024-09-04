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
    fig.plot(xt, zt, 'k', linewidth=1, linestyle=(0, (5, 5)))
    fig.plotdata(xi, zi)
    fig.plotgp(xt, zpm, zpv, colorscheme='simple')
    fig.xylabels('$x$', '$z$')
    fig.title('Posterior GP with parameters selected by ReML')
    fig.show(grid=True, xlim=[-1.0, 1.0], legend=True, legend_fontsize=9)


def main():
    xt, zt, xi, zi = generate_data()

    model = gp.core.Model(constant_mean, kernel)

    # Automatic selection of parameters using REML
    model, info = gp.kernel.select_parameters_with_reml(model, xi, zi, info=True)
    gp.misc.modeldiagnosis.diag(model, info, xi, zi)

    # Prediction
    zpm, zpv = model.predict(xi, zi, xt)

    # Visualization
    print('\nVisualization')
    print('-------------')
    plot_likelihood = True
    if plot_likelihood:
        gp.misc.modeldiagnosis.plot_likelihood_sigma_rho(model, info)
        
    visualize_results(xt, zt, xi, zi, zpm, zpv)


if __name__ == '__main__':
    main()
