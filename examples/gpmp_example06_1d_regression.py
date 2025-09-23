'''
Gaussian process regression in 1D with noisy evaluations.

We want the posterior of the latent process (noise-free function) from noisy
observations. To distinguish both, we add an extra column to each input row:
[coords..., flag], with flag=0 for latent targets and flag=1 for noisy
observations.

The kernel uses only the coordinates for distances. For K(x, x), the latent
covariance is computed and noise variance is added on the diagonal where
flag=1. This means only noisy observations get a noise term. Cross-covariances
K(x, y) with x not equal to y ignore flags and are purely latent. With
pairwise=True, the returned vector is the diagonal of this same covariance
matrix.

Thus:
- xi (flag=1) are noisy observations, so K(xi, xi) has diagonal noise.
- xt (flag=0) are latent targets, so K(xt, xt) is noise-free.
- The cross block K(xi, xt) is noise-free.

This yields the posterior distribution of the latent process. To predict noisy
outputs at targets instead, pass xt with flag=1.

----
Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2025, CentraleSupelec
License: GPLv3 (see LICENSE)
'''
import math
import numpy as np
import gpmp.num as gnp
import gpmp as gp


def generate_data(noise_std):
    """Create a 1D dataset with noisy observed values."""
    dim = 1
    nt = 200
    box = [[-1], [1]]
    xt = gp.misc.designs.regulargrid(dim, nt, box)
    zt = gp.misc.testfunctions.twobumps(xt)

    ind = [10, 45, 100, 130, 130, 130, 131, 132, 133, 133, 133, 134, 160]
    xi = xt[ind]
    zi = zt[ind] + noise_std * np.random.randn(len(ind))
    return xt, zt, xi, zi


def _add_noise_information(x, flag):
    """Append one flag column: 0 latent, 1 noisy."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    f = np.full((n, 1), int(flag)) if np.isscalar(flag) else np.asarray(flag).reshape(n, 1)
    return np.hstack((x, f))


def _split(x):
    """Split flagged x into (coords, flag)."""
    return x[:, :-1], x[:, -1].reshape(-1)


def constant_mean(x, _):
    """Constant mean."""
    return gnp.ones((x.shape[0], 1))


def kernel_ii_or_tt(x, param, pairwise=False):
    """Latent Matern on (x,x); add diag(noise*flag) only here."""
    p = 2
    sigma2 = math.exp(param[0])
    noise_var = math.exp(param[1])
    loginvrho = param[2:]

    x_coord, flag = _split(x)

    if pairwise:
        # Prior variance; add observation noise where flag == 1.
        return sigma2 * gnp.ones((x_coord.shape[0],)) + noise_var * flag

    D = gnp.scaled_distance(loginvrho, x_coord, x_coord)
    K = sigma2 * gp.kernel.maternp_kernel(p, D)
    K = K + gnp.diag(noise_var * flag)  # noise only on observed points
    return K


def kernel_it(x, y, param, pairwise=False):
    """Latent Matern cross-covariance (no noise on any block)."""
    p = 2
    sigma2 = math.exp(param[0])
    loginvrho = param[2:]

    x_coord, _ = _split(x)
    y_coord, _ = _split(y)

    if pairwise:
        D = gnp.scaled_distance_elementwise(loginvrho, x_coord, y_coord)
    else:
        D = gnp.scaled_distance(loginvrho, x_coord, y_coord)
    return sigma2 * gp.kernel.maternp_kernel(p, D)


def kernel(x, y, param, pairwise=False):
    """Dispatch to (x,x) or (x,y)."""
    return kernel_ii_or_tt(x, param, pairwise) if (y is x or y is None) else kernel_it(x, y, param, pairwise)


def main():
    """Fit the GP and predict the latent posterior on a grid."""
    noise_std = 1e-1
    xt, zt, xi, zi = generate_data(noise_std)

    # Flag observed inputs as noisy (1), targets as latent (0)
    xi_with_side_information = _add_noise_information(xi, 1)
    xt_with_side_information = _add_noise_information(xt, 0)

    mean = constant_mean
    meanparam = None
    covparam = np.array([
        math.log(0.5 ** 2),        # log sigma^2 (signal variance)
        2.0 * math.log(noise_std), # log noise variance
        math.log(1 / 0.7),         # log 1/rho (length-scale inverse) for 1D
    ])
    
    model = gp.core.Model(mean, kernel, meanparam, covparam, meantype="linear_predictor")
    zpm, zpv = model.predict(xi_with_side_information, zi, xt_with_side_information)
    return xt, zt, xi, zi, zpm, zpv


def visualize(xt, zt, xi, zi, zpm, zpv):
    """Plot reference function, observations, and GP posterior (latent)."""
    fig = gp.misc.plotutils.Figure(isinteractive=True)
    fig.plot(xt, zt, 'C0', linestyle=(0, (5, 5)), linewidth=1)
    fig.plot(xi, zi, 'rs')
    fig.plotgp(xt, zpm, zpv)
    fig.xylabels('x', 'z')
    fig.title('GP regression with noisy evaluations (latent posterior)')
    fig.show(grid=True)


if __name__ == "__main__":
    xt, zt, xi, zi, zpm, zpv = main()
    visualize(xt, zt, xi, zi, zpm, zpv)
