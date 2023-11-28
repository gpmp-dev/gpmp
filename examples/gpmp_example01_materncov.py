""" Plot the Matern nu = p + 1/2 kernel/covariance functions

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)
"""
import gpmp.num as gnp
import gpmp as gp

def main():
    h = gnp.linspace(-2.0, 2.0, 500)

    fig = gp.misc.plotutils.Figure()

    for p in [0, 1, 4]:
        r = gp.kernel.maternp_kernel(p, gnp.abs(h))
        fig.plot(h, r, label='p={} / nu={}/2'.format(p, 2*p+1))

    fig.title('Matern covariances')
    fig.xlabel('h')
    fig.ylabel('$k_{p+1/2}(h)$')
    fig.legend()
    fig.show(grid=True)


if __name__ == '__main__':
    main()
