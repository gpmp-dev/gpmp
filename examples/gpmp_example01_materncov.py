''' Plot the Matern nu = p + 1/2 kernel/covariance functions

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022, CentraleSupelec
License: GPLv3 (see LICENSE)
'''
import numpy as np
import gpmp as gp

h = np.linspace(-2.0, 2.0, 400)

fig = gp.misc.plotutils.Figure()

for p in [0, 1, 4]:
    r = gp.kernel.maternp_kernel(p, np.abs(h))
    fig.plot(h, r, label='p={} / nu={}/2'.format(p, 2*p+1))

fig.xlabel('h')
fig.ylabel('$k_{p+1/2}(h)$')
fig.legend()
fig.show()
