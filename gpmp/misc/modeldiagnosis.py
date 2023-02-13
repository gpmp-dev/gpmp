# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import math
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import gpmp as gp
from gpmp.misc.dataframe import DataFrame


def print_dict(d, fp=4):
    for k, v in d.items():
        if isinstance(v, float):
            s = f'{{:>20s}}: {{:.{fp}f}}'
            print(s.format(k, v))
        else:
            print('{:>20s}: {}'.format(k, v))


def diag(model, info_select_parameters, xi, zi):
    md = model_diagnosis(model, info_select_parameters)
    disp(md, xi, zi)


def model_diagnosis(model, info):

    md = {
        'optim_info': info,
        'param_selection': {},
        'parameters': {},
        'loo': {},
        'data': {}
    }
    
    md['param_selection'] = {
        'cvg_reached': info.success,
        'n_evals': info.nfev,
        'time': info.time,
        'initial_val': float(info.selection_criterion(info.covparam0)),
        'final_val': info.fun
    }

    covparam = info['covparam']
    md['parameters'] = sigma_rho(covparam)

    return md


def disp(md, xi, zi):

    print('Model diagnosis')
    print('----------------')
    print(' * Parameter selection')
    print_dict(md['param_selection'])

    print(' * Parameters')
    print_dict(md['parameters'])

    print(' * Data')
    print('   {:>0}: {:d}'.format('count', zi.shape[0]))
    print('   ----')

    if zi.ndim == 1:
        rownames = ['zi']
    else:
        rownames = [f'zi_{j}' for j in range(zi.shape[1])]
    df_zi = describe(zi, rownames)

    rownames = [f'xi_{j}' for j in range(xi.shape[1])]
    df_xi = describe(xi, rownames)

    print(df_zi.concat(df_xi))


def describe(x, rownames):

    n_descriptors = 5
    dim = 1 if x.ndim == 1 else x.shape[1]
    data = np.empty((dim, n_descriptors))

    colnames = ['mean', 'std', 'min', 'max', 'delta']
    data[:, 0] = jnp.mean(x, axis=0)
    data[:, 1] = jnp.std(x, axis=0)
    data[:, 2] = jnp.min(x, axis=0)
    data[:, 3] = jnp.max(x, axis=0)
    data[:, 4] = data[:, 3] - data[:, 2]

    return DataFrame(data, colnames, rownames)


def sigma_rho(covparam):
    s = {}
    s['sigma'] = jnp.exp(0.5 * covparam[0])
    for i in range(covparam.size - 1):
        k = 'rho{:d}'.format(i)
        v = jnp.exp(-covparam[i+1])
        s[k] = v

    return s


def plot_likelihood_sigma_rho(model, info):
    """plot likelihood profile"""

    n = 200
    sigma_0 = math.exp(model.covparam[0] / 2)
    rho_0 = math.exp(-model.covparam[1])
    f = 4
    sigma = np.logspace(math.log10(sigma_0) - math.log10(f),
                        math.log10(sigma_0) + math.log(f), n)
    rho = np.logspace(math.log10(rho_0) - math.log10(f),
                      math.log10(rho_0) + math.log(f), n)

    sigma_mesh, rho_mesh = np.meshgrid(sigma, rho)

    selection_criterion = info.selection_criterion

    selection_criterion_values = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            covparam = np.array(
                [math.log(sigma_mesh[i, j]**2), math.log(1 / rho_mesh[i, j])])

            selection_criterion_values[i, j] = selection_criterion(covparam)

            if np.isnan(selection_criterion_values[i, j]):
                selection_criterion_values[i, j] = 0

    shift_criterion = True
    shift = - np.min(selection_criterion_values) if shift_criterion else 0

    plt.contourf(np.log10(sigma_mesh), np.log10(rho_mesh),
                 np.log10(np.maximum(1e-2,
                                     selection_criterion_values + shift)))
    plt.plot(0.5*np.log10(np.exp(info.covparam[0])),
             - np.log10(np.exp(info.covparam[1])),
             'ro')
    plt.plot(0.5*np.log10(np.exp(info.covparam0[0])),
             - np.log10(np.exp(info.covparam0[1])),
             'bo')
    plt.xlabel('sigma (log10)')
    plt.ylabel('rho (log10)')
    plt.title('log10 of the {}negative log restricted likelihood'.
              format('shifted ' if shift_criterion else ''))
    plt.colorbar()
    plt.show()
