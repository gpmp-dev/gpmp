# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2023, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
import time
import math
import numpy as np
import gpmp.num as gnp
import gpmp as gp
from gpmp.misc.dataframe import DataFrame, ftos
import matplotlib.pyplot as plt


def diag(model, info_select_parameters, xi, zi):
    """Run model diagnosis and display the results.

    Parameters
    ----------
    model : object
        GP Model object.
    info_select_parameters : object
        Information object containing the parameter selection process.
    xi : array-like
        Input data matrix.
    zi : array-like
        Output data matrix.
    """
    md = modeldiagnosis_init(model, info_select_parameters)
    model_diagnosis_disp(md, xi, zi)


def perf(model, xi, zi, loo=True, loo_res=None, xtzt=None, zpmzpv=None):
    perf = compute_performance(model, xi, zi, loo, loo_res, xtzt, zpmzpv)
    perf_disp = perf
    try:
        perf_disp.pop("loo_pit")
    except:
        pass
    try:
        perf_disp.pop("test_pit")
    except:
        pass
    print("Prediction performances")
    print("-----------------------")
    pretty_print_dictionnary(perf_disp)


def modeldiagnosis_init(model, info):
    """Build model diagnosis based on the provided model and information.

    Parameters
    ----------
    model : object
        Model object.
    info : object
        Information object containing the parameter selection process.

    Returns
    -------
    dict
        Model diagnosis information.
    """
    md = {
        "optim_info": info,
        "param_selection": {},
        "parameters": {},
        "loo": {},
        "data": {},
    }

    md["param_selection"] = {
        "cvg_reached": info.success,
        "optimal_val": info.best_value_returned,
        "n_evals": info.nfev,
        "time": info.total_time,
        "initial_val": float(info.selection_criterion(info.initial_params)),
        "final_val": info.fun,
    }

    covparam = gnp.asarray(model.covparam)
    md["parameters"] = sigma_rho_from_covparam(covparam)

    return md


def compute_performance(model, xi, zi, loo=True, loo_res=None, xtzt=None, zpmzpv=None):
    """Compute performance metrics of the GP model.

    Parameters
    ----------
    model : instance of GP model
        The GP model used to make predictions.
    xi : ndarray of shape (n, d)
        The input data used to fit the GP model.
    zi : ndarray of shape (n,)
        The target data used to fit the GP model.
    loo : bool, optional
        Whether or not to compute the leave-one-out (LOO) metrics. Default is True.
    loo_res : tuple, optional
        The output of the `loo` method of the GP model if already computed. Default is None.
    xtzt : tuple, optional
        The test input data and corresponding targets to be used for computing test set metrics. Default is None.
    zpmzpv : tuple, optional
        The predicted mean and variance on the test set data. Default is None.

    Returns
    -------
    perf : dict

        A dictionary containing the computed performance metrics. The
        metrics that are computed depend on the input arguments.
        If `loo` is True, the following keys are present:
        - 'data_tss': total sum of squares in the data.
        - 'loo_press': predictive residual error sum of squares in the
          leave-one-out predictions.
        - 'loo_Q2': coefficient of determination for the leave-one-out predictions.
        - 'loo_log10ratio': logarithm of the predictive
          residual error sum of squares to the total sum of squares in
          the leave-one-out predictions.
        - 'loo_pit': probability integral transform (PIT) of the
          leave-one-out predictions.  If `xtzt` and `zpmzpv` are not
          None, the following keys are present:
        - 'test_tss': total sum of squares in the test set predictions.
        - 'test_press': predictive residual error sum of squares in
          the test set predictions.
        - 'test_Q2': coefficient of determination for the test set predictions.
        - 'test_log10ratio': logarithm of the predictive
          residual error sum of squares to the total sum of squares in
          the test set predictions.
        - 'test_pit': PIT of the test set predictions.
    """
    xi = gnp.asarray(xi)
    zi = gnp.asarray(zi)

    if loo and loo_res == None:
        zloom, zloov, eloo = model.loo(xi, zi)
    elif loo and type(loo_res) is tuple:
        zloom, zloov, eloo = loo_res

    test_set = False
    if xtzt is not None:
        test_set = True
        xt, zt = xtzt
        xt = gnp.asarray(xt)
        zt = gnp.asarray(zt)
    if test_set and zpmzpv is None:
        zpm, zpv = model.predict(xi, zi, xt)
    elif xtzt is not None and zpmzpv is not None:
        zpm, zpv = zpmzpv
        zpm = gnp.asarray(zpm)
        zpv = gnp.asarray(zpv)

    perf = {}

    if loo:
        # total sum of squares
        perf["data_tss"] = gnp.norm(zi - gnp.mean(zi), ord=2)
        # Predictive residual Error sum of squares
        perf["loo_press"] = gnp.norm(eloo, ord=2)
        # Q2
        perf["loo_Q2"] = 1 - perf["loo_press"] / perf["data_tss"]
        # L2err
        perf["loo_log10ratio"] = gnp.log10(perf["loo_press"] / perf["data_tss"])
        # PIT
        perf["loo_pit"] = gnp.normal.cdf(zi, loc=zloom, scale=gnp.sqrt(zloov))

    if test_set:
        perf["test_tss"] = gnp.norm(zt - gnp.mean(zt), ord=2)
        perf["test_press"] = gnp.norm(zpm - zt, ord=2)
        perf["test_Q2"] = 1 - perf["test_press"] / perf["test_tss"]
        perf["test_log10ratio"] = gnp.log10(perf["test_press"] / perf["test_tss"])
        perf["test_pit"] = gnp.normal.cdf(zt, loc=zpm, scale=gnp.sqrt(zpv))

    return perf


def plot_pit_ecdf(pit, fig=None):
    """Plot the empirical cumulative distribution function (ECDF) of a Probability
    Integral Transform (PIT) vector.

    Parameters
    ----------
    pit : gnp.ndarray, shape (n,)
        An array of PIT values.
    fig : matplotlib.figure.Figure, optional
        Matplotlib figure object to plot the PIT ECDF on. If None, a
        new figure is created. Default is None.

    Returns
    -------
    None
    """
    n = pit.shape[0]
    p = gnp.concatenate((gnp.array([0]), gnp.linspace(0, 1, n)))
    pit_sorted = gnp.concatenate((gnp.array([0.0]), gnp.sort(pit)))

    if fig is None:
        plt.figure()
    plt.step(pit_sorted, p)
    plt.plot([0, 1], [0, 1])
    plt.title("PIT (Probability Integral Transform) ECDF")
    plt.show()


def model_diagnosis_disp(md, xi, zi):
    """Display model diagnosis information.

    Parameters
    ----------
    md : dict
        Model diagnosis information.
    xi : array-like
        Input data matrix.
    zi : array-like
        Output data matrix.
    """
    print("Model diagnosis")
    print("----------------")
    print("  ***  Parameter selection")
    pretty_print_dictionnary(md["param_selection"])

    print("  ***  Parameters")
    pretty_print_dictionnary(md["parameters"])

    print("  ***  Data")
    print("   {:>0}: {:d}".format("count", zi.shape[0]))
    print("   ----")

    param_values = np.array(list(md["parameters"].values()))

    # zi
    if zi.ndim == 1:
        rownames = ["zi"]
    else:
        rownames = [f"zi_{j}" for j in range(zi.shape[1])]

    df_zi = describe_array(zi, rownames, 1 / param_values[0])

    # xi
    rownames = [f"xi_{j}" for j in range(xi.shape[1])]
    df_xi = describe_array(xi, rownames, 1 / param_values[1:])

    # zi + xi
    print(df_zi.concat(df_xi))


def plot_2d_criterion_params(
    model,
    info,
    param_indices=(0, 1),
    param_names=None,
    criterion_name='selection criterion',
):
    """
    Plot selection criterion profile for any two parameters in covparam.

    Parameters
    ----------
    model : object
        Model object.
    info : object
        Information object containing the parameter selection process.
    param_indices : tuple, optional
        Indices of the two parameters to plot (default is (0, 1)).
    param_names : list, optional
        Names of the two parameters for labeling the axes (default is None).
    criterion_name : string, optional
        Name of the selection criterion to be displayed in the title
    """
    n = 180
    tic = time.time()
    
    def print_progress(i):
        elapsed_time = time.time() - tic
        average_time_per_iteration = elapsed_time / (i + 1)
        remaining_time = average_time_per_iteration * (n - (i + 1))
        percentage = (i + 1) / n * 100
        print(
            f"       Progress: {percentage:.2f}% | time remaining: {remaining_time:.1f}s",
            end="\r",
        )
    def print_final_time():
        elapsed_time = time.time() - tic
        print(f"       Progress: 100% complete | Total time: {elapsed_time:.3f}s")
        print(f"       number of evaluations: {n * n}")

    print(f"  ***  Computing {criterion_name} profile for plotting...")
        
    param_1_idx, param_2_idx = param_indices

    # Initialize param1 and param2 based on their indices (standard deviation or scale parameter)
    param_1_0 = math.exp(
        model.covparam[param_1_idx] / 2
        if param_1_idx == 0
        else -model.covparam[param_1_idx]
    )
    param_2_0 = math.exp(
        model.covparam[param_2_idx] / 2
        if param_2_idx == 0
        else -model.covparam[param_2_idx]
    )

    f = 4  # multiplying factor
    param_1 = np.logspace(
        math.log10(param_1_0) - math.log10(f), math.log10(param_1_0) + math.log(f), n
    )
    param_2 = np.logspace(
        math.log10(param_2_0) - math.log10(f), math.log10(param_2_0) + math.log(f), n
    )

    param_1_mesh, param_2_mesh = np.meshgrid(param_1, param_2)
    log_param_1 = (
        np.log(param_1_mesh**2) if param_1_idx == 0 else np.log(1 / param_1_mesh)
    )
    log_param_2 = (
        np.log(param_2_mesh**2) if param_2_idx == 0 else np.log(1 / param_2_mesh)
    )

    selection_criterion = info.selection_criterion
    selection_criterion_values = np.zeros((n, n))

    covparam = gnp.copy(info.covparam)
    for i in range(n):
        print_progress(i)
        for j in range(n):
            covparam = gnp.set_elem1(covparam, param_1_idx, log_param_1[i, j])
            covparam = gnp.set_elem1(covparam, param_2_idx, log_param_2[i, j])
            selection_criterion_values[i, j] = selection_criterion(covparam)

    selection_criterion_values = np.nan_to_num(selection_criterion_values, copy=False)
    print_final_time()

    shift_criterion = True
    shift = -np.min(selection_criterion_values) if shift_criterion else 0

    # Plot the selection criterion profile
    plt.figure()
    plt.contourf(
        np.log10(param_1_mesh),
        np.log10(param_2_mesh),
        np.log10(np.maximum(1e-2, selection_criterion_values + shift)),
    )
    plt.plot(
        0.5 * np.log10(np.exp(info.covparam[param_1_idx])),
        -np.log10(np.exp(info.covparam[param_2_idx])),
        "ro",
    )
    plt.plot(
        0.5 * np.log10(np.exp(info.covparam0[param_1_idx])),
        -np.log10(np.exp(info.covparam0[param_2_idx])),
        "bo",
    )

    # Define axis labels (use names if provided)
    x_label = param_names[0] if param_names else f"Parameter {param_1_idx} (log10)"
    y_label = param_names[1] if param_names else f"Parameter {param_2_idx} (log10)"

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'log10 of the {"shifted " if shift_criterion else ""}{criterion_name}')
    plt.colorbar()
    plt.show()


def plot_likelihood_sigma_rho(model, info, criterion_name="negative log restricted likelihood"):
    """
    Specific case of likelihood plotting for sigma (param 0) and rho (param 1).

    Parameters
    ----------
    model : object
        Model object.
    info : object
        Information object containing the parameter selection process.
    criterion_name : string, optional
        Name of the selection criterion to be displayed in the title
    """
    # Use the generalized function to plot sigma and rho with custom names
    plot_2d_criterion_params(
        model,
        info,
        param_indices=(0, 1),
        param_names=["sigma (log10)", "rho (log10)"],
        criterion_name=criterion_name,
    )


def sigma_rho_from_covparam(covparam):
    """Extract sigma and rho parameters from the covariance parameters.

    Parameters
    ----------
    covparam : array-like
        Covariance parameters.

    Returns
    -------
    dict
        Dictionary containing sigma and rho values.
    """
    pdict = {}
    pdict["sigma"] = gnp.exp(0.5 * covparam[0])
    for i in range(covparam.shape[0] - 1):
        k = "rho{:d}".format(i)
        v = gnp.exp(-covparam[i + 1])
        pdict[k] = v

    return pdict


def describe_array(x, rownames, normalizing_factor=None):
    """Create a DataFrame containing descriptive statistics for the given data.

    Parameters
    ----------
    x : array-like
        Input data matrix.
    rownames : list
        List of row names for the DataFrame.
    normalizing_factor : float, optional
        Normalizing factor to compute the 'delta_norm' column, by default None.

    Returns
    -------
    DataFrame
        DataFrame with descriptive statistics.
    """
    x = np.array(x)
    if normalizing_factor is None:
        n_descriptors = 5
        colnames = ["mean", "std", "min", "max", "delta"]
    else:
        n_descriptors = 6
        colnames = ["mean", "std", "min", "max", "delta", "delta_norm"]
    dim = 1 if x.ndim == 1 else x.shape[1]

    data = np.empty((dim, n_descriptors))

    data[:, 0] = np.mean(x, axis=0)
    data[:, 1] = np.std(x, axis=0)
    data[:, 2] = np.min(x, axis=0)
    data[:, 3] = np.max(x, axis=0)
    data[:, 4] = data[:, 3] - data[:, 2]

    if normalizing_factor is not None:
        data[:, 5] = data[:, 4] * normalizing_factor

    return DataFrame(data, colnames, rownames)


def pretty_print_dictionnary(d, fp=4):
    """Print a dictionary with formatted values.

    Parameters
    ----------
    d : dict
        The dictionary to be printed.
    fp : int, optional
        Number of decimal places for floating-point values, by default 4.
    """
    max_key_length = max(15, max(len(str(k)) for k in d.keys()) + 2)

    for k, v in d.items():
        if not gnp.isscalar(v):
            v = v.item()
        if isinstance(v, float):
            s = f"{{:>{max_key_length}s}}: {{:s}}"
            print(s.format(k, ftos(v, fp)))
        else:
            print(f"{k:>{max_key_length}s}: {v}")
