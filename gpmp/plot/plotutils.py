# gpmp/plot/plotutils.py
# --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022-2026, CentraleSupelec
# License: GPLv3 (see LICENSE)
# --------------------------------------------------------------
"""
Plotting utilities for GPmp.
"""

import sys
import numpy as np
import gpmp.num as gnp
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import interactive


class Figure:
    """Figures manager class.

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(self, nrows=1, ncols=1, isinteractive=True, boxoff=True, **kargs):
        # Check if we run in interpreter mode
        # https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode
        self.interpreter = False
        try:
            if sys.ps1:
                self.interpreter = True
        except AttributeError:
            self.interpreter = False
            if sys.flags.interactive:
                self.interpreter = True

        if isinteractive & self.interpreter:
            interactive(True)

        self.boxoff = boxoff

        self.fig = plt.figure(**kargs)

        self.nrows = nrows
        self.ncols = ncols
        self.axes = []
        for i in range(nrows * ncols):
            self.axes.append(self.fig.add_subplot(nrows, ncols, i + 1))
        self.ax = self.axes[0]
        if self.boxoff:
            self.set_boxoff()

    def set_boxoff(self):
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["top"].set_visible(False)
        self.ax.tick_params(direction="in")

    def subplot(self, i):
        self.ax = self.axes[i - 1]
        if self.boxoff:
            self.set_boxoff()

    def show(self, grid=None, legend=None, legend_fontsize=None, xlim=None):
        if grid:
            self.grid()
        if legend and legend_fontsize is not None:
            self.legend(fontsize=legend_fontsize)
        elif legend:
            self.legend()
        if xlim is not None:
            self.xlim(xlim)
        plt.show()

    def plot(self, x, z, *args, **kargs):
        self.ax.plot(x, z, *args, **kargs)

    def plotdata(self, x, z, label='data'):
        self.ax.plot(x, z, "rs", markerfacecolor="none", markersize=6, label=label)

    def xlabel(self, s):
        self.ax.set_xlabel(s)

    def ylabel(self, s):
        self.ax.set_ylabel(s)

    def xylabels(self, sx="", sy=""):
        self.ax.set_xlabel(sx)
        self.ax.set_ylabel(sy)

    def title(self, s):
        self.ax.set_title(s)

    def legend(self, **kwargs):
        self.ax.legend(**kwargs)

    def grid(
        self,
        visible=True,
        which="major",
        linestyle=(0, (1, 5)),
        linewidth=0.5,
        **kwargs
    ):
        self.ax.grid(visible, which, linestyle=linestyle, linewidth=linewidth, **kwargs)

    def xlim(self, new_limits=None):
        if new_limits is None:
            return self.ax.get_xlim()
        else:
            self.ax.set_xlim(new_limits)
            return new_limits

    def ylim(self, new_limits=None):
        if new_limits is None:
            return self.ax.get_ylim()
        else:
            self.ax.set_ylim(new_limits)
            return new_limits

    def axhline(self, y, **kwargs):
        self.ax.axhline(y, **kwargs)

    def axvline(self, x, **kwargs):
        self.ax.axvline(x, **kwargs)
        
    def plotgp(
        self,
        x,
        mean,
        variance,
        colorscheme="default",
        rgb_hue=None,
        ax=None,
        fignum=None,
        mean_label="posterior mean",
        show_mean_label=True,
        ci=[0.95, 0.99, 0.999],  # CI levels
        ci_labels=["CI 95%", "CI 99%", "CI 99.9%"],
        show_ci_labels=True,
        **kwargs
    ):
        """Coverage intervals.

        norminv (1 - 0.05/2)  = 1.959964
        norminv (1 - 0.01/2)  = 2.575829
        norminv (1 - 0.001/2) = 3.290527
        """
        if not show_mean_label:
            mean_label = ""
        if not show_ci_labels:
            ci_labels = [""] * 3

        mean = mean.flatten()
        x = x.flatten()

        # Compute delta values using scipy.stats.norm.ppf for the given CI levels
        delta0 = [stats.norm.ppf((1 + level) / 2) for level in ci]

        if colorscheme == "hue":
            hex_code = "#" + "".join([format(i, "02x") for i in rgb_hue])
            mcol = hex_code
            mwidth = 2.0
            delta0 = [delta0[0]]
            ci_labels = [ci_labels[0]]
            fillcol = [hex_code]
            alpha = 0.5
            kwargs["linewidth"] = 0.5
            drawulb = False
        if colorscheme == "bw":
            mcol = "#000000"
            mwidth = 2.0
            edgecol = "#000000"
            delta0 = [delta0[0]]
            ci_labels = [ci_labels[0]]
            fillcol = ["#F2F2F2"]
            alpha = 0.0
            drawulb = True
        if colorscheme == "simple":
            mcol = "#F2404C"
            mwidth = 2.0
            delta0 = [delta0[0]]
            ci_labels = [ci_labels[0]]
            fillcol = ["#BFBFBF"]
            alpha = 0.8
            kwargs["linewidth"] = 0.5
            drawulb = False
        if colorscheme == "default":
            mcol = "#F2404C"
            mwidth = 2.0
            delta0 = delta0[::-1]
            ci_labels = ci_labels[::-1]
            fillcol = ["#F2F2F2", "#D8D8D8", "#BFBFBF"]
            alpha = 0.8
            kwargs["linewidth"] = 0.5
            drawulb = False

        # mean
        self.ax.plot(x, mean, mcol, linewidth=mwidth, label=mean_label)

        for i, delta in enumerate(delta0):
            kwargs["alpha"] = alpha

            lower = mean - delta * np.sqrt(variance.flatten())
            upper = mean + delta * np.sqrt(variance.flatten())

            self.ax.fill(
                np.hstack((x, x[::-1])),
                np.hstack((upper, lower[::-1])),
                color=fillcol[i],
                label=ci_labels[i],
                **kwargs
            )

            if drawulb:
                self.ax.plot(
                    x,
                    upper,
                    color=edgecol,
                    linestyle="dashed",
                    dashes=(10, 8),
                    linewidth=0.5,
                )
                self.ax.plot(
                    x,
                    lower,
                    color=edgecol,
                    linestyle="dashed",
                    dashes=(10, 8),
                    linewidth=0.5,
                )


def crosssections(
    model,
    xi,
    zi,
    box,
    ind_i=None,
    ind_dim=None,
    nt=100,
    show_data=True,
    figsize=None,
):
    """Display prediction cross sections.

    Each cross section starts from one anchor observation and sweeps one
    input coordinate through its range while keeping all other coordinates
    fixed. The posterior mean and coverage intervals are plotted along the
    slice.

    Parameters
    ----------
    model : object
        GP model with a ``predict(xi, zi, xt)`` method.
    xi : array-like
        Observation points, shape ``(n, d)``.
    zi : array-like
        Scalar observations, shape ``(n,)`` or ``(n, 1)``.
    box : array-like
        Input domain bounds, shape ``(2, d)``.
    ind_i : sequence of int, int, {"min", "max"}, optional
        Anchor observation indices. If ``"min"`` or ``"max"``, use the
        observation with the smallest or largest scalar value. If ``None``,
        use ``"min"``.
    ind_dim : sequence of int or int, optional
        Input dimensions to sweep. If ``None``, sweep all dimensions.
    nt : int, optional
        Number of points per cross section, by default 100.
    show_data : bool, optional
        If True, plot observations projected onto each swept coordinate and
        highlight the anchor observation.
    figsize : tuple of float, optional
        Matplotlib figure size. If ``None``, choose a size from the number
        of swept dimensions and anchor observations.

    Returns
    -------
    Figure
        GPmp figure object.
    """
    xi_np = np.asarray(gnp.to_np(xi))
    zi_np = np.asarray(gnp.to_np(zi))
    box = np.asarray(box, dtype=float)
    nt = int(nt)

    if xi_np.ndim != 2:
        raise ValueError("xi must have shape (n, d).")
    if box.shape != (2, xi_np.shape[1]):
        raise ValueError("box must have shape (2, d).")
    if zi_np.shape[0] != xi_np.shape[0] or zi_np.size != xi_np.shape[0]:
        raise ValueError("zi must be scalar-valued with shape (n,) or (n, 1).")
    if nt < 2:
        raise ValueError("nt must be >= 2.")

    zi_vec = zi_np.reshape(-1)

    if ind_i is None:
        ind_i = "min"
    if isinstance(ind_i, str):
        if ind_i == "min":
            ind_i = [int(np.nanargmin(zi_vec))]
        elif ind_i == "max":
            ind_i = [int(np.nanargmax(zi_vec))]
        else:
            raise ValueError("ind_i must be None, 'min', 'max', an int, or a sequence.")
    elif np.isscalar(ind_i):
        ind_i = [int(ind_i)]
    else:
        ind_i = [int(i) for i in ind_i]

    if ind_dim is None:
        ind_dim = list(range(xi_np.shape[1]))
    elif np.isscalar(ind_dim):
        ind_dim = [int(ind_dim)]
    else:
        ind_dim = [int(d) for d in ind_dim]

    num_crosssections = len(ind_i)
    num_dims = len(ind_dim)

    if figsize is None:
        figsize = (4.8 * num_crosssections, 2.4 * num_dims)

    fig = Figure(num_dims, num_crosssections, figsize=figsize)

    for i in range(num_crosssections):
        anchor_idx = ind_i[i]
        if anchor_idx < 0 or anchor_idx >= xi_np.shape[0]:
            raise IndexError("ind_i contains an out-of-bounds observation index.")
        for d in range(num_dims):
            dim_idx = ind_dim[d]
            if dim_idx < 0 or dim_idx >= xi_np.shape[1]:
                raise IndexError("ind_dim contains an out-of-bounds dimension index.")

            t = np.sort(
                np.concatenate(
                    (
                        np.linspace(box[0, dim_idx], box[1, dim_idx], nt - 1),
                        np.array([xi_np[anchor_idx, dim_idx]]),
                    )
                )
            )
            xt = np.tile(xi_np[anchor_idx, :], (nt, 1))
            xt[:, dim_idx] = t
            (zpm, zpv) = model.predict(xi, zi, gnp.asarray(xt))
            zpm = np.asarray(gnp.to_np(zpm)).reshape(-1)
            zpv = np.maximum(np.asarray(gnp.to_np(zpv)).reshape(-1), 0.0)

            fig.subplot(num_crosssections * d + i + 1)
            first_panel = i == 0 and d == 0
            fig.plotgp(
                t,
                zpm,
                zpv,
                show_mean_label=first_panel,
                show_ci_labels=first_panel,
            )
            if show_data:
                fig.ax.plot(
                    xi_np[:, dim_idx],
                    zi_vec,
                    "ko",
                    alpha=0.25,
                    markersize=3,
                    label="projected observations" if first_panel else None,
                )
                fig.ax.plot(
                    xi_np[anchor_idx, dim_idx],
                    zi_vec[anchor_idx],
                    "ro",
                    markersize=5,
                    label="anchor" if first_panel else None,
                )
            fig.ax.axvline(
                xi_np[anchor_idx, dim_idx],
                color="k",
                linestyle=":",
                linewidth=1,
            )
            fig.grid()
            fig.ax.set_xlabel(r"$x_{:d}$".format(dim_idx))
            if i == 0:
                fig.ax.set_ylabel(r"$z$ along $x_{:d}$".format(dim_idx))
            if d == 0:
                fig.ax.set_title("cross section {:d}".format(i + 1))
            if first_panel and show_data:
                fig.ax.legend(fontsize=8)

    fig.fig.tight_layout()
    return fig


def plot_loo(zi, zloom, zloov):
    """LOO plot.

    Parameters
    ----------
    zi : _type_
        _description_
    zloom : _type_
        _description_
    zloov : _type_
        _description_
    """
    fig = Figure()
    zi, zloom, zloov = gnp.to_np(zi), gnp.to_np(zloom), gnp.to_np(zloov)
    fig.ax.errorbar(zi, zloom, 1.96 * np.sqrt(zloov), fmt="ko", ls="None")
    fig.ax.set_xlabel("true values"), plt.ylabel("predicted")
    fig.ax.set_title("LOO predictions with 95% coverage intervals")
    (xmin, xmax), (ymin, ymax) = fig.ax.get_xlim(), fig.ax.get_ylim()
    xmin = min(xmin, ymin)
    xmax = max(xmax, ymax)
    fig.ax.plot([xmin, xmax], [xmin, xmax], "--")
    fig.ax.grid(True, "major", linestyle=(0, (1, 5)), linewidth=0.5)
    fig.show()
