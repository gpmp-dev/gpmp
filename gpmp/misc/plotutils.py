## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
import sys
import numpy as np
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


def crosssections(model, xi, zi, box, ind_i, ind_dim, nt=100):
    """Display "cross-section" predictions at points xi[ind_i] along dimensions
    specified in ind_dim.

    Parameters
    ----------
    model : _type_
        _description_
    xi : _type_
        _description_
    zi : _type_
        _description_
    box : _type_
        _description_
    ind_i : _type_
        _description_
    ind_dim : _type_
        _description_
    nt : int, optional
        _description_, by default 100
    """
    box = np.array(box)
    num_crosssections = len(ind_i)
    num_dims = len(ind_dim)

    fig = Figure(num_dims, num_crosssections)

    for i in range(num_crosssections):
        for d in range(num_dims):
            t = np.sort(
                np.concatenate(
                    (
                        np.linspace(box[0, d], box[1, d], nt - 1),
                        np.array([xi[ind_i[i], ind_dim[d]]]),
                    )
                )
            )
            xt = np.tile(xi[ind_i[i], :], (nt, 1))
            xt[:, ind_dim[d]] = t
            (zpm, zpv) = model.predict(xi, zi, xt)
            zpv = np.maximum(zpv, 0)
            fig.subplot(num_crosssections * d + i + 1)
            fig.plotgp(t, zpm, zpv)
            fig.plot(xi[ind_i[i], ind_dim[d]] * np.array([1, 1]), fig.ax.get_ylim())
            fig.grid()
            if i == 0:
                fig.ax.set_ylabel("z along x_{:d}".format(d + 1))
            if d == 0:
                fig.ax.set_title("cross section {:d}".format(i + 1))


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
    fig.ax.errorbar(zi, zloom, 1.96 * np.sqrt(zloov), fmt="ko", ls="None")
    fig.ax.set_xlabel("true values"), plt.ylabel("predicted")
    fig.ax.set_title("LOO predictions with 95% coverage intervals")
    (xmin, xmax), (ymin, ymax) = fig.ax.get_xlim(), fig.ax.get_ylim()
    xmin = min(xmin, ymin)
    xmax = max(xmax, ymax)
    fig.ax.plot([xmin, xmax], [xmin, xmax], "--")
    fig.ax.grid(True, "major", linestyle=(0, (1, 5)), linewidth=0.5)
    fig.show()
