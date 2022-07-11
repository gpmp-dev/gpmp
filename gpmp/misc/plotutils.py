## --------------------------------------------------------------
# Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
# Copyright (c) 2022, CentraleSupelec
# License: GPLv3 (see LICENSE)
## --------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive


class Figure:
    """
    Figures manager class

	Attributes
    ----------

    Methods
    -------
    
    """

    def __init__(self, nrows=1, ncols=1, isinteractive=True, **kargs):

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

        self.fig = plt.figure(**kargs)

        self.nrows = nrows
        self.ncols = ncols
        self.axes = []
        for i in range(nrows*ncols):
            self.axes.append(self.fig.add_subplot(nrows, ncols, i+1))
        self.ax = self.axes[0]

    def subplot(self, i):
        self.ax = self.axes[i-1]

    def show(self):
        plt.show()

    def plot(self, x, z, *args, **kargs):
        self.ax.plot(x, z, *args, **kargs)

    def plotdata(self, x, z):
        self.ax.plot(x, z,'rs', markerfacecolor='none', markersize=6)

    def xlabel(self, s):
        self.ax.set_xlabel(s)

    def ylabel(self, s):
        self.ax.set_ylabel(s)

    def title(self, s):
        self.ax.set_title(s)

    def legend(self):
        self.ax.legend()

    def plotgp(self,
               x,
               mean,
               variance,
               colorscheme='default',
               ax=None,
               fignum=None,
               **kwargs):
        mean = mean.flatten()
        x = x.flatten()

        ''' coverage intervals
           norminv (1 - 0.05/2)  = 1.959964
           norminv (1 - 0.01/2)  = 2.575829
           norminv (1 - 0.001/2) = 3.290527
        '''
        
        if colorscheme == 'bw':
            mcol = '#000000'
            mwidth = 2.0
            edgecol='#000000'
            delta0 = [ 1.959964 ]
            fillcol = [ '#F2F2F2' ]
            alpha = 0.0
            drawulb=True
        if colorscheme == 'simple':
            mcol = '#F2404C'
            mwidth = 2.0
            delta0 = [ 1.959964 ]
            fillcol = [ '#BFBFBF' ]
            alpha = 0.8
            kwargs['linewidth'] = 0.5
            drawulb=False
        else:
            mcol = '#F2404C'
            mwidth = 2.0
            delta0 = [ 3.290527, 2.575829, 1.959964 ]
            fillcol = [ '#F2F2F2', '#D8D8D8', '#BFBFBF' ]
            alpha = 0.8
            kwargs['linewidth'] = 0.5
            drawulb=False

            
        # mean
        self.ax.plot(x, mean, mcol, linewidth=mwidth)
            
        for i in range(len(delta0)):
            kwargs['alpha'] = alpha

            lower = mean - delta0[i] * np.sqrt(variance.flatten())
            upper = mean + delta0[i] * np.sqrt(variance.flatten())

            self.ax.fill(np.hstack((x, x[::-1])),
                         np.hstack((upper, lower[::-1])),
                         color=fillcol[i],
                         **kwargs)

            if drawulb:
                self.ax.plot(x, upper, color=edgecol, linestyle='dashed', dashes=(10, 8), linewidth=0.5)
                self.ax.plot(x, lower, color=edgecol, linestyle='dashed', dashes=(10, 8), linewidth=0.5)


def crosssections(model, xi, zi, box, ind_i, ind_dim, nt=100):
    """Display "cross-section" predictions at points xi[ind_i] along
    dimensions specified in ind_dim

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
            t = np.sort(np.concatenate((
                np.linspace(box[0, d], box[1, d], nt-1),
                np.array([xi[ind_i[i], ind_dim[d]]]))))
            xt = np.tile(xi[ind_i[i], :], (nt, 1))
            xt[:, ind_dim[d]] = t
            (zpm, zpv) = model.predict(xi, zi, xt)
            zpv = np.maximum(zpv, 0)
            fig.subplot(num_crosssections * d + i + 1)
            fig.plotgp(t, zpm, zpv)
            fig.plot(xi[ind_i[i], ind_dim[d]]*np.array([1, 1]), fig.ax.get_ylim())
            if i == 0:
                fig.ax.set_ylabel('z along x_{:d}'.format(d+1))
            if d == 0:
                fig.ax.set_title('cross section {:d}'.format(i+1))


def plot_loo(zi, zloom, zloov):
    """LOO plot

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

    fig.ax.errorbar(zi, zloom, 1.96 * np.sqrt(zloov), fmt='ko', ls='None')
    fig.ax.set_xlabel('true values'), plt.ylabel('predicted')
    fig.ax.set_title('LOO predictions with 95% coverage intervals')
    (xmin, xmax), (ymin, ymax) = fig.ax.get_xlim(), fig.ax.get_ylim()
    xmin = min(xmin, ymin)
    xmax = max(xmax, ymax)
    fig.ax.plot([xmin, xmax], [xmin, xmax], '--')
    fig.show()
