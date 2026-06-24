gpmp.plot module
================

The ``gpmp.plot`` module contains Matplotlib-based helpers for common GP
figures. It accepts backend-native :mod:`gpmp.num` objects and converts them
to NumPy internally before plotting. The functions are intentionally
lightweight and return a GPmp ``Figure`` wrapper when applicable.

Main functions
--------------

* ``Figure`` wraps a Matplotlib figure and axes and exposes convenience methods
  such as ``plotgp`` for posterior means and coverage intervals.
* ``crosssections(model, xi, zi, box, ...)`` plots one-dimensional predictive
  sweep studies through selected anchor observations. It can automatically use
  the observation with minimum or maximum response via ``ind_i="min"`` or
  ``ind_i="max"`` and can show projected observation points.
* ``plot_loo(zi, zloom, zloov)`` plots leave-one-out predictions with 95%
  intervals against observed values.

.. automodule:: gpmp.plot
   :members:
