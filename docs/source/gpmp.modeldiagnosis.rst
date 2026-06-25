gpmp.modeldiagnosis module
==========================

The ``modeldiagnosis`` module reports parameter-selection results, predictive
performance, leave-one-out scores, selection-criterion cross sections, and
simple data summaries. Use it after parameters have been selected and
``model.covparam`` is set.

Typical inputs
--------------

* ``model``: a :class:`gpmp.core.Model` with selected ``covparam``.
* ``info``: the diagnostics object returned by ``gpmp.kernel.select_*`` with
  ``info=True``.
* ``xi, zi``: observation points and scalar observations.

Inputs can be backend-native :mod:`gpmp.num` objects. Reporting and plotting
helpers convert to NumPy internally where required by tabular display or
Matplotlib.

Main diagnostics
----------------

``diag(model, info, xi, zi)`` prints a model report. ``perf`` prints
prediction-performance metrics. ``selection_criterion_statistics`` and
``selection_criterion_statistics_fast`` summarize one-dimensional criterion
cross sections around selected covariance parameters.

The performance metrics use the following definitions. For a block of
reference values indexed by :math:`A`, ``tss`` is
:math:`\sum_{a\in A}(z_a-\bar z_A)^2`. For leave-one-out prediction,
``press`` is :math:`\sum_i (z_i - \widehat z_{-i}(x_i))^2` and ``Q2`` is
:math:`1 - \mathrm{press}/\mathrm{tss}_{\mathrm{obs}}`. For test-set
prediction, ``rss`` is :math:`\sum_t (z_t - \widehat z(x_t))^2` and ``R2`` is
:math:`1 - \mathrm{rss}/\mathrm{tss}_{\mathrm{test}}`. ``rmse`` is
:math:`\sqrt{\mathrm{sse}/n}`. ``rmse/std(z)`` divides it by the empirical
standard deviation of the reference values in the block.

Plotting functions are loaded lazily to avoid importing Matplotlib at package
import time.

.. automodule:: gpmp.modeldiagnosis
   :members:
