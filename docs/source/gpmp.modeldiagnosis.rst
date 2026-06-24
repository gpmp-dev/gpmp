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

The performance metrics use the following definitions. For targets ``z``,
``tss`` is :math:`\sum_i (z_i - \bar z)^2`. For leave-one-out prediction,
``press`` is :math:`\sum_i (z_i - \widehat z_{-i}(x_i))^2` and
``Q2`` is :math:`1 - \mathrm{press}/\mathrm{tss}`. For test-set prediction,
``rss`` is :math:`\sum_i (z_i - \widehat z(x_i))^2` and ``R2`` is
:math:`1 - \mathrm{rss}/\mathrm{tss}`. ``rmse`` is
:math:`\sqrt{\mathrm{sse}/n}`. ``rmse/std(z)`` divides it by the empirical
standard deviation of the reference values.

Plotting functions are loaded lazily to avoid importing Matplotlib at package
import time.

.. automodule:: gpmp.modeldiagnosis
   :members:
