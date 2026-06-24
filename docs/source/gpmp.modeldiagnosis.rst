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

Plotting functions are loaded lazily to avoid importing Matplotlib at package
import time.

.. automodule:: gpmp.modeldiagnosis
   :members:
