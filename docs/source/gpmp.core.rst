gpmp.core module
================

The ``core`` module defines :class:`gpmp.core.Model`, the array-based GP
model object. Arrays are backend-native objects provided by :mod:`gpmp.num`.
With the NumPy backend they are NumPy arrays; with the torch backend they are
PyTorch tensors. The core module does not perform parameter selection by
itself; construct a model here, then use :mod:`gpmp.kernel` to choose
covariance parameters when needed.

Use this module when you already have a mean function, a covariance function,
and covariance parameters, or when you want direct access to likelihoods,
prediction, leave-one-out diagnostics, and sample paths.

Model construction contract
---------------------------

``Model(mean, covariance, meanparam=None, covparam=None, meantype="linear_predictor")``
expects:

* ``mean(x, meanparam)`` returning either a vector of mean values or a matrix
  of linear-predictor basis functions, depending on ``meantype``.
* ``covariance(x, y, covparam, pairwise=False)`` returning a covariance
  matrix. ``x`` has shape ``(n, d)``. ``y`` has shape ``(m, d)`` or is
  ``None``. If ``pairwise=True``, return elementwise covariances.
* ``covparam`` as a one-dimensional covariance-parameter vector.

For backend-independent models, write ``mean`` and ``covariance`` with
``gpmp.num`` operations and constructors, not direct NumPy or torch calls.

Main call sequence
------------------

1. Build ``model = gp.core.Model(mean, covariance, meantype=...)``.
2. Set or select ``model.covparam``.
3. Call ``zpm, zpv = model.predict(xi, zi, xt)``.
4. Use ``zloom, zloov, eloo = model.loo(xi, zi)`` for leave-one-out checks.

Common return shapes
--------------------

* ``predict`` returns posterior mean and variance at ``xt``. With the default
  ``convert_out=True``, results are converted to NumPy arrays. With
  ``convert_out=False``, results remain backend-native objects.
* ``loo`` returns leave-one-out means, variances, and errors at ``xi``.
* Likelihood methods return scalar objective values.

.. automodule:: gpmp.core
   :members:
   :exclude-members: Model

.. autoclass:: gpmp.core.Model
   :members:
