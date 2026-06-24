gpmp.kernel module
==================

The ``kernel`` package contains covariance functions, covariance-parameter
initial guesses, likelihood and posterior objective functions, and
parameter-selection wrappers. These routines operate on backend-native
:mod:`gpmp.num` arrays and on :class:`gpmp.core.Model` instances.

The conventions below apply across the package. The detailed API is split into
focused pages for covariance helpers, parameter selection, and prior terms.

Covariance parameter convention
-------------------------------

For anisotropic Matern covariances, GPmp uses the vector

``covparam = [log(sigma2), -log(rho_0), ..., -log(rho_{d-1})]``.

Here ``sigma2`` is the process variance and ``rho_j`` is the lengthscale in
coordinate ``j``. The sign convention is intentional: internally the
lengthscale coordinates are stored as ``loginvrho_j = -log(rho_j)``. Larger
``loginvrho_j`` therefore means a shorter lengthscale.

Data-source contract
--------------------

Parameter-selection functions accept either explicit arrays ``xi, zi`` or a
``dataloader``. Do not pass both. If arrays are used, ``xi`` has shape
``(n, d)`` and ``zi`` has shape ``(n,)`` or ``(n, 1)``. Arrays are converted
with ``gpmp.num.asarray`` internally when needed, and returned parameters are
backend-native objects.

Parameter-selection contract
----------------------------

Use ``select_*`` functions when passing an explicit initial covariance vector.
Use ``update_*`` functions when the current ``model.covparam`` should be used
as the optimizer start when available.

All selection helpers return ``(model, info_ret)``. If ``info=False``,
``info_ret`` is ``None``. If ``info=True``, ``info_ret`` contains the selected
``covparam``, optimizer status, objective history, and callable criteria
``selection_criterion`` and ``selection_criterion_nograd``.

If a custom selection criterion is used, it must accept backend-native
``gpmp.num`` objects and return a scalar backend object or Python scalar.

Where to look
-------------

* :doc:`gpmp.kernel.covariance` documents Matern and exponential covariance
  helpers and automatic initial guesses for covariance parameters.
* :doc:`gpmp.kernel.selection` documents likelihood objectives, gradient
  wrappers, and ML / REML / REMAP selection procedures.
* :doc:`gpmp.kernel.priors` documents prior terms and REMAP posterior
  objective functions.

.. toctree::
   :maxdepth: 2

   gpmp.kernel.covariance
   gpmp.kernel.selection
   gpmp.kernel.priors
