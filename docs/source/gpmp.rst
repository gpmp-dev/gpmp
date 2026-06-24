API reference
=============

GPmp's primary modeling interface is array-based. Arrays are backend-native
objects managed by :mod:`gpmp.num`: NumPy arrays with the NumPy backend and
PyTorch tensors with the torch backend. The :mod:`gpmp.core`
module defines the GP model object, and :mod:`gpmp.kernel` provides
covariance functions and parameter-selection procedures that operate on
plain covariance-parameter vectors. The :mod:`gpmp.parameter` module is an
optional helper layer for naming, normalizing, displaying, and inspecting
these vectors. :mod:`gpmp.core` and :mod:`gpmp.kernel` do not depend on it.

Common API conventions
----------------------

Use these conventions when calling the API programmatically.

* Observation points are arrays ``xi`` with shape ``(n, d)``.
* Scalar observations are arrays ``zi`` with shape ``(n,)`` or ``(n, 1)``.
* Prediction points are arrays ``xt`` with shape ``(m, d)``.
* Covariance parameters are one-dimensional arrays. For anisotropic Matérn
  covariances, use ``covparam = [log(sigma2), -log(rho_0), ..., -log(rho_{d-1})]``.
* Selection procedures modify ``model.covparam`` and return ``(model, info)``
  when ``info=True``. With ``info=False``, they return ``(model, None)``.
* Use :mod:`gpmp.num` arrays or objects convertible by ``gpmp.num.asarray``.
  Unless a function documents a conversion, outputs are backend-native
  objects. Use ``gpmp.num.to_np`` when NumPy arrays are needed outside GPmp.
* Write custom mean, covariance, and criterion functions with :mod:`gpmp.num`
  operations when they must work with both backends.

Module order
------------

.. toctree::
   :maxdepth: 3

   gpmp.num
   gpmp.core
   gpmp.kernel
   gpmp.parameter
   gpmp.modeldiagnosis
   gpmp.mcmc
   gpmp.designs
   gpmp.plotutils   
   gpmp.testfunctions
