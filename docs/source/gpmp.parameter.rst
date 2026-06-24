gpmp.parameter module
=====================

The ``gpmp.parameter`` module provides structured helpers around
covariance-parameter vectors. It assigns names, paths, normalizations, and
display metadata to parameters, especially in reports and model-container code.

This module is a convenience layer on top of the lower-level GPmp API.
The core model and kernel routines remain independent from it: they accept
plain arrays for covariance parameters and do not require ``Param`` objects.

Use this module when you need a human-readable object for a covariance vector,
not when calling :mod:`gpmp.core` or :mod:`gpmp.kernel` directly.

Normalization convention
------------------------

``Param`` stores normalized values. The supported normalizations are:

* ``"none"``: stored value equals physical value.
* ``"log"``: stored value is ``log(value)``.
* ``"log_inv"``: stored value is ``-log(value)``.

For the anisotropic Matérn convention,
``param_from_covparam_anisotropic(covparam)`` interprets
``covparam = [log(sigma2), -log(rho_0), ..., -log(rho_{d-1})]`` and attaches
names ``sigma2``, ``rho_0``, ...

Access patterns
---------------

Use ``get_by_name`` for a named scalar, ``get_by_path`` for hierarchical
groups, and ``denormalized_values`` to inspect physical values. Bounds stored
in ``Param`` are metadata for display and diagnostics. They are not enforced by
:class:`Param` itself.

.. automodule:: gpmp.parameter
   :members:
