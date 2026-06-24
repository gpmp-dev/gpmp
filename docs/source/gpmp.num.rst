gpmp.num module
===============

The ``gpmp.num`` module is the numerical backend layer used by GPmp. It
dispatches a NumPy-like API to the backend selected for the current Python
process.

Backend object contract
-----------------------

GPmp functions operate on backend-native numerical objects:

* with ``GPMP_BACKEND=numpy``, backend arrays are NumPy ``ndarray`` objects;
* with ``GPMP_BACKEND=torch``, backend arrays are PyTorch ``Tensor`` objects.

Inputs passed to GPmp should be backend objects or objects convertible by
``gpmp.num.asarray``. Outputs are backend objects unless a function explicitly
documents conversion to NumPy. Use ``gpmp.num.to_np`` at plotting, reporting,
or external-library boundaries when a NumPy array is required.

When writing custom mean functions, covariance functions, or selection
criteria, use ``gpmp.num`` operations rather than direct ``numpy`` or ``torch``
calls if the code must be backend-independent.

Backend selection and dtype
---------------------------

The backend is selected at import time. Set ``GPMP_BACKEND`` to ``"numpy"`` or
``"torch"`` before importing GPmp, or call ``gpmp.config.set_backend(...)``
before importing ``gpmp.num``. If no backend is requested, GPmp uses PyTorch
when available and otherwise falls back to NumPy.

GPmp uses ``float64`` only. The portable dtype is available through
``gpmp.num.get_dtype()`` and backend-specific constructors default to that
dtype where applicable.

Core constructors and conversion
--------------------------------

.. automodule:: gpmp.num

.. autofunction:: gpmp.num.asarray

.. autofunction:: gpmp.num.array

.. autofunction:: gpmp.num.zeros

.. autofunction:: gpmp.num.ones

.. autofunction:: gpmp.num.full

.. autofunction:: gpmp.num.eye

.. autofunction:: gpmp.num.to_np

.. autofunction:: gpmp.num.to_scalar

.. autofunction:: gpmp.num.get_dtype

Mathematical operations
-----------------------

These wrappers use NumPy-style argument names where possible. In particular,
reductions use ``axis`` and ``ddof`` even when the torch backend is active.

.. autofunction:: gpmp.num.exp

.. autofunction:: gpmp.num.log

.. autofunction:: gpmp.num.sqrt

.. autofunction:: gpmp.num.sum

.. autofunction:: gpmp.num.mean

.. autofunction:: gpmp.num.var

.. autofunction:: gpmp.num.std

.. autofunction:: gpmp.num.cov

Linear algebra and distances
----------------------------

.. autofunction:: gpmp.num.solve

.. autofunction:: gpmp.num.solve_triangular

.. autofunction:: gpmp.num.cho_factor

.. autofunction:: gpmp.num.cho_solve

.. autofunction:: gpmp.num.cholesky_solve

.. autofunction:: gpmp.num.cholesky_inv

.. autofunction:: gpmp.num.scaled_distance

.. autofunction:: gpmp.num.scaled_distance_elementwise

Random numbers
--------------

Use ``gpmp.num.set_seed`` for backend-specific reproducibility. Random
functions return backend-native objects.

.. autofunction:: gpmp.num.set_seed

.. autofunction:: gpmp.num.rand

.. autofunction:: gpmp.num.randn

.. autofunction:: gpmp.num.choice

.. autofunction:: gpmp.num.permutation

Differentiation helpers
-----------------------

The torch backend provides automatic differentiation. The NumPy backend uses
finite-difference helpers where available. Higher-level parameter-selection
code relies on ``DifferentiableSelectionCriterion`` to interface with SciPy
optimizers.

.. autofunction:: gpmp.num.grad

.. autofunction:: gpmp.num.value_and_grad

.. autoclass:: gpmp.num.DifferentiableSelectionCriterion
   :members:

.. autoclass:: gpmp.num.BatchDifferentiableSelectionCriterion
   :members:
