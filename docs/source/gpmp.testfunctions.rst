gpmp.misc.testfunctions module
==============================

The ``gpmp.misc.testfunctions`` module provides deterministic benchmark
functions used by examples and diagnostics. Inputs are NumPy arrays with shape
``(n, d)`` except for one-dimensional functions that also accept shape ``(n,)``.
Outputs are one-dimensional arrays with shape ``(n,)``.

Common choices are ``twobumps`` for one-dimensional interpolation,
``braninhoo`` for two-dimensional optimization examples, ``hartmann4`` and
``hartmann6`` for unit-hypercube tests, and ``ishigami`` for sensitivity and
screening examples.

.. automodule:: gpmp.misc.testfunctions
   :members:
