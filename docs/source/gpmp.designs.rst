gpmp.misc.designs module
========================

The ``gpmp.misc.designs`` module provides simple design-of-experiments helpers.
All design functions use NumPy arrays. Boxes are represented as ``[lower,
upper]`` with shape ``(2, d)``.

Use ``regulargrid`` for tensor grids, ``randunif`` for independent uniform
points, and ``ldrandunif``/``maximinlhs``/``maximinldlhs`` for low-discrepancy
or Latin-hypercube designs. Distance helpers such as ``mindist``, ``maxdist``,
and ``filldist_approx`` summarize point-set geometry.

.. automodule:: gpmp.misc.designs
   :members:
