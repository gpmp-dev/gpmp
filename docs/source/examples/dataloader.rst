Dataloader-based parameter selection
====================================

This example shows how observations can be passed through ``Dataset`` and
``DataLoader`` objects during parameter selection. Use this form when the
selection criterion should be evaluated on batches instead of a single full
array at every call.

What this example does
----------------------

The script creates a test problem, stores observations in a ``Dataset``, wraps
it with a ``DataLoader``, and calls ``select_parameters_with_remap`` with the
``dataloader`` argument. The selected model is then used for prediction and the
preview plots predicted values against reference values.

Mathematical object
-------------------

The model and REMAP criterion are the same as in the array-based examples. The
dataloader changes how the criterion is evaluated. If batches
:math:`b_1,\ldots,b_q` have sizes :math:`n_1,\ldots,n_q`, the batch wrapper
evaluates a weighted scalar objective of the form

.. math::

   \overline J(\theta)
   =
   \frac{\sum_{\ell=1}^q n_\ell J_\ell(\theta)}
        {\sum_{\ell=1}^q n_\ell},

where :math:`J_\ell(\theta)` is the selection criterion evaluated on batch
:math:`b_\ell`. With ``batches_per_eval=0``, one criterion call uses the full
loader. With a positive ``batches_per_eval``, one criterion call uses only that
many successive batches, cycling through the loader.

Outputs
-------

The displayed quantities are GP predictions and reference values at test points.
Points near the diagonal indicate accurate predictions. Systematic deviations
from the diagonal suggest bias, poor covariance parameters, or insufficient
observations.

API points
----------

* Selection helpers accept either explicit ``xi, zi`` arrays or a
  ``dataloader``. Do not pass both.
* ``DataLoader`` controls batching. The selection criterion still returns a
  scalar objective for optimization.
* The selected ``model.covparam`` is used normally by ``model.predict`` after
  batched selection.

.. jupyter-execute::
   :hide-code:

   from examples import gpmp_example30_dataloader as ex
   from gpmp.dataloader import Dataset, DataLoader

   problem_name, f, dim, box, ni, xi, nt, xt = ex.choose_test_case(1, ni=200)
   zi = f(xi)
   zt = f(xt)
   loader = DataLoader(Dataset(xi, zi), batch_size=100, shuffle=False)
   model = ex.gp.core.Model(ex.constant_mean, ex.kernel)
   model, info = ex.gp.kernel.select_parameters_with_remap(
       model, dataloader=loader, info=True
   )
   zpm, zpv = model.predict(xi, zi, xt)
   ex.visualize_predictions(problem_name, zt, zpm)

Script: ``examples/gpmp_example30_dataloader.py``

.. literalinclude:: ../../../examples/gpmp_example30_dataloader.py
   :language: python
   :linenos:
