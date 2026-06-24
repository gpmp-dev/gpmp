Dataloader-based parameter selection
====================================

This example shows how observations can be passed through ``Dataset`` and
``DataLoader`` objects during parameter selection. This is useful when the
selection criterion should be evaluated on batches rather than by manually
passing a single full array every time.

What this example does
----------------------

The script creates a test problem, stores observations in a ``Dataset``, wraps
it with a ``DataLoader``, and calls ``select_parameters_with_remap`` with the
``dataloader`` argument. The selected model is then used for prediction and the
preview plots predicted values against reference values.

How to read the figure
----------------------

The plot compares GP predictions with reference values at test points. Points
near the diagonal indicate accurate predictions. Systematic deviations from the
diagonal suggest bias, poor covariance parameters, or insufficient observations.

API points
----------

* Selection helpers accept either explicit ``xi, zi`` arrays or a
  ``dataloader``. Do not pass both.
* ``DataLoader`` controls batching; the selection criterion still returns a
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
