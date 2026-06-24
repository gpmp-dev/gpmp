Examples
========

The repository contains runnable scripts in the ``examples/`` directory. The
pages below are rendered from a selected subset of those scripts. Each page
states the modeling objective, shows at least one plot, and then includes the
full script so that the example can be copied or run directly.

How to use these examples
-------------------------

Start with :doc:`interpolation_1d` if you want the minimal GPmp workflow:
construct a model, select covariance parameters, predict, and plot. Move to
:doc:`interpolation_2d` or :doc:`interpolation_nd` for multidimensional
problems. Use :doc:`parameter_selection` when the main question is how ML,
REML, and REMAP differ. Use :doc:`dataloader` when observations are organized
through batching utilities.

The rendered previews are intentionally shorter than some scripts. Expensive
sections, such as long posterior sampling loops, may be discussed but not fully
executed during the documentation build. The literal script included on each
page remains the reference implementation.

Selected examples
-----------------

.. list-table:: Selected examples
   :header-rows: 1
   :widths: 28 32 40

   * - Topic
     - Page
     - What it illustrates
   * - Matern covariance functions
     - :doc:`materncov`
     - Compare half-integer Matern kernels and their smoothness behavior.
   * - 1D interpolation
     - :doc:`interpolation_1d`
     - Build a noise-free GP model, select covariance parameters, and plot the
       posterior.
   * - 2D interpolation
     - :doc:`interpolation_2d`
     - Build an anisotropic Matern model and inspect reference, prediction,
       error, and uncertainty fields.
   * - Higher-dimensional interpolation
     - :doc:`interpolation_nd`
     - Use leave-one-out diagnostics when spatial plotting is no longer
       practical.
   * - Custom covariance
     - :doc:`custom_kernel`
     - Define and use a covariance callable with the GPmp model interface.
   * - Noisy observations
     - :doc:`regression_noisy`
     - Model noisy observations while predicting the latent process.
   * - Conditional sample paths
     - :doc:`sample_paths`
     - Generate posterior sample paths in the noise-free setting.
   * - Noisy conditional sample paths
     - :doc:`sample_paths_noisy`
     - Generate conditional paths when observations have heteroscedastic noise.
   * - ML / REML / REMAP comparison
     - :doc:`parameter_selection`
     - Compare parameter-selection criteria on the same one-dimensional setup.
   * - Posterior parameter sampling
     - :doc:`posterior_sampling`
     - Start from REMAP selection and explore covariance-parameter uncertainty.
   * - Dataloader-based selection
     - :doc:`dataloader`
     - Use ``Dataset`` and ``DataLoader`` objects for batched parameter
       selection.

Running examples locally
------------------------

Run an example from the repository root, for instance:

.. code-block:: shell

   python examples/gpmp_example02_1d_interpolation.py

When writing new examples, prefer the same structure: define data generation,
define the model, select covariance parameters, run prediction or diagnostics,
and keep plotting in small helper functions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   materncov
   interpolation_1d
   interpolation_2d
   interpolation_nd
   custom_kernel
   regression_noisy
   sample_paths
   sample_paths_noisy
   parameter_selection
   posterior_sampling
   dataloader
