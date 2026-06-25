Examples
========

The repository contains runnable scripts in the ``examples/`` directory. The
pages below are rendered from a selected subset of those scripts. Each page
states the modeling objective, shows at least one plot, and then includes the
full script so that the example can be copied or run directly.

Example coverage
----------------

The selected examples cover the main GPmp operations:

* :doc:`interpolation_1d` gives the minimal sequence: model construction,
  covariance-parameter selection, prediction, and plotting.
* :doc:`interpolation_2d` and :doc:`interpolation_nd` show the same operations
  when the input dimension is larger than one.
* :doc:`parameter_selection` compares ML, REML, and REMAP on one data set.
* :doc:`dataloader` passes observations through ``Dataset`` and ``DataLoader``
  objects during parameter selection.
* :doc:`posterior_sampling` uses reduced iteration counts in the documentation
  build. The code block shows the sampler arguments that control longer runs.

Notation used below
-------------------

The example pages use the same names as the scripts. Observation points are
stored in ``xi`` and written mathematically as :math:`x_i`. Prediction points
are stored in ``xt`` and written as :math:`x_t`.

The latent process is denoted by :math:`Z`. A typical model is

.. math::

   Z \sim \mathcal{GP}(m, k_\theta),

where :math:`m` is the mean function and :math:`k_\theta` is the covariance
kernel. In the Matern examples, the covariance parameter vector follows the
convention

.. math::

   \theta = \mathrm{covparam}
   =
   \left(\log(\sigma^2), -\log(\rho_0), \ldots,
   -\log(\rho_{d-1})\right).

The covariance kernel is written with a lowercase symbol
:math:`k_\theta`. Covariance matrices and blocks are written with uppercase
symbols. For example, :math:`K_{ii}` is the matrix with entries
:math:`k_\theta(x_i^a, x_i^b)`, and :math:`K_{it}` contains the covariances
between observation points and prediction points.

Random variables use uppercase letters. Thus :math:`Z_i = Z(x_i)` and
:math:`Z_t = Z(x_t)` are random variables. The arrays ``zi`` and ``zt`` store
realizations, denoted by lowercase :math:`z_i` and :math:`z_t`.

Selected examples
-----------------

.. list-table::
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
