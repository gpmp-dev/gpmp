Examples
========

The repository contains runnable scripts in the ``examples/`` directory. The
pages below are rendered from a selected subset of those scripts. Each page
states the modeling objective, shows at least one plot, and then includes the
full script so that the example can be copied or run directly.

How to use these examples
-------------------------

Start with :doc:`interpolation_1d` if you want the minimal GPmp sequence:
construct a model, select covariance parameters, predict, and plot. Move to
:doc:`interpolation_2d` or :doc:`interpolation_nd` for multidimensional
problems. Use :doc:`parameter_selection` when the main question is how ML,
REML, and REMAP differ. Use :doc:`dataloader` when observations are organized
through batching utilities.

The rendered previews are intentionally shorter than some scripts. Expensive
sections, such as long posterior sampling loops, may be discussed but not fully
executed during the documentation build. The literal script included on each
page remains the reference implementation.

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
GPmp convention

.. math::

   \theta = \mathrm{covparam}
   = \left(\log(\sigma^2), -\log(\rho_1), \ldots, -\log(\rho_d)\right).

The covariance blocks are written with lowercase symbols. For example,
:math:`k_{ii}` is the matrix with entries
:math:`k_\theta(x_i^a, x_i^b)`, and :math:`k_{it}` contains the covariances
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
