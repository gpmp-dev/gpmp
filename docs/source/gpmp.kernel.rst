gpmp.kernel module
==================

.. contents::
   :local:

The kernel module defines helper functions to build covariance functions and select its parameters.

Functions
---------

exponential_kernel
^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.exponential_kernel
   :no-index:

matern32_kernel
^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.matern32_kernel

maternp_kernel
^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.maternp_kernel

maternp_covariance_ii_or_tt
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.maternp_covariance_ii_or_tt

maternp_covariance_it
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.maternp_covariance_it

maternp_covariance
^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.maternp_covariance

anisotropic_parameters_initial_guess_zero_mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.anisotropic_parameters_initial_guess_zero_mean

anisotropic_parameters_initial_guess
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.anisotropic_parameters_initial_guess

make_selection_criterion_with_gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.make_selection_criterion_with_gradient

autoselect_parameters
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.autoselect_parameters

select_parameters_with_reml
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.select_parameters_with_reml

update_parameters_with_reml
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmp.kernel.update_parameters_with_reml
