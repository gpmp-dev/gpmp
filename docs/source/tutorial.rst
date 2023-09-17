GPmp Tutorial
=============

This tutorial is an introduction to GPmp, the Gaussian process micro
package, which provides simple building blocks for GP-based
algorithms. GPmp is designed to be fast and highly customizable,
offering three numerical computation backends: plain numpy, JAX, or
PyTorch.

GPmp includes several core features such as GP interpolation and
regression, known or unknown mean / intrinsic kriging, the standard
Gaussian likelihood, and the restricted likelihood of a model. It also
supports leave-one-out predictions and conditional sample paths using
fast cross-validation formulas.

Assuming basic knowledge of GPs, this tutorial aims to provide a
concise and straightforward introduction to using GPmp to develop
GP-based algorithms. For those who require further background
information on GPs, the following references are recommended:

 -  Stein, M. (1999). Interpolation of Spatial Data: Some Theory for Kriging. Springer.
 -  Rasmussen, C. E., & Williams, C. K. I. (2005). Gaussian Processes
    for Machine Learning. MIT Press.

Installation and backend selection
----------------------------------

In this tutorial, we assume that

To install PyTorch using pip:

.. code-block:: rst

   $ pip install torch

To install JAX using pip:

.. code-block:: rst

   $ pip install jax

These commands can be run in your terminal or command prompt to install
PyTorch or JAX respectively.


Basics
------

As mentioned above, we use the framework of kriging to obtain the
posterior distribution of a Gaussian process from
observations. Hereafter, the notation :math:`\xi \sim \mathrm{GP}(m,k)`
means that :math:`\xi` is a Gaussian process with mean function
:math:`m:x\in\mathcal{X}\mapsto\mathrm{E}(\xi(x))` and covariance
function :math:`k:(x,y)\in\mathcal{X}^2\mapsto
\mathrm{cov}(\xi(x),\xi(y))`, with respect to a probability space
:math:`(\Omega,\mathcal{B},\mathbb{P}_0)`.

Consider the model :math:`\xi` defined by:

.. math::

   \left\{
   \begin{array}{l}
   \xi \sim \mathrm{GP}(m,k), \\
   m(\cdot) = \sum_{i=1}^q \beta_i p_i(\cdot), \\
   \beta_1,\ldots,\beta_q \in \mathbb{R},
   \end{array}\right.

where the :math:`\beta_i`'s are unknown parameters, the :math:`p_i`'s
form a basis of d-variate polynomials, and :math:`k` is a continuous,
strictly positive-definite function.
