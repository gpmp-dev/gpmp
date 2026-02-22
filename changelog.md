- 0.9.32 dataloader and parameter selection updates
  * parameter_selection.py: expose batches_per_eval in high-level API
  * dataloader.py: enable full-dataset batch mode
  * testfunctions.py: add ishigami test function
  * normalize file headers and reorganize files

- 0.9.31 reorganize modules, improve performance, backend cleanup
  * plotutils: reorganize into dedicated gpmp.plot package
    - New API: gp.plot.Figure(), gp.plot.plot_loo(), gp.plot.crosssections()
    - Backward compatible: gpmp.misc.plotutils still accessible
  * modeldiagnosis: reorganize into dedicated gpmp.modeldiagnosis package  
  * num.py: performance improvements and bug fixes
  * num.py: fix dtype handling across backends
  * num.py: add type hinting for better code clarity
  * num.py: narrow selection-criterion exceptions to linalg errors
  * remove JAX backend support (deprecated)
  * fix various typos
  * update all examples to use new plot API

- 0.9.30 reorganize project layout
  * restructure package organization for better maintainability

- 0.9.27 use a a global config object
- 0.9.26 improve bound management
- 0.9.25 dataloader implementation + several improvements
- 0.9.24 update smc code
- 0.9.23 update mcmc code and implement Fisher information matrix
- 0.9.22 implement sampling from posterior
- 0.9.21 bug correction introduced in v0.9.20 about computation of posterior variance
- 0.9.20 update num.py to implement functions needed in gpmp-contrib
- 0.9.19 introduce 'DifferentiableFunction' to encapsulate gradient computation, code enhancement
- 0.9.18 small improvements in kernel.py and num.py
- 0.9.17 refactor parameter selection procedures for more general selection criteria support
- 0.9.16 changes in core.py, num.py about  Cholesky factorization
- 0.9.15 mid-2024 version
- 0.9.10 add documentation, unit tests, torch 
- 0.9.0 -> 0.9.5 alpha versions based on JAX
