"""TuRBO-1 (Eriksson et al., NeurIPS 2019, arXiv 1910.01739) re-implementation.

A from-scratch re-implementation of TuRBO-1 (single trust-region variant) as the
7th Tier-1 baseline for the paper's §VI-B/§VI-C comparison — the canonical
high-dimensional BO method, added to close the "did you compare against modern
high-d BO?" gap left by the vanilla scikit-optimize BO baseline.

Modules:
  - gp_model.py        local GP, Matern-5/2 ARD kernel, MLE fit, Thompson sample
  - trust_region.py    TR side-length management (expand/shrink/restart)
  - acquisition.py     Thompson-sampling acquisition over perturbation candidates
  - train.py           TuRBO-1 main loop (algorithm 1)
  - benchmark_adapter.py  continuous-ξ benchmark -> [0,1]^d objective (BO-parity)
  - config.py          hyperparameters (paper §A.1)

Implementation note: the GP is pure numpy + scipy.linalg/optimize (no gpytorch,
no torch). This matches the task's "minimal dependencies, self-contained"
philosophy and keeps TuRBO free of the torch/libstdc++ clash that affects the
RA sweep. The acquisition uses the paper's Thompson sampling (NOT EI/UCB).
"""
