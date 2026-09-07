"""Tier-1 black-box baseline optimizers.

Each module exposes a unified ``optimize`` entry-point so the
grid script can swap baselines with a single function-pointer change:

    optimize(eval_fn, x_init, x_lo, x_hi, budget, n_mc, seed) -> dict
        x_best, history (list of (x, loss, yield_est)), spice_actual

eval_fn(x, rng) returns dict with 'E_loss' and 'yield_'; one call costs
n_mc SPICE simulations. The optimizer is responsible for stopping
before its cumulative SPICE exceeds budget.
"""
from .bo import optimize as bo_optimize
from .cmaes import optimize as cmaes_optimize
from .pso import optimize as pso_optimize
from .warm_start import warm_start_select

__all__ = ["bo_optimize", "cmaes_optimize", "pso_optimize",
           "warm_start_select"]
