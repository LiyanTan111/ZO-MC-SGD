"""Random-search warm start shared by every method.

`warm_start_select` samples ``n_warm_pts`` perturbations of X_init in a
small ball ([0.85, 1.18] per-dim, multiplicative), evaluates the yield of
each at n_mc=4 (cheap-filter), and returns the design that gives the
highest yield estimate (tie-break: closest to X_init in Euclidean
distance after per-coord scaling).

Adaptive sizing: at very low budgets the caller supplies
n_warm_pts via ``n_warm_pts = max(1, min(10, B // 16))`` so warm-start
costs ≤ B/4.

The warm-start cost is exactly ``n_warm_pts * 4`` SPICE.
"""
from __future__ import annotations

import numpy as np


PERTURB_LO = 0.85
PERTURB_HI = 1.18
WARM_N_MC = 4


def warm_start_select(eval_fn, x_init, x_lo, x_hi, n_warm_pts, seed):
    """Pick the best warm-start point from a small-ball RS sample.

    Args:
      eval_fn   : callable (x, rng, n_mc=...) -> dict('E_loss','yield_').
                  We pass n_mc=WARM_N_MC=4 explicitly here.
      x_init    : starting design (real-space, not normalized).
      x_lo, x_hi: design-box bounds.
      n_warm_pts: number of candidate perturbations (≥1).
      seed      : reproducibility seed (independent stream from outer ZO).

    Returns:
      dict with keys
        x_best        — chosen warm-start design
        spice_actual  — n_warm_pts * 4
        history       — list of (x, yield_)
    """
    rng = np.random.default_rng(seed)
    x_init = np.asarray(x_init, dtype=float)
    x_lo = np.asarray(x_lo, dtype=float)
    x_hi = np.asarray(x_hi, dtype=float)
    n = len(x_init)

    n_warm_pts = max(1, int(n_warm_pts))
    history = []

    # Always include X_init itself as candidate-0 (so warm-start ≥ X_init).
    candidates = [x_init.copy()]
    for _ in range(n_warm_pts - 1):
        factors = rng.uniform(PERTURB_LO, PERTURB_HI, size=n)
        x = np.clip(x_init * factors, x_lo, x_hi)
        candidates.append(x)

    # Evaluate
    spice = 0
    yields = []
    for x in candidates:
        res = eval_fn(x, rng, n_mc=WARM_N_MC)
        spice += WARM_N_MC
        history.append(dict(x=x.copy(), yield_=float(res["yield_"]),
                            E_loss=float(res["E_loss"])))
        yields.append(float(res["yield_"]))

    # Pick highest yield; tie-break by closer-to-X_init.
    yields_arr = np.array(yields)
    best_y = yields_arr.max()
    tied = np.flatnonzero(yields_arr >= best_y - 1e-9)
    if len(tied) > 1:
        dists = [float(np.linalg.norm((candidates[i] - x_init) / np.maximum(np.abs(x_init), 1e-30)))
                 for i in tied]
        best_idx = int(tied[int(np.argmin(dists))])
    else:
        best_idx = int(tied[0])

    return dict(
        x_best=candidates[best_idx],
        spice_actual=spice,
        history=history,
        chosen_idx=best_idx,
    )
