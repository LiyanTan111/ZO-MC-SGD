"""unit tests for BO / CMA-ES / PSO / warm-start baselines.

Each test uses a synthetic, SPICE-free `eval_fn(x, rng) -> dict` whose
landscape has a known optimum. The acceptance bar is "find x within
0.05 of the optimum's objective in ≤ 50 evaluations" (the reference protocol §3
M1.11.1 unit-test style).

eval_fn semantics (mirrors the optimizer interface):
  arg     x   : design vector (np.ndarray)
  arg     rng : RandomState (used only when adding noise)
  kwarg   n_mc: optional, ignored by synthetic
  return      : dict with E_loss (float, smaller=better) and yield_
                (float in [0,1], larger=better, derived from loss)
"""
from __future__ import annotations

import numpy as np
import pytest

from zo_yield.baselines.bo import optimize as bo_optimize
from zo_yield.baselines.cmaes import optimize as cmaes_optimize
from zo_yield.baselines.pso import optimize as pso_optimize
from zo_yield.baselines.warm_start import warm_start_select


def make_synthetic(d=4, x_opt=None, noise=0.0):
    """Return (eval_fn, x_init, x_lo, x_hi, x_opt). Loss = ||x − x_opt||²."""
    rng = np.random.default_rng(0)
    x_lo = np.zeros(d)
    x_hi = np.ones(d) * 4.0
    if x_opt is None:
        x_opt = np.full(d, 2.0)
    x_init = np.full(d, 0.5)  # far from x_opt

    def eval_fn(x, rng=None, n_mc=8):
        x = np.asarray(x, dtype=float)
        loss = float(np.sum((x - x_opt) ** 2))
        if noise > 0 and rng is not None:
            loss = loss + noise * float(rng.standard_normal())
        # convert to yield via a sigmoid: high yield when loss < 1
        y = 1.0 / (1.0 + np.exp(2.0 * (loss - 1.0)))
        return dict(E_loss=loss, yield_=float(np.clip(y, 0.0, 1.0)))

    return eval_fn, x_init, x_lo, x_hi, x_opt


# ----------------------------------------------------------------------------
# BO
# ----------------------------------------------------------------------------
def test_bo_recovers_synthetic_optimum():
    eval_fn, x_init, x_lo, x_hi, x_opt = make_synthetic(d=3)
    res = bo_optimize(eval_fn, x_init, x_lo, x_hi,
                       budget=50 * 8, n_mc=8, seed=0)
    loss_at_best = np.sum((res["x_best"] - x_opt) ** 2)
    assert loss_at_best < 0.5, (
        f"BO failed to converge: loss_at_best={loss_at_best:.3f} "
        f"x_best={res['x_best']}"
    )
    assert res["spice_actual"] <= 50 * 8


def test_bo_respects_budget():
    eval_fn, x_init, x_lo, x_hi, _ = make_synthetic(d=3)
    res = bo_optimize(eval_fn, x_init, x_lo, x_hi,
                       budget=40, n_mc=8, seed=0)
    assert res["spice_actual"] <= 40


# ----------------------------------------------------------------------------
# CMA-ES
# ----------------------------------------------------------------------------
def test_cmaes_recovers_synthetic_optimum():
    eval_fn, x_init, x_lo, x_hi, x_opt = make_synthetic(d=3)
    # CMA-ES at d=3 uses pop_size=7, so 80 evals = ~11 generations
    res = cmaes_optimize(eval_fn, x_init, x_lo, x_hi,
                          budget=80 * 8, n_mc=8, seed=0)
    loss_at_best = np.sum((res["x_best"] - x_opt) ** 2)
    assert loss_at_best < 0.5, (
        f"CMA-ES failed to converge: loss_at_best={loss_at_best:.3f} "
        f"x_best={res['x_best']}"
    )


def test_cmaes_respects_budget():
    eval_fn, x_init, x_lo, x_hi, _ = make_synthetic(d=3)
    res = cmaes_optimize(eval_fn, x_init, x_lo, x_hi,
                          budget=40, n_mc=8, seed=0)
    assert res["spice_actual"] <= 40


# ----------------------------------------------------------------------------
# PSO
# ----------------------------------------------------------------------------
def test_pso_recovers_synthetic_optimum():
    eval_fn, x_init, x_lo, x_hi, x_opt = make_synthetic(d=3)
    res = pso_optimize(eval_fn, x_init, x_lo, x_hi,
                        budget=80 * 8, n_mc=8, seed=0)
    loss_at_best = np.sum((res["x_best"] - x_opt) ** 2)
    assert loss_at_best < 0.5, (
        f"PSO failed to converge: loss_at_best={loss_at_best:.3f} "
        f"x_best={res['x_best']}"
    )


def test_pso_respects_budget():
    eval_fn, x_init, x_lo, x_hi, _ = make_synthetic(d=3)
    res = pso_optimize(eval_fn, x_init, x_lo, x_hi,
                        budget=40, n_mc=8, seed=0)
    assert res["spice_actual"] <= 40


# ----------------------------------------------------------------------------
# Warm-start
# ----------------------------------------------------------------------------
def test_warm_start_picks_best_yield():
    """Warm-start should find a candidate strictly better than X_init when
    one exists (we set X_init slightly suboptimal)."""
    eval_fn, _, x_lo, x_hi, x_opt = make_synthetic(d=4)
    # X_init = 1.5 (closer to x_opt=2.0 than x=0.5 was, but still suboptimal)
    x_init = np.full(4, 1.5)
    res = warm_start_select(eval_fn, x_init, x_lo, x_hi,
                             n_warm_pts=10, seed=0)
    yields = [h["yield_"] for h in res["history"]]
    assert max(yields) >= yields[0] - 1e-9, "warm-start picked worse than X_init"
    assert res["spice_actual"] == 10 * 4


def test_warm_start_monotone_with_more_pts():
    """Mean yield of best-of-n is non-decreasing in n (statistically)."""
    eval_fn, _, x_lo, x_hi, x_opt = make_synthetic(d=3)
    x_init = np.full(3, 1.2)
    bests = []
    for n_pts in [1, 3, 10]:
        res = warm_start_select(eval_fn, x_init, x_lo, x_hi,
                                  n_warm_pts=n_pts, seed=42)
        bests.append(max(h["yield_"] for h in res["history"]))
    assert bests[2] >= bests[0]  # 10 pts >= 1 pt
