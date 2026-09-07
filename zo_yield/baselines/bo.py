"""Bayesian Optimization baseline (skopt GP-EI).

Configuration:
  - GP, Matern-5/2, learned per-dim length scales (skopt default)
  - Acquisition: Expected Improvement, no lookahead
  - Initial design: hard X_init + random uniform-in-box, capped to leave
    ≥ 5 SPICE remaining for at least one BO iter.
  - noise='gaussian' to model the n_mc=8 stderr of yield estimates
  - Bounds: rejection-by-clip when GP suggests out-of-box (skopt's space
    auto-bounds, so this is a safety net)

We optimize *mean loss* (smooth, simulator's native output) rather than
yield directly. Both come from the same n_mc samples; loss is what skopt's
GP can model well.
"""
from __future__ import annotations

import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from skopt import Optimizer
    from skopt.space import Real


def optimize(eval_fn, x_init, x_lo, x_hi, budget, n_mc, seed):
    rng = np.random.default_rng(seed)
    n = len(x_init)
    x_lo = np.asarray(x_lo, dtype=float)
    x_hi = np.asarray(x_hi, dtype=float)
    x_init = np.asarray(x_init, dtype=float)

    # n_init = X_init + min(9, B/(2·n_mc) − 5), capped to leave
    # ≥ 5 SPICE for at least one BO iter (i.e., n_init·n_mc ≤ B − n_mc).
    n_init_extra = min(9, max(0, budget // (2 * n_mc) - 5))
    n_init_total = max(1, 1 + n_init_extra)
    # Cap so at least one iteration left:
    while n_init_total > 1 and n_init_total * n_mc > budget - n_mc:
        n_init_total -= 1

    space = [Real(float(lo), float(hi)) for lo, hi in zip(x_lo, x_hi)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bo = Optimizer(
            space,
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=0,  # we tell init points manually
            random_state=int(seed),
            acq_optimizer="auto",
        )

    spice = 0
    history = []  # list of dicts: x, E_loss, yield_

    def _tell(x, res):
        history.append(dict(x=np.asarray(x, dtype=float).copy(),
                            E_loss=float(res["E_loss"]),
                            yield_=float(res["yield_"])))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bo.tell(list(x.astype(float)), float(res["E_loss"]))

    # 1) X_init
    res = eval_fn(x_init, rng)
    spice += n_mc
    _tell(x_init, res)

    # 2) random init pts
    for _ in range(n_init_total - 1):
        if spice + n_mc > budget:
            break
        x = rng.uniform(x_lo, x_hi)
        res = eval_fn(x, rng)
        spice += n_mc
        _tell(x, res)

    # 3) BO iterations
    while spice + n_mc <= budget:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                x_ask = bo.ask()
            except Exception as e:
                # GP failed (e.g., all observations identical); fall back to RS
                x_ask = list(rng.uniform(x_lo, x_hi))
        x = np.clip(np.asarray(x_ask, dtype=float), x_lo, x_hi)
        res = eval_fn(x, rng)
        spice += n_mc
        _tell(x, res)

    # Canonical return: lowest observed loss (skopt's `result.x` semantics).
    losses = np.array([h["E_loss"] for h in history])
    best_idx = int(np.argmin(losses))
    return dict(
        x_best=history[best_idx]["x"],
        history=history,
        spice_actual=spice,
        n_init_total=n_init_total,
        method="BO",
    )
