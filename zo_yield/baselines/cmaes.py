"""CMA-ES baseline (pycma).

Configuration:
  - pop_size = 4 + ⌊3 ln(n)⌋
  - σ_init = 0.3 (in normalized [0, 1] coordinates)
  - mean_init = hard X_init (normalized)
  - bounds: [0, 1] per dim with rejection sampling (cma's own bound handler)
  - Internally rescale each design var to [0, 1] before passing to cma —
    cma works much better on a unit cube than on raw circuit-design
    bounds that span 5 decades.
"""
from __future__ import annotations

import numpy as np


def optimize(eval_fn, x_init, x_lo, x_hi, budget, n_mc, seed):
    import cma

    rng = np.random.default_rng(seed)
    n = len(x_init)
    x_init = np.asarray(x_init, dtype=float)
    x_lo = np.asarray(x_lo, dtype=float)
    x_hi = np.asarray(x_hi, dtype=float)
    span = x_hi - x_lo
    span = np.where(span > 0, span, 1.0)

    def to_unit(x):
        return np.clip((x - x_lo) / span, 0.0, 1.0)

    def from_unit(z):
        return np.clip(np.asarray(z) * span + x_lo, x_lo, x_hi)

    z_init = to_unit(x_init)
    pop_size = 4 + int(3 * np.log(max(n, 2)))

    es = cma.CMAEvolutionStrategy(
        z_init.tolist(),
        0.3,
        {
            "popsize": pop_size,
            "bounds": [[0.0] * n, [1.0] * n],
            "seed": int(seed) + 1,  # cma's seed=0 is reserved for "no seed"
            "verbose": -9,
            "maxfevals": int(budget // n_mc) + pop_size + 1,
        },
    )

    spice = 0
    history = []

    while spice + n_mc <= budget:
        if es.stop():
            break
        zs = es.ask()
        losses_for_tell = []
        evaluated_xs = []
        for z in zs:
            if spice + n_mc > budget:
                # cma needs every member of the population to have a value;
                # use the worst observed loss so far (or +inf) as a sentinel
                losses_for_tell.append(float("inf"))
                evaluated_xs.append(None)
                continue
            x = from_unit(z)
            res = eval_fn(x, rng)
            spice += n_mc
            history.append(dict(x=x.copy(),
                                E_loss=float(res["E_loss"]),
                                yield_=float(res["yield_"])))
            losses_for_tell.append(float(res["E_loss"]))
            evaluated_xs.append(x)
        # tell only if at least one finite value (else cma errors out)
        if any(np.isfinite(l) for l in losses_for_tell):
            try:
                es.tell(zs, losses_for_tell)
            except Exception:
                # partial population at end-of-budget: tolerate failure
                pass

    if not history:
        # budget too small to evaluate even one design
        return dict(x_best=x_init.copy(), history=[], spice_actual=0,
                    method="CMA-ES")

    losses = np.array([h["E_loss"] for h in history])
    best_idx = int(np.argmin(losses))
    return dict(
        x_best=history[best_idx]["x"],
        history=history,
        spice_actual=spice,
        pop_size=pop_size,
        method="CMA-ES",
    )
