"""PSO baseline (self-rolled Constriction PSO).

Configuration:
  - swarm_size = 20
  - w (inertia) = 0.729, c1 = c2 = 1.49445  (Constriction PSO defaults)
  - swarm[0] = X_init, swarm[1..19] = uniform in design box
  - Per particle: yield estimated at n_mc=8 (one eval_fn call)
  - Bounds: rejection-by-clip; positions clipped to box, velocities free
  - canonical return: gbest

Self-rolled rather than pyswarms because pyswarms' bound handling routes
through Reporter and rejected-sample logging in ways that are awkward to
splice with our SPICE-budget accounting.
"""
from __future__ import annotations

import numpy as np


SWARM_SIZE = 20
W = 0.729
C1 = 1.49445
C2 = 1.49445


def optimize(eval_fn, x_init, x_lo, x_hi, budget, n_mc, seed):
    rng = np.random.default_rng(seed)
    n = len(x_init)
    x_init = np.asarray(x_init, dtype=float)
    x_lo = np.asarray(x_lo, dtype=float)
    x_hi = np.asarray(x_hi, dtype=float)
    span = x_hi - x_lo

    # --- swarm init ---
    positions = rng.uniform(x_lo, x_hi, size=(SWARM_SIZE, n))
    positions[0] = x_init.copy()
    # velocity init: small random fraction of box span
    velocities = rng.uniform(-0.1 * span, 0.1 * span, size=(SWARM_SIZE, n))

    pbest = positions.copy()
    pbest_loss = np.full(SWARM_SIZE, np.inf)
    gbest = positions[0].copy()
    gbest_loss = np.inf

    spice = 0
    history = []

    def eval_particle(idx):
        nonlocal spice, gbest, gbest_loss
        x = positions[idx]
        res = eval_fn(x, rng)
        spice += n_mc
        loss = float(res["E_loss"])
        history.append(dict(x=x.copy(), E_loss=loss,
                            yield_=float(res["yield_"])))
        if loss < pbest_loss[idx]:
            pbest_loss[idx] = loss
            pbest[idx] = x.copy()
            if loss < gbest_loss:
                gbest_loss = loss
                gbest = x.copy()

    # initial pass — evaluate as many particles as fit
    for i in range(SWARM_SIZE):
        if spice + n_mc > budget:
            break
        eval_particle(i)

    # update loop
    while spice + n_mc <= budget:
        for i in range(SWARM_SIZE):
            if spice + n_mc > budget:
                break
            r1 = rng.uniform(0.0, 1.0, n)
            r2 = rng.uniform(0.0, 1.0, n)
            velocities[i] = (W * velocities[i]
                             + C1 * r1 * (pbest[i] - positions[i])
                             + C2 * r2 * (gbest - positions[i]))
            positions[i] = np.clip(positions[i] + velocities[i], x_lo, x_hi)
            eval_particle(i)

    if not history:
        return dict(x_best=x_init.copy(), history=[], spice_actual=0,
                    method="PSO")

    return dict(
        x_best=gbest if np.isfinite(gbest_loss) else history[0]["x"],
        history=history,
        spice_actual=spice,
        method="PSO",
    )
