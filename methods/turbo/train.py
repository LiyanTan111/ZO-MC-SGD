"""TuRBO-1 main optimization loop (paper algorithm 1).

Single trust-region variant. Works in the [0,1]^d unit cube; the objective
`f(u)` is supplied by the caller (the benchmark adapter denormalizes u into the
circuit design box and returns the surrogate loss to MINIMIZE).

Budget is counted in OBJECTIVE QUERIES (the adapter charges the SPICE cost per
query). Algorithm:
  1. N_init = 2d Latin-Hypercube initial points (or as many as the budget
     allows — at very low budget TuRBO degrades to best-of-partial-LHS).
  2. TR center = best observed; L = L_init.
  3. Loop until budget exhausted: fit GP on points in the current TR box,
     Thompson-sample the next point, evaluate, update the TR; restart the TR
     (new random center, L=L_init) if L < L_min.
  4. Return the best point found across the whole run.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .acquisition import propose
from .config import TuRBOConfig
from .gp_model import GPModel
from .trust_region import TrustRegion


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube sample of n points in [0,1]^d (pure numpy)."""
    if n <= 0:
        return np.empty((0, d))
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, d))
    pts = cut[:n, None] + u * (1.0 / n)
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def train_turbo(
    objective: Callable[[np.ndarray], float],
    d: int,
    budget_queries: int,
    cfg: TuRBOConfig,
    seed: int = 0,
    x0_unit: np.ndarray = None,
) -> Dict:
    """Run TuRBO-1 for `budget_queries` objective evaluations.

    If `x0_unit` (a point in [0,1]^d) is given it is evaluated as the FIRST
    initial point, with the remaining N_init-1 drawn by LHS. This mirrors the
    vanilla-BO baseline (which seeds x_init) so TuRBO has the same starting
    information as every other method — a fairness/parity requirement, not part
    of the original paper (which has no distinguished start point).

    Returns dict with best point (unit cube), best value, query count, number
    of TR restarts, and the best-so-far trace.
    """
    rng = np.random.default_rng(seed)
    n_init = cfg.n_init(d)

    X = np.empty((0, d))
    y = np.empty(0)
    n_queries = 0
    best_trace = []

    def evaluate(u: np.ndarray) -> float:
        nonlocal X, y, n_queries
        val = float(objective(u))
        X = np.vstack([X, u.reshape(1, -1)])
        y = np.append(y, val)
        n_queries += 1
        best_trace.append(float(y.min()))
        return val

    # ---- 1) initial design (x0 seed + LHS), truncated to budget ----
    n_init_eff = min(n_init, budget_queries)
    if x0_unit is not None and n_queries < n_init_eff:
        evaluate(np.clip(np.asarray(x0_unit, dtype=float), 0.0, 1.0))
        init_pts = _lhs(max(0, n_init_eff - 1), d, rng)
    else:
        init_pts = _lhs(n_init_eff, d, rng)
    for u in init_pts:
        if n_queries >= budget_queries:
            break
        evaluate(u)

    n_restarts = 0
    tr = TrustRegion(d, cfg)
    if n_queries > 0:
        tr.set_center(X[int(np.argmin(y))])

    # ---- 3) main loop ----
    while n_queries < budget_queries:
        # Points inside the current TR box define the local GP training set.
        gp_ls = None
        # Fit a quick GP first to get lengthscales for the anisotropic box;
        # bootstrap with all points if the TR has too few.
        lo, hi = tr.get_bounding_box(gp_ls)
        in_tr = np.all((X >= lo[None, :] - 1e-12) & (X <= hi[None, :] + 1e-12), axis=1)
        idx = np.where(in_tr)[0]
        if idx.size < max(3, d + 1):
            # Not enough local points: use the nearest points to the center.
            order = np.argsort(np.sum((X - tr.center[None, :]) ** 2, axis=1))
            idx = order[:max(min(len(X), 2 * (d + 1)), 3)]
        # Cap the local GP training set (closest-to-center) so fitting stays
        # tractable at high budget — standard TuRBO local-GP practice.
        if idx.size > cfg.gp_max_points:
            dist = np.sum((X[idx] - tr.center[None, :]) ** 2, axis=1)
            idx = idx[np.argsort(dist)[:cfg.gp_max_points]]

        gp = GPModel(n_restarts=cfg.gp_n_restarts, max_iter=cfg.gp_max_iter,
                     noise_floor=cfg.gp_noise_floor, jitter=cfg.gp_jitter)
        gp.fit(X[idx], y[idx], rng=rng)

        # Anisotropic TR box from the fitted ARD lengthscales.
        lo, hi = tr.get_bounding_box(gp.ls)
        u_next = propose(gp, tr.center, lo, hi, cfg, rng)

        y_best_before = float(y.min())
        y_new = evaluate(u_next)
        tr.update_after_eval(y_new, y_best_before)
        # Recenter on the (possibly new) incumbent best.
        tr.set_center(X[int(np.argmin(y))])

        if tr.restart_needed:
            n_restarts += 1
            # New TR at a fresh random center (paper: restart on collapse).
            if n_queries < budget_queries:
                u_seed = rng.random(d)
                evaluate(u_seed)
            tr.init_new()
            tr.set_center(X[int(np.argmin(y))])

    # ---- 4) denoised final selection ----
    # The per-query objective is noisy (8-sample mean), so argmin(observed loss)
    # is dominated by lucky-low-noise points and gets WORSE as more queries
    # accumulate. Standard noise-robust fix: return the observed point with the
    # lowest GP POSTERIOR MEAN (spatial denoising over all observations).
    obs_idx = int(np.argmin(y)) if len(y) else -1
    best_idx = obs_idx
    if len(X) >= max(4, d + 1):
        try:
            sel_gp = GPModel(n_restarts=cfg.gp_n_restarts,
                             max_iter=cfg.gp_max_iter,
                             noise_floor=cfg.gp_noise_floor,
                             jitter=cfg.gp_jitter)
            sel_gp.fit(X, y, rng=rng)
            mu, _ = sel_gp.predict(X)
            best_idx = int(np.argmin(mu))
        except Exception:
            best_idx = obs_idx
    return dict(
        x_best=X[best_idx].copy() if best_idx >= 0 else None,
        y_best=float(y[best_idx]) if best_idx >= 0 else float("nan"),
        y_best_observed=float(y[obs_idx]) if obs_idx >= 0 else float("nan"),
        selected_by="gp_posterior_mean" if best_idx != obs_idx else "observed",
        n_queries=int(n_queries),
        n_init_used=int(n_init_eff),
        n_restarts=int(n_restarts),
        completed_init=bool(n_init_eff >= n_init),
        best_trace=best_trace,
    )
