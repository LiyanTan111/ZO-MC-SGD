"""Yield-MC ("natural") objective for BO/CMA-ES/PSO.

The objective is the batched yield estimate at x over n_mc fresh ξ samples:
  Ŷ(x) = (#{ξ_i : all specs satisfied at (x, ξ_i)}) / n_valid
where n_valid drops simulator-NaN samples. Methods MINIMIZE, so the
optimizer objective is E_loss = −Ŷ(x). Each call consumes exactly n_mc SPICE.

`make_natural_eval_fn` wraps this into the eval_fn(x_norm, rng, n_mc) shape the
existing zo_yield.baselines optimizers expect, so only the objective changes —
the BO/CMA-ES/PSO algorithms are reused unmodified.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np


def natural_objective(simulator, specs: Sequence, x_real: np.ndarray,
                      xi_samples: np.ndarray) -> Dict:
    """Yield estimate at x_real over the given ξ samples (NaN-dropping).

    Returns dict(yield_, e_loss=−yield_, n_mc, n_valid, n_nan). Consumes
    len(xi_samples) simulator calls (= n_mc SPICE).
    """
    n_mc = len(xi_samples)
    passes = 0
    n_valid = 0
    n_nan = 0
    for xi in xi_samples:
        try:
            _, metrics = simulator.evaluate_with_metrics(x_real, xi)
        except Exception:
            n_nan += 1
            continue
        ok = True
        valid = True
        for s in specs:
            try:
                v = s.extract(metrics)
                if v is None or not np.isfinite(float(v)):
                    valid = False
                    break
                ok = ok and bool(s.satisfies(v))
            except Exception:
                valid = False
                break
        if not valid:
            n_nan += 1
            continue
        n_valid += 1
        if ok:
            passes += 1
    yield_ = (passes / n_valid) if n_valid > 0 else 0.0
    return dict(yield_=float(yield_), e_loss=float(-yield_), n_mc=int(n_mc),
                n_valid=int(n_valid), n_nan=int(n_nan))


def make_natural_eval_fn(simulator, sampler, specs, x_lo, x_hi, scale_x,
                         run_seed: int, n_mc: int = 10,
                         stats: Optional[Dict] = None) -> Callable:
    """Build an eval_fn(x_norm, rng=None, n_mc=...) for the existing optimizers.

    - x_norm is in the optimizers' normalized space; mapped to real units via
      scale_x and clipped to [x_lo, x_hi] (mirrors baseline_comparison.py).
    - Each call draws n_mc FRESH ξ from a persistent per-run stream (every query uses its own deterministic ξ-sample stream).
    - Returns dict(E_loss=−yield, yield_=yield) — optimizers minimize E_loss,
      so they maximize the yield estimate.
    - If `stats` (a dict) is given, accumulates n_queries / n_sims / n_nan_total
      so the caller can apply the >30%-NaN run-failure flag.
    """
    x_lo = np.asarray(x_lo, dtype=float)
    x_hi = np.asarray(x_hi, dtype=float)
    scale_x = np.asarray(scale_x, dtype=float)
    rng_xis = np.random.default_rng(run_seed * 10_000 + 7)
    if stats is not None:
        stats.setdefault("n_queries", 0)
        stats.setdefault("n_sims", 0)
        stats.setdefault("n_nan_total", 0)

    def _eval(x_norm, rng=None, n_mc_override: Optional[int] = None) -> Dict:
        m = int(n_mc_override) if n_mc_override is not None else n_mc
        x_real = np.clip(np.asarray(x_norm, dtype=float) * scale_x, x_lo, x_hi)
        xi_samples = sampler.sample(m, rng=rng_xis)
        res = natural_objective(simulator, specs, x_real, xi_samples)
        if stats is not None:
            stats["n_queries"] += 1
            stats["n_sims"] += m
            stats["n_nan_total"] += res["n_nan"]
        return dict(E_loss=res["e_loss"], yield_=res["yield_"],
                    n_nan=res["n_nan"], n_valid=res["n_valid"])

    return _eval
