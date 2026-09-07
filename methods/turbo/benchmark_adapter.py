"""Adapter: the continuous-ξ benchmarks -> a TuRBO objective on [0,1]^d.

Mirrors the BO baseline's per-query protocol so TuRBO is apples-to-apples with
the other methods:
  - objective(u): u∈[0,1]^d is denormalized to the real design box
    [X_LO, X_HI] and scored on n_mc=10 FRESH ξ samples drawn per query,
    returning the negative Monte Carlo yield -- the same objective the
    natural-objective BO/CMA-ES/PSO baselines optimize. Cost = 10 SPICE.
  - final yield: n_mc=80 at the FIXED ξ-set (seed 555) -- identical to every
    baseline, so TuRBO's yield_init reproduces each circuit's yield_init.

The optimized objective never doubles as the score: the final yield is always
measured by the shared protocol above.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from zo_yield import circuit_config as _cc

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CIRCUITS = _cc.CIRCUITS

N_MC_PER_EVAL = 10       # yield-MC samples per query, matching BO/CMA-ES/PSO
N_MC_YIELD = 80
YIELD_XI_SEED = 555


def _load_json(p):
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return None


def _circuit_cfg(circuit):
    """Return (spec module, σ-scale, α, x_init) from configs/<circuit>.json."""
    cfg = _cc.load_circuit_config(circuit)
    return (_cc.load_spec_module(circuit), float(cfg["sigma_scale"]),
            cfg["alphas"], cfg["x_init"])


class TuRBOEnv:
    """[0,1]^d objective wrapper around a benchmark, BO-protocol-faithful."""

    def __init__(self, circuit: str, seed: int = 0):
        self.circuit = circuit
        mod, scale, alphas, x_init_real = _circuit_cfg(circuit)
        self.mod = mod
        self.scale = scale
        self.alphas = alphas
        self.x_init_real = x_init_real
        self.x_lo = np.asarray(mod.X_LO, dtype=float)
        self.x_hi = np.asarray(mod.X_HI, dtype=float)
        self.n_design = len(x_init_real)
        self.sampler = mod.make_sampler(scale=scale)
        self.specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS
        # Per-query simulator + fresh-ξ stream (mirrors BO's make_eval_fn).
        self._sim = mod.build_simulator(alphas=alphas)
        self._rng_xis = np.random.default_rng(seed + 13_579)
        self.spice_used = 0
        self.n_queries = 0
        # Fixed-ξ yield evaluator (seed 555) shared by all baselines.
        self._eval_sim = mod.build_simulator(alphas=alphas)
        self._yield_xis = self.sampler.sample(
            N_MC_YIELD, rng=np.random.default_rng(YIELD_XI_SEED))
        self.yield_init = self._yield_at(x_init_real)
        # x_init in unit-cube coords, so TuRBO can seed it as its first init
        # point (parity with the BO baseline which evaluates x_init first).
        span = np.where((self.x_hi - self.x_lo) > 0, self.x_hi - self.x_lo, 1.0)
        self.x_init_unit = np.clip((x_init_real - self.x_lo) / span, 0.0, 1.0)

    def denormalize(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        return self.x_lo + u * (self.x_hi - self.x_lo)

    def objective(self, u: np.ndarray) -> float:
        """Negative yield estimate over n_mc=10 fresh ξ (the TuRBO query, yield-MC
        objective;. 10 SPICE/call. Returns
        −Ŷ(x) ∈ [−1, 0] — TuRBO minimizes. Identical objective to the
        natural-objective BO/CMA/PSO baselines (apples-to-apples)."""
        from methods.natural_objective.objective import natural_objective
        x_real = np.clip(self.denormalize(u), self.x_lo, self.x_hi)
        xis = self.sampler.sample(N_MC_PER_EVAL, rng=self._rng_xis)
        res = natural_objective(self._sim, self.specs, x_real, xis)
        self.spice_used += N_MC_PER_EVAL
        self.n_queries += 1
        return float(res["e_loss"])

    def _yield_at(self, x_real: np.ndarray) -> float:
        from zo_yield.evaluation import evaluate_design

        class _Fixed:
            dim = self.sampler.dim
            def sample(self_inner, n, rng=None):
                return self._yield_xis[:n]
        res = evaluate_design(self._eval_sim,
                              np.clip(x_real, self.x_lo, self.x_hi),
                              _Fixed(), self.specs, n_mc=N_MC_YIELD)
        return float(res["yield_"])

    def evaluate_final_yield(self, u_best: np.ndarray) -> float:
        """Final yield at the fixed ξ-set (seed 555), n_mc=80 — baseline parity."""
        return self._yield_at(self.denormalize(u_best))


def make_env(circuit: str, seed: int = 0) -> TuRBOEnv:
    return TuRBOEnv(circuit, seed=seed)
