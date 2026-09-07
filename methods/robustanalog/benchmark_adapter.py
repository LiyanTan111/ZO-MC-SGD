"""Adapter: our continuous-ξ benchmarks -> RobustAnalog corner tasks.

RobustAnalog's "PVT corner" structure is discrete; our framework is continuous
Pelgrom-style ξ ~ N(0, Σ). The adapter samples K fixed ξ-corners once (fixed
``xi_corner_seed``), with corner 0 pinned to ξ=0 (the nominal corner, kept by
the pruner per paper §3.4), and wires the benchmark's simulator + spec targets
into a :class:`MultiTaskEnv`.

Reads (does NOT modify):
  - benchmarks/{circuit}/spec.py  (build_simulator, make_sampler, targets)
  - configs/{circuit}.json        (σ_scale, α, x_init) — α is unused by the RA
    reward but is loaded so the simulator is built identically to every other
    method's.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .agent import DDPGAgent
from .config import DEFAULT_CONFIG, RAConfig
from .multi_task_env import Constraint, MultiTaskEnv

from zo_yield import circuit_config as _cc

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CIRCUITS = _cc.CIRCUITS

# All 5 roster circuits use simple ge/le Spec constraints on raw metric values;
# build_constraints() recovers (target, direction) directly from each Spec so
# the adapter does NOT depend on per-circuit target-dict key naming (which
# differs: cs_amp hardcodes targets in its Specs, others use DEFAULT_TARGETS).


def _load_json(path: str) -> Optional[dict]:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_circuit_config(circuit: str) -> Dict:
    """Load σ_scale, α, x_init (real) and the spec module for a circuit."""
    if circuit not in CIRCUITS:
        raise ValueError(f"unknown circuit {circuit!r}; known: {list(CIRCUITS)}")
    cfg = _cc.load_circuit_config(circuit)
    return dict(mod=_cc.load_spec_module(circuit),
                scale=float(cfg["sigma_scale"]),
                alphas=cfg["alphas"],
                x_init_real=cfg["x_init"],
                rs_dir=os.path.join(REPO, "experiments", "results", circuit))


def sample_xi_corners(sampler, K: int, seed: int) -> np.ndarray:
    """K fixed ξ-corners with corner 0 = ξ=0 (nominal). The remaining K-1 are
    drawn from ρ(ξ) with a fixed seed (shared across all RA training seeds)."""
    d = sampler.dim
    corners = np.zeros((K, d), dtype=float)            # corner 0 = nominal ξ=0
    if K > 1:
        rng = np.random.default_rng(seed)
        corners[1:] = sampler.sample(K - 1, rng=rng)
    return corners


def build_constraints(mod) -> List[Constraint]:
    """Build direction-aware Constraints from the benchmark's Spec objects.

    Recovers (target, direction) by probing each Spec (which takes the
    *extracted* value): a Spec where large values satisfy is a ">=" constraint
    with target = margin(0) (margin(v) = target - v); one where large values
    violate is a "<=" constraint with target = -margin(0) (margin(v) = v -
    target). This is robust to per-circuit target-dict naming differences.
    """
    specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS
    BIG, SMALL = 1e30, -1e30
    cons = []
    for s in specs:
        sat_big = bool(s.satisfies(BIG))
        sat_small = bool(s.satisfies(SMALL))
        if sat_big and not sat_small:
            direction, target = "ge", float(s.margin(0.0))
        elif sat_small and not sat_big:
            direction, target = "le", float(-s.margin(0.0))
        else:
            # Ambiguous predicate — fall back to ">=" with margin(0) as target.
            direction, target = "ge", float(s.margin(0.0))
        cons.append(Constraint(name=s.name, extract=s.extract,
                               target=target, direction=direction))
    return cons


def make_env(circuit: str, cfg: Optional[RAConfig] = None) -> MultiTaskEnv:
    """Build a ready-to-train :class:`MultiTaskEnv` for one circuit."""
    cfg = cfg or DEFAULT_CONFIG
    cc = load_circuit_config(circuit)
    mod = cc["mod"]
    sampler = mod.make_sampler(scale=cc["scale"])
    specs = mod.build_specs() if hasattr(mod, "build_specs") else mod.SPECS
    simulator = mod.build_simulator(alphas=cc["alphas"])
    corners = sample_xi_corners(sampler, cfg.n_corners, cfg.xi_corner_seed)
    constraints = build_constraints(mod)
    env = MultiTaskEnv(
        simulator=simulator, constraints=constraints, specs=specs,
        xi_corners=corners, x_lo=mod.X_LO, x_hi=mod.X_HI, cfg=cfg,
    )
    return env


def make_agent(env: MultiTaskEnv, cfg: Optional[RAConfig] = None,
               seed: int = 0, device: str = "cpu") -> DDPGAgent:
    """Build a DDPG agent sized to the env's state/action/task dims."""
    cfg = cfg or DEFAULT_CONFIG
    return DDPGAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                     n_tasks=env.num_tasks, cfg=cfg, device=device, seed=seed)


def make_eval_sampler(circuit: str):
    """Return the benchmark's ρ(ξ) sampler (for the n_eval=200 yield protocol)."""
    cc = load_circuit_config(circuit)
    return cc["mod"].make_sampler(scale=cc["scale"])
