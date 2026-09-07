"""RobustAnalog multi-task environment + reward (paper eqs 2-3).

The environment wraps one of our benchmarks (simulator + spec + sampler) into
the corner-set API the training loop expects:

  - K fixed ξ-corners are the discrete "tasks".
  - An action (sizing in [-1,1]^n) is denormalized to the design box and
    simulated at each corner; the per-corner reward uses RobustAnalog's own
    reward formula (NOT our softplus surrogate).
  - SPICE accounting: each corner simulation = 1 SPICE call.

Reward (paper eqs 2-3), per corner:
  eq 3: rel_dist_i = (m_i - m_i*) / (m_i + m_i* + eps), summed over specs as
        sum_i min(rel_dist_i, 0)  with per-spec direction handling
  eq 2: if that sum < -0.02 -> return the sum  (negative, "how far from spec")
        else                 -> return +0.2    (all specs satisfied)

Direction: for ">=" specs feasibility is m >= target so we use
(m - target); for "<=" specs feasibility is m <= target so we use
(target - m). The min(., 0) then fires only when that spec is violated.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import RAConfig


# A constraint descriptor extracted from a benchmark's specs.
class Constraint:
    """One performance constraint with explicit direction for RA's reward.

    direction: "ge" means feasible when value >= target; "le" means feasible
    when value <= target.
    """

    def __init__(self, name: str, extract: Callable[[dict], float],
                 target: float, direction: str):
        if direction not in ("ge", "le"):
            raise ValueError(f"direction must be 'ge' or 'le', got {direction}")
        self.name = name
        self.extract = extract
        self.target = float(target)
        self.direction = direction


def per_corner_reward(metrics: dict, constraints: List[Constraint],
                      cfg: RAConfig) -> float:
    """RobustAnalog per-corner reward (paper eqs 2-3).

    Returns +cfg.reward_satisfied (0.2) when all constraints are satisfied
    (i.e. the summed shortfall >= cfg.reward_sat_threshold), else the negative
    summed relative shortfall.
    """
    r = 0.0
    for c in constraints:
        try:
            m_i = float(c.extract(metrics))
        except Exception:
            # Missing metric -> treat as maximally violated for this spec.
            r += -1.0
            continue
        m_star = c.target
        denom = abs(m_i) + abs(m_star) + cfg.reward_eps
        if c.direction == "ge":
            rel = (m_i - m_star) / denom          # >=0 when feasible
        else:  # "le"
            rel = (m_star - m_i) / denom          # >=0 when feasible
        r += min(rel, 0.0)
    if r < cfg.reward_sat_threshold:
        return r
    return cfg.reward_satisfied


class MultiTaskEnv:
    """Corner-set environment over a single benchmark.

    Args:
      simulator    : object with evaluate_with_metrics(x_real, xi) -> (loss, metrics)
      constraints  : list of Constraint (direction-aware spec descriptors)
      specs        : the benchmark's zo_yield Spec list (for hard-pass yield eval)
      xi_corners   : (K, d_xi) fixed corner pool
      x_lo, x_hi   : design-box bounds (real units) for action denormalization
      cfg          : RAConfig
    """

    def __init__(self, simulator, constraints: List[Constraint], specs,
                 xi_corners: np.ndarray, x_lo: np.ndarray, x_hi: np.ndarray,
                 cfg: RAConfig):
        self.sim = simulator
        self.constraints = constraints
        self.specs = specs
        self.xi_corners = np.asarray(xi_corners, dtype=float)
        self.x_lo = np.asarray(x_lo, dtype=float)
        self.x_hi = np.asarray(x_hi, dtype=float)
        self.cfg = cfg
        self.num_tasks = self.xi_corners.shape[0]
        self.action_dim = self.x_lo.shape[0]
        self.spice_used = 0
        # State = task-id one-hot.
        self.state_dim = self.num_tasks

    # ---- state helpers --------------------------------------------------- #
    def task_state(self, task_id: int) -> np.ndarray:
        s = np.zeros(self.num_tasks, dtype=np.float32)
        s[int(task_id)] = 1.0
        return s

    def reset(self) -> np.ndarray:
        """Return the nominal-corner state (task 0)."""
        self.spice_used = 0
        return self.task_state(0)

    # ---- action denormalization ----------------------------------------- #
    def denormalize(self, action: np.ndarray) -> np.ndarray:
        """Map action a in [-1, 1]^n to a real sizing in [x_lo, x_hi]."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return self.x_lo + (a + 1.0) * 0.5 * (self.x_hi - self.x_lo)

    # ---- simulation primitives ------------------------------------------ #
    def _sim_metrics(self, x_real: np.ndarray, xi: np.ndarray) -> dict:
        self.spice_used += 1
        try:
            _, metrics = self.sim.evaluate_with_metrics(x_real, xi)
        except Exception:
            metrics = {}
        return metrics

    def reward_at_corner(self, x_real: np.ndarray, task_id: int) -> float:
        metrics = self._sim_metrics(x_real, self.xi_corners[int(task_id)])
        return per_corner_reward(metrics, self.constraints, self.cfg)

    def step(self, action: np.ndarray, task_ids: List[int]
             ) -> Tuple[Dict[int, float], Dict]:
        """Simulate ``action`` at each task in ``task_ids``; return per-task
        rewards. Consumes len(task_ids) SPICE. (1-step episodes: done=True.)"""
        x_real = self.denormalize(action)
        rewards: Dict[int, float] = {}
        for tid in task_ids:
            rewards[int(tid)] = self.reward_at_corner(x_real, int(tid))
        info = dict(x_real=x_real, spice_used=self.spice_used)
        return rewards, info

    def evaluate_full_corner_set(self, action_or_x, is_action: bool = True
                                 ) -> Dict:
        """Outer-loop full-corner evaluation over all K corners.

        Returns dict with per-corner rewards, mean/min reward, and the number
        of corners passing (reward == +reward_satisfied). Consumes K SPICE.
        """
        x_real = self.denormalize(action_or_x) if is_action else np.asarray(action_or_x)
        rewards = np.empty(self.num_tasks)
        for tid in range(self.num_tasks):
            rewards[tid] = self.reward_at_corner(x_real, tid)
        n_pass = int(np.sum(rewards >= self.cfg.reward_satisfied - 1e-12))
        return dict(
            x_real=x_real,
            corner_rewards=rewards,
            mean_reward=float(rewards.mean()),
            min_reward=float(rewards.min()),
            n_corners_pass=n_pass,
            all_pass=bool(n_pass == self.num_tasks),
        )

    def full_corner_pass(self, action_or_x, is_action: bool = True) -> Dict:
        """Single K-SPICE pass over all corners returning everything the outer
        loop + pruner need: perf matrix, eq-2 rewards, eq-3 raw rewards, and
        pass count. Folding these into ONE pass (instead of three) is what lets
        a B=25 run afford an outer eval at all.
        """
        x_real = self.denormalize(action_or_x) if is_action else np.asarray(action_or_x)
        m = len(self.constraints)
        perf = np.full((self.num_tasks, m), np.nan)
        eq2 = np.empty(self.num_tasks)
        eq3 = np.empty(self.num_tasks)
        for k in range(self.num_tasks):
            metrics = self._sim_metrics(x_real, self.xi_corners[k])
            for j, c in enumerate(self.constraints):
                try:
                    perf[k, j] = float(c.extract(metrics))
                except Exception:
                    perf[k, j] = np.nan
            eq2[k] = per_corner_reward(metrics, self.constraints, self.cfg)
            eq3[k] = _eq3_raw_reward(metrics, self.constraints, self.cfg)
        n_pass = int(np.sum(eq2 >= self.cfg.reward_satisfied - 1e-12))
        return dict(
            x_real=x_real, perf=perf, eq2_rewards=eq2, eq3_rewards=eq3,
            mean_reward=float(eq2.mean()), min_reward=float(eq2.min()),
            n_corners_pass=n_pass, all_pass=bool(n_pass == self.num_tasks),
        )

    def perf_matrix(self, x_real: np.ndarray) -> np.ndarray:
        """Performance matrix perf in R^{K x M} for k-means pruning.

        Row k = the M raw metric values of constraint extractors at corner k.
        Consumes K SPICE.
        """
        m = len(self.constraints)
        perf = np.full((self.num_tasks, m), np.nan)
        for k in range(self.num_tasks):
            metrics = self._sim_metrics(x_real, self.xi_corners[k])
            for j, c in enumerate(self.constraints):
                try:
                    perf[k, j] = float(c.extract(metrics))
                except Exception:
                    perf[k, j] = np.nan
        return perf

    def corner_rewards_for_pruning(self, x_real: np.ndarray) -> np.ndarray:
        """Per-corner reward (eq 3 raw, NOT post-clip eq 2) for all K corners,
        used to pick the worst corner in each k-means cluster (step 4).

        Spec says use the eq-3 reward (pre-clipping) so the *degree* of
        violation differentiates corners even when several are satisfied.
        Consumes K SPICE.
        """
        out = np.empty(self.num_tasks)
        for k in range(self.num_tasks):
            metrics = self._sim_metrics(x_real, self.xi_corners[k])
            out[k] = _eq3_raw_reward(metrics, self.constraints, self.cfg)
        return out


def _eq3_raw_reward(metrics: dict, constraints: List[Constraint],
                    cfg: RAConfig) -> float:
    """Eq-3 reward WITHOUT the eq-2 clip — the signed summed shortfall."""
    r = 0.0
    for c in constraints:
        try:
            m_i = float(c.extract(metrics))
        except Exception:
            r += -1.0
            continue
        denom = abs(m_i) + abs(c.target) + cfg.reward_eps
        if c.direction == "ge":
            rel = (m_i - c.target) / denom
        else:
            rel = (c.target - m_i) / denom
        r += min(rel, 0.0)
    return r
