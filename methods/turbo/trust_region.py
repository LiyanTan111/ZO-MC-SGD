"""Trust-region management for TuRBO-1 (paper §2 + §A.1).

Tracks the base side length L, success/failure counters, and produces the
per-dimension TR bounding box (rescaled by the GP's ARD lengthscales and
clipped to the [0,1]^d unit cube). Expands after τ_succ consecutive successes,
shrinks after τ_fail consecutive failures, and signals a restart when L falls
below L_min.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import TuRBOConfig


class TrustRegion:
    def __init__(self, d: int, cfg: TuRBOConfig):
        self.d = d
        self.cfg = cfg
        self.tau_fail = cfg.tau_fail(d)
        self.tau_succ = cfg.tau_succ
        self.init_new()

    def init_new(self, center: Optional[np.ndarray] = None) -> None:
        self.L = self.cfg.length_init
        self.c_succ = 0
        self.c_fail = 0
        self.center = (np.asarray(center, dtype=float) if center is not None
                       else None)
        self.restart_needed = False

    def set_center(self, center: np.ndarray) -> None:
        self.center = np.asarray(center, dtype=float)

    def update_after_eval(self, y_new: float, y_best: float,
                          tol: float = 1e-3) -> None:
        """A 'success' is a strict improvement over the incumbent best
        (we are MINIMIZING the surrogate loss)."""
        if y_new < y_best - tol:
            self.c_succ += 1
            self.c_fail = 0
        else:
            self.c_fail += 1
            self.c_succ = 0

        if self.c_succ >= self.tau_succ:
            self.L = min(self.cfg.length_max, 2.0 * self.L)
            self.c_succ = 0
            self.c_fail = 0
        elif self.c_fail >= self.tau_fail:
            self.L = self.L / 2.0
            self.c_succ = 0
            self.c_fail = 0

        if self.L < self.cfg.length_min:
            self.restart_needed = True

    def per_dim_lengths(self, gp_lengthscales: Optional[np.ndarray]) -> np.ndarray:
        """Per-dim side lengths L_i = λ_i * L / (∏_j λ_j)^(1/d) (paper §2).

        With no GP lengthscales (before first fit) the box is isotropic (L_i=L).
        """
        if gp_lengthscales is None:
            return np.full(self.d, self.L)
        ls = np.asarray(gp_lengthscales, dtype=float)
        # Geometric-mean normalization so the box volume tracks L^d.
        log_gm = np.mean(np.log(ls))
        weights = ls / np.exp(log_gm)
        return weights * self.L

    def get_bounding_box(self, gp_lengthscales: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """TR hyperrectangle [lo, hi] clipped to the [0,1]^d unit cube."""
        if self.center is None:
            raise RuntimeError("trust region center not set")
        Li = self.per_dim_lengths(gp_lengthscales)
        lo = np.clip(self.center - Li / 2.0, 0.0, 1.0)
        hi = np.clip(self.center + Li / 2.0, 0.0, 1.0)
        return lo, hi
