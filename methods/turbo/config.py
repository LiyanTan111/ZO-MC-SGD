"""TuRBO-1 hyperparameters (Eriksson et al. 2019, supplementary §A.1)."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TuRBOConfig:
    # ---- trust region (paper §A.1) ----
    length_init: float = 0.8          # L_init
    length_max: float = 1.6           # L_max
    length_min: float = 2.0 ** -7     # L_min = 0.0078125
    tau_succ: int = 3                 # consecutive successes to expand
    tau_fail_floor: int = 4           # τ_fail = max(4, ceil(d/10))

    # ---- initial design + candidates ----
    n_init_mult: int = 2              # N_init = n_init_mult * d (= 2d)
    n_cand_mult: int = 100            # N_cand = min(n_cand_mult*d, n_cand_cap)
    n_cand_cap: int = 1500            # cap for tractable joint Thompson sampling
                                      # (paper uses 5000; capped for CPU joint TS)

    # ---- GP fitting ----
    gp_n_restarts: int = 3            # MLE restarts
    gp_max_iter: int = 60             # L-BFGS-B iterations per restart
    gp_max_points: int = 150          # cap local-GP training set (closest pts)
    gp_noise_floor: float = 1e-6      # minimum noise variance
    gp_jitter: float = 1e-6           # Cholesky jitter

    # ---- yield protocol (parity with all baselines) ----
    n_mc_per_eval: int = 8            # ξ batch per query (matches BO baseline)
    n_mc_yield: int = 80              # final yield MC pool
    yield_xi_seed: int = 555          # fixed eval ξ-set seed

    def tau_fail(self, d: int) -> int:
        return max(self.tau_fail_floor, math.ceil(d / 10))

    def n_init(self, d: int) -> int:
        return self.n_init_mult * d

    def n_cand(self, d: int) -> int:
        return min(self.n_cand_mult * d, self.n_cand_cap)


DEFAULT_CONFIG = TuRBOConfig()
