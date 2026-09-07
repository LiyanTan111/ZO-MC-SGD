"""RobustAnalog hyperparameters (paper §4.2, verbatim where specified).

Values marked "paper §4.2" are taken directly from Shi et al. Values marked
"DDPG standard" are standard defaults the paper left unspecified; the task
This file documents each such choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RAConfig:
    # ---- DDPG networks ----
    hidden_size: int = 256          # paper says "4-layer MLP"; 256 standard
    n_hidden_layers: int = 3        # 3 hidden + 1 output = "4-layer MLP"

    # ---- DDPG optimization (paper §4.2 + DDPG standard) ----
    batch_size: int = 64            # N_s, paper §4.2
    replay_capacity: int = 1000     # per task, paper §4.2
    actor_lr: float = 1e-4          # DDPG standard
    critic_lr: float = 1e-3         # DDPG standard
    gamma: float = 0.99             # DDPG standard (episodes ~1-step here)
    tau: float = 0.005              # target-net soft update, DDPG standard
    explore_noise: float = 0.2      # exploration sigma, paper §4.2

    # ---- training schedule ----
    warmup_episodes: int = 50       # W; paper unspecified, reasonable @ buf=1000
    train_iters_per_step: int = 1   # gradient steps per env interaction

    # ---- multi-task structure ----
    n_corners: int = 20             # K, fixed corner pool (revised down)
    xi_corner_seed: int = 0         # fixed corner-sampling seed
    n_eval_mc: int = 200            # final yield eval MC pool

    # ---- k-means task pruning ----
    cluster_range: Tuple[int, int] = (2, 4)   # adaptive via silhouette
    include_nominal: bool = True    # always keep the xi=0 corner (paper §3.4)
    reprune_every: int = 5          # re-run pruning every N outer evals

    # ---- reward (paper eqs 2-3) ----
    reward_satisfied: float = 0.2   # eq 2: r = +0.2 when all specs satisfied
    reward_sat_threshold: float = -0.02   # eq 2: r >= -0.02 counts as satisfied
    reward_eps: float = 1e-9        # eq 3 denominator guard

    # ---- action denormalization ----
    # action a in [-1, 1] maps linearly to design box [x_lo, x_hi].
    action_clip: float = 1.0

    def __post_init__(self):
        if self.n_corners < 2:
            raise ValueError("n_corners must be >= 2 for multi-task structure")


DEFAULT_CONFIG = RAConfig()
