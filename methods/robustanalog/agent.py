"""DDPG actor + critic networks for RobustAnalog (paper §4.2).

Key methodological detail (paper §3.3):
  - Actor  μ(S)        -> action A in [-1, 1]^n_action  (task-agnostic input)
  - Critic Q(S, A, z_i) -> scalar                        (task id z_i is input)

The actor is task-agnostic in the sense that it takes only the state S (which,
in our adaptation, IS the per-corner descriptor); the critic additionally
consumes the task-id one-hot z_i. Getting this split wrong defeats the
multi-task contribution, so it is implemented and unit-tested explicitly.

Both are 4-layer MLPs (3 hidden + output), hidden width 256, ReLU hidden,
tanh on the actor output (squash to [-1, 1]), linear critic output.
"""
from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn

from .config import RAConfig
from .replay_buffer import MultiTaskReplay


def _mlp(in_dim: int, out_dim: int, hidden: int, n_hidden: int,
         out_activation: nn.Module | None = None) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    if out_activation is not None:
        layers += [out_activation]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """μ(S) -> A in [-1, 1]^action_dim (tanh-squashed)."""

    def __init__(self, state_dim: int, action_dim: int, cfg: RAConfig):
        super().__init__()
        self.net = _mlp(state_dim, action_dim, cfg.hidden_size,
                        cfg.n_hidden_layers, out_activation=nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    """Q(S, A, z_i) -> scalar. z_i is a task-id one-hot of width n_tasks."""

    def __init__(self, state_dim: int, action_dim: int, n_tasks: int,
                 cfg: RAConfig):
        super().__init__()
        in_dim = state_dim + action_dim + n_tasks
        self.net = _mlp(in_dim, 1, cfg.hidden_size, cfg.n_hidden_layers,
                        out_activation=None)

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                task_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action, task_onehot], dim=-1)
        return self.net(x).squeeze(-1)


class DDPGAgent:
    """DDPG agent bundling actor, critic, their target nets, and optimizers.

    Provides ``act`` (with exploration noise) and ``act_deterministic`` for
    the training loop; the loop itself owns the gradient updates so it can
    route the K per-task losses through PCGrad (see train.py).
    """

    def __init__(self, state_dim: int, action_dim: int, n_tasks: int,
                 cfg: RAConfig, device: str = "cpu",
                 seed: int | None = None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_tasks = n_tasks
        if seed is not None:
            torch.manual_seed(seed)

        self.actor = Actor(state_dim, action_dim, cfg).to(self.device)
        self.critic = Critic(state_dim, action_dim, n_tasks, cfg).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Per-task replay buffers; the training loop pushes/samples here.
        self.buffers = MultiTaskReplay(n_tasks, capacity=cfg.replay_capacity)

    # ---- action selection ------------------------------------------------ #
    def act(self, state: np.ndarray, rng: np.random.Generator,
            noise_scale: float | None = None) -> np.ndarray:
        """Return a noisy action in [-1, 1] for the given state (exploration)."""
        a = self.act_deterministic(state)
        sigma = self.cfg.explore_noise if noise_scale is None else noise_scale
        if sigma > 0:
            a = a + sigma * rng.standard_normal(self.action_dim)
        return np.clip(a, -self.cfg.action_clip, self.cfg.action_clip)

    def act_deterministic(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(state, dtype=np.float32),
                                device=self.device).reshape(1, -1)
            a = self.actor(s).cpu().numpy().reshape(-1)
        return a

    # ---- helpers --------------------------------------------------------- #
    def task_onehot(self, task_ids) -> torch.Tensor:
        ids = torch.as_tensor(np.asarray(task_ids).reshape(-1), dtype=torch.long,
                              device=self.device)
        return torch.nn.functional.one_hot(ids, num_classes=self.n_tasks).float()

    def soft_update(self) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for tp, p in zip(self.actor_target.parameters(),
                             self.actor.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
            for tp, p in zip(self.critic_target.parameters(),
                             self.critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
