"""Unit tests for the RobustAnalog DDPG agent (3 tests)."""
import numpy as np
import torch

from methods.robustanalog.agent import DDPGAgent
from methods.robustanalog.config import RAConfig


def _agent(state_dim=5, action_dim=7, n_tasks=5):
    cfg = RAConfig(n_corners=n_tasks, hidden_size=32)
    return DDPGAgent(state_dim, action_dim, n_tasks, cfg, seed=0), cfg


def test_actor_output_dim_matches_action_dim():
    agent, _ = _agent(state_dim=5, action_dim=7, n_tasks=5)
    s = np.zeros(5, dtype=np.float32)
    a = agent.act_deterministic(s)
    assert a.shape == (7,)


def test_actor_output_in_minus_one_one():
    agent, _ = _agent(state_dim=5, action_dim=7, n_tasks=5)
    rng = np.random.default_rng(0)
    # Deterministic (tanh-squashed) output must lie strictly in [-1, 1].
    for _ in range(20):
        s = rng.standard_normal(5).astype(np.float32)
        a = agent.act_deterministic(s)
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_critic_takes_state_action_taskid_returns_scalar():
    agent, _ = _agent(state_dim=5, action_dim=7, n_tasks=5)
    s = torch.zeros(3, 5)
    a = torch.zeros(3, 7)
    z = agent.task_onehot([0, 1, 2])
    q = agent.critic(s, a, z)
    assert q.shape == (3,)            # one scalar Q per batch element
    assert z.shape == (3, 5)          # task-id one-hot width == n_tasks
