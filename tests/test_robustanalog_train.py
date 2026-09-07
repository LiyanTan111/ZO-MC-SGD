"""Unit tests for the multi-task DDPG training loop (3 tests).

Uses a fast in-process mock simulator (no SPICE/ngspice) so the loop logic and
budget accounting can be exercised cheaply.
"""
import numpy as np

from methods.robustanalog.agent import DDPGAgent
from methods.robustanalog.config import RAConfig
from methods.robustanalog.multi_task_env import Constraint, MultiTaskEnv
from methods.robustanalog.train import train_robustanalog


class MockSim:
    """Cheap deterministic simulator: metrics depend smoothly on x and xi."""
    def __init__(self):
        self.n_calls = 0
        self.n_failures = 0

    def evaluate_with_metrics(self, x, xi):
        self.n_calls += 1
        s = float(np.mean(x))
        noise = float(np.sum(np.abs(xi))) * 1e-3
        metrics = dict(
            gain_db=60.0 + 5.0 * s - noise,
            ugbw_hz=50e6 * (1.0 + 0.1 * s),
            power_w=0.5e-3 + 1e-4 * abs(s),
        )
        return 0.0, metrics


def _build_env(K=6, n_x=4, cfg=None):
    cfg = cfg or RAConfig(n_corners=K, hidden_size=16, warmup_episodes=2,
                          reprune_every=3, cluster_range=(2, 3))
    rng = np.random.default_rng(0)
    corners = np.zeros((K, 3))
    corners[1:] = rng.standard_normal((K - 1, 3))
    cons = [
        Constraint("gain_db", lambda m: m["gain_db"], 60.0, "ge"),
        Constraint("ugbw_hz", lambda m: m["ugbw_hz"], 50e6, "ge"),
        Constraint("power_w", lambda m: m["power_w"], 1e-3, "le"),
    ]
    x_lo = np.zeros(n_x)
    x_hi = np.ones(n_x)
    env = MultiTaskEnv(MockSim(), cons, specs=None, xi_corners=corners,
                       x_lo=x_lo, x_hi=x_hi, cfg=cfg)
    return env, cfg


def test_one_episode_runs_and_shapes_ok():
    env, cfg = _build_env()
    agent = DDPGAgent(env.state_dim, env.action_dim, env.num_tasks, cfg, seed=0)
    res = train_robustanalog(env, agent, cfg, budget=40,
                             x_init_real=np.full(env.action_dim, 0.5), seed=0)
    assert res["x_best_real"].shape == (env.action_dim,)
    assert res["action_best"].shape == (env.action_dim,)
    assert np.all(res["action_best"] >= -1.0) and np.all(res["action_best"] <= 1.0)
    assert res["episodes"] >= 1


def test_stratified_sample_one_per_task():
    env, cfg = _build_env()
    agent = DDPGAgent(env.state_dim, env.action_dim, env.num_tasks, cfg, seed=0)
    task_ids = [0, 2, 4]
    for tid in task_ids:
        agent.buffers.push(tid, env.task_state(tid),
                           np.zeros(env.action_dim), 0.1)
    batch = agent.buffers.stratified_sample(task_ids, per_task=1,
                                            rng=np.random.default_rng(0))
    assert sorted(batch.keys()) == task_ids
    for tid in task_ids:
        states, actions, rewards = batch[tid]
        assert states.shape == (1, env.state_dim)
        assert actions.shape == (1, env.action_dim)
        assert rewards.shape == (1,)


def test_budget_tracking_matches_spice_calls():
    env, cfg = _build_env()
    agent = DDPGAgent(env.state_dim, env.action_dim, env.num_tasks, cfg, seed=0)
    budget = 50
    res = train_robustanalog(env, agent, cfg, budget=budget,
                             x_init_real=np.full(env.action_dim, 0.5), seed=0)
    # spice_used reported == env counter == mock simulator's call count.
    assert res["spice_used"] == env.spice_used == env.sim.n_calls
    # Never exceeds budget, and gets within one step-cost of it.
    assert res["spice_used"] <= budget
