"""Multi-task DDPG training loop for RobustAnalog (paper algorithm 1).

SPICE budget is the controlling stop condition rather than episode count.
Each ``env.step`` consumes ``len(pruned_tasks)`` SPICE; each full-corner outer
eval / pruning pass consumes K SPICE. The loop stops as soon as it can no
longer afford another interaction within ``budget``.

Faithful to paper algorithm 1:
  - random warmup actions for the first W episodes,
  - actor + exploration noise afterwards,
  - per-task critic losses -> PCGrad -> critic step,
  - per-task actor  losses -> PCGrad -> actor step,
  - k-means task pruning re-run periodically (budget permitting),
  - early exit when the current sizing passes all pruned corners.

Best sizing is tracked by mean eq-2 reward over the corners evaluated each
step (no extra SPICE spent purely on best-tracking); the final yield is
measured separately with a fresh n_eval MC pool.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .agent import DDPGAgent
from .config import RAConfig
from .multi_task_env import MultiTaskEnv
from .pcgrad import pcgrad_backward
from .pruning import prune_tasks


def _train_step(agent: DDPGAgent, env: MultiTaskEnv, batch_by_task: Dict,
                np_rng: np.random.Generator, torch_rng: torch.Generator) -> Dict:
    """One multi-task DDPG gradient step (critic then actor, both via PCGrad).

    batch_by_task: dict task_id -> (states, actions, rewards) numpy arrays.
    Returns a small dict of scalar losses for logging.
    """
    task_ids = sorted(batch_by_task.keys())
    dev = agent.device

    # ---- critic update: per-task MSE(Q(S,A,z_i), R_i) (1-step terminal) ----
    critic_losses = []
    for tid in task_ids:
        s, a, r = batch_by_task[tid]
        s_t = torch.as_tensor(s, dtype=torch.float32, device=dev)
        a_t = torch.as_tensor(a, dtype=torch.float32, device=dev)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=dev)
        z = agent.task_onehot(np.full(len(s), tid))
        q = agent.critic(s_t, a_t, z)
        critic_losses.append(((q - r_t) ** 2).mean())

    agent.critic_opt.zero_grad(set_to_none=True)
    pcgrad_backward(critic_losses, list(agent.critic.parameters()), rng=torch_rng)
    agent.critic_opt.step()

    # ---- actor update: per-task -mean Q(S, mu(S), z_i) ----
    actor_losses = []
    for tid in task_ids:
        s, a, r = batch_by_task[tid]
        s_t = torch.as_tensor(s, dtype=torch.float32, device=dev)
        z = agent.task_onehot(np.full(len(s), tid))
        a_pred = agent.actor(s_t)
        actor_losses.append(-agent.critic(s_t, a_pred, z).mean())

    agent.actor_opt.zero_grad(set_to_none=True)
    pcgrad_backward(actor_losses, list(agent.actor.parameters()), rng=torch_rng)
    agent.actor_opt.step()

    agent.soft_update()
    return dict(
        critic_loss=float(torch.stack(critic_losses).mean().item()),
        actor_loss=float(torch.stack(actor_losses).mean().item()),
    )


def train_robustanalog(
    env: MultiTaskEnv,
    agent: DDPGAgent,
    cfg: RAConfig,
    budget: int,
    x_init_real: np.ndarray,
    seed: int = 0,
    verbose: bool = False,
) -> Dict:
    """Run the RobustAnalog training loop under a SPICE ``budget``.

    Returns a result dict with the best sizing found (real units), its
    normalized action, SPICE consumed, episode count, and a history of
    (episode, spice, mean_reward) tuples.
    """
    np_rng = np.random.default_rng(seed)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(seed)
    torch.manual_seed(seed)

    K = env.num_tasks
    W = cfg.warmup_episodes
    action_dim = env.action_dim

    history = []

    def action_from_x(x_real: np.ndarray) -> np.ndarray:
        """Inverse of env.denormalize: real sizing -> action in [-1, 1]."""
        x = np.clip(np.asarray(x_real, float), env.x_lo, env.x_hi)
        a = 2.0 * (x - env.x_lo) / (env.x_hi - env.x_lo) - 1.0
        return np.clip(a, -1.0, 1.0)

    # ---- initial full-corner pass on x_init (doubles as first prune input) ----
    # A design may only be RETURNED after a full-corner outer eval validates it
    # (ranked by corners-passed, then mean reward). x_init is always a
    # candidate. This avoids overfitting the returned design to the 2-4 pruned
    # corners (cheap pruned rewards still drive DDPG training + pruning below).
    best = dict(x_real=np.asarray(x_init_real, float),
                action=action_from_x(x_init_real),
                n_pass=-1, mean_reward=-np.inf, min_reward=-np.inf,
                source="x_init")
    pruned = list(range(K))
    if env.spice_used + K <= budget:
        init_pass = env.full_corner_pass(x_init_real, is_action=False)
        best.update(n_pass=init_pass["n_corners_pass"],
                    mean_reward=init_pass["mean_reward"],
                    min_reward=init_pass["min_reward"], source="x_init")
        pruned = prune_tasks(init_pass["perf"], init_pass["eq3_rewards"],
                             cfg, seed=cfg.xi_corner_seed)
        history.append(dict(episode=0, spice=env.spice_used,
                            mean_reward=init_pass["mean_reward"],
                            n_pass=init_pass["n_corners_pass"],
                            event="init_outer_eval"))
    if verbose:
        print(f"  [RA] init pass: spice={env.spice_used} pruned={pruned} "
              f"n_pass={best['n_pass']}/{K} mean_r={best['mean_reward']:.4f}",
              flush=True)

    def better(a_pass, a_mean, b_pass, b_mean):
        """Rank designs by corners-passed, then mean reward (yield proxy)."""
        return (a_pass, a_mean) > (b_pass, b_mean)

    episode = 0
    steps_since_prune = 0
    S0 = env.task_state(0)   # nominal-corner query state for action selection
    # Incumbent: best design (by cheap pruned-corner score) since the last
    # outer eval — this is what gets full-corner-validated at the next reprune.
    incumbent = None

    while env.spice_used + len(pruned) <= budget:
        episode += 1
        steps_since_prune += 1

        if episode <= W:
            action = np_rng.uniform(-1.0, 1.0, size=action_dim)
        else:
            action = agent.act(S0, np_rng)

        rewards, info = env.step(action, pruned)   # len(pruned) SPICE
        for tid in pruned:
            agent_state = env.task_state(tid)
            env_buf_push(agent, tid, agent_state, action, rewards[tid])

        mean_r = float(np.mean(list(rewards.values())))
        min_r = float(np.min(list(rewards.values())))
        score = (min_r, mean_r)
        if incumbent is None or score > incumbent["score"]:
            incumbent = dict(score=score, x_real=info["x_real"],
                             action=np.asarray(action, float), episode=episode)

        # ---- gradient update once past warmup and buffers populated ----
        if episode > W and agent.buffers.ready(pruned, min_size=1):
            per_task = min(cfg.batch_size, max(1, min(
                len(agent.buffers.buffers[t]) for t in pruned)))
            for _ in range(cfg.train_iters_per_step):
                batch = agent.buffers.stratified_sample(pruned, per_task, np_rng)
                if batch:
                    _train_step(agent, env, batch, np_rng, torch_rng)

        # ---- periodic full-corner validation of the incumbent + reprune ----
        can_outer = env.spice_used + K + len(pruned) <= budget
        if steps_since_prune >= cfg.reprune_every and can_outer and incumbent:
            op = env.full_corner_pass(incumbent["x_real"], is_action=False)
            if better(op["n_corners_pass"], op["mean_reward"],
                      best["n_pass"], best["mean_reward"]):
                best.update(x_real=op["x_real"], action=incumbent["action"],
                            n_pass=op["n_corners_pass"],
                            mean_reward=op["mean_reward"],
                            min_reward=op["min_reward"],
                            source=f"outer_eval_ep{episode}")
            pruned = prune_tasks(op["perf"], op["eq3_rewards"], cfg,
                                 seed=cfg.xi_corner_seed)
            history.append(dict(episode=episode, spice=env.spice_used,
                                mean_reward=op["mean_reward"],
                                n_pass=op["n_corners_pass"],
                                event="reprune_outer_eval"))
            steps_since_prune = 0
            incumbent = None
            if verbose:
                tag = " (ALL PASS)" if op["all_pass"] else ""
                print(f"  [RA] ep={episode} validate+reprune: "
                      f"spice={env.spice_used} pruned={pruned} "
                      f"n_pass={op['n_corners_pass']}/{K} "
                      f"mean_r={op['mean_reward']:.4f}{tag}", flush=True)
            # Paper algorithm 1: passing all corners exits the INNER loop to
            # re-prune (done above), NOT terminate — the SPICE budget is the
            # sole terminator so we keep refining until it is spent.

    return dict(
        x_best_real=np.asarray(best["x_real"], float),
        action_best=np.asarray(best["action"], float),
        best_n_pass=int(best["n_pass"]),
        best_mean_reward=float(best["mean_reward"]),
        best_min_reward=float(best["min_reward"]),
        best_source=best["source"],
        spice_used=int(env.spice_used),
        episodes=int(episode),
        n_pruned_final=int(len(pruned)),
        history=history,
    )


def env_buf_push(agent: DDPGAgent, task_id: int, state, action, reward) -> None:
    """Push a transition into the agent's per-task replay buffer."""
    agent.buffers.push(task_id, state, action, reward)
