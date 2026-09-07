"""Per-task replay buffers for RobustAnalog's multi-task DDPG (§3.3).

RobustAnalog keeps ONE replay buffer per task (ξ-corner). Each buffer stores
(S, A, R, z_i) transitions for its task. The training loop draws a *stratified*
batch — one (or a fixed number) of samples from each task's buffer per gradient
step — so every task contributes to every update (paper §3.3).

Episodes in our adaptation are effectively 1-step (a single sizing proposal is
evaluated at the corner and terminates), so we store no next-state / done flag;
the critic target is just the immediate reward (γ has no bootstrap target).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np


class TaskReplayBuffer:
    """A single task's FIFO replay buffer of (state, action, reward) tuples."""

    def __init__(self, capacity: int = 1000):
        self.capacity = int(capacity)
        self._buf: deque = deque(maxlen=self.capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        self._buf.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
        ))

    def __len__(self) -> int:
        return len(self._buf)

    def sample(self, n: int, rng: np.random.Generator):
        """Sample ``n`` transitions (with replacement if n > len). Returns
        (states, actions, rewards) as stacked float32 arrays."""
        if len(self._buf) == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = rng.integers(0, len(self._buf), size=n)
        items = [self._buf[i] for i in idx]
        states = np.stack([it[0] for it in items])
        actions = np.stack([it[1] for it in items])
        rewards = np.asarray([it[2] for it in items], dtype=np.float32)
        return states, actions, rewards


class MultiTaskReplay:
    """Container of one :class:`TaskReplayBuffer` per task id."""

    def __init__(self, n_tasks: int, capacity: int = 1000):
        self.n_tasks = int(n_tasks)
        self.buffers: Dict[int, TaskReplayBuffer] = {
            i: TaskReplayBuffer(capacity) for i in range(self.n_tasks)
        }

    def push(self, task_id: int, state, action, reward) -> None:
        self.buffers[int(task_id)].push(state, action, reward)

    def ready(self, task_ids: List[int], min_size: int = 1) -> bool:
        """True iff every task in ``task_ids`` has >= ``min_size`` samples."""
        return all(len(self.buffers[i]) >= min_size for i in task_ids)

    def stratified_sample(
        self,
        task_ids: List[int],
        per_task: int,
        rng: np.random.Generator,
    ):
        """Draw ``per_task`` samples from EACH task in ``task_ids``.

        Returns a dict task_id -> (states, actions, rewards). This is the
        stratified batch of paper §3.3: every active task contributes equally
        to the gradient step, so the per-task critic/actor losses (and PCGrad
        over them) are well-defined.
        """
        out = {}
        for tid in task_ids:
            buf = self.buffers[int(tid)]
            if len(buf) == 0:
                continue
            out[int(tid)] = buf.sample(per_task, rng)
        return out
