"""Unit tests for k-means task pruning (2 tests)."""
import numpy as np

from methods.robustanalog.config import RAConfig
from methods.robustanalog.pruning import prune_tasks


def test_returns_cluster_count_ids_in_range():
    rng = np.random.default_rng(0)
    K = 10
    perf = rng.standard_normal((K, 3))
    rewards = rng.standard_normal(K)
    cfg = RAConfig(n_corners=K, cluster_range=(2, 4), include_nominal=False)
    sel = prune_tasks(perf, rewards, cfg, seed=0)
    assert 2 <= len(sel) <= 4
    assert all(0 <= i < K for i in sel)
    assert len(set(sel)) == len(sel)        # unique


def test_each_selected_is_worst_reward_in_its_cluster():
    # Two well-separated clusters: corners 0-4 (metric ~0), 5-9 (metric ~100).
    K = 10
    perf = np.zeros((K, 1))
    perf[5:] = 100.0
    # Within cluster A, corner 2 is worst; within cluster B, corner 7 is worst.
    rewards = np.array([0.5, 0.4, -0.9, 0.3, 0.2,    # cluster A -> argmin = 2
                        0.6, 0.7, -0.8, 0.9, 0.1])   # cluster B -> argmin = 7
    cfg = RAConfig(n_corners=K, cluster_range=(2, 2), include_nominal=False)
    sel = prune_tasks(perf, rewards, cfg, seed=0)
    assert set(sel) == {2, 7}
