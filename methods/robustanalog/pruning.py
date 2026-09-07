"""k-means task pruning for RobustAnalog (paper §3.4).

Pruning shrinks the active training-task set so the agent focuses SPICE on the
corners that actually bind. Algorithm (paper §3.4):

  1. Simulate the current best sizing on ALL K corners -> perf matrix (K x M).
  2. k-means cluster the corners by their performance rows.
  3. In each cluster, pick the corner with the WORST per-corner reward
     (eq-3 raw reward, NOT the post-clip eq-2 value).
  4. Return the worst-corner-per-cluster set as the next training subset.

Number of clusters is chosen adaptively in [2, 4] via silhouette score
(paper used 2 for strongARM; the spec allows 2-4 for our circuits). A nominal
corner (task 0, sampled at ξ=0) is optionally always included to aid learning
(paper §3.4).

k-means (Lloyd) and the silhouette score are implemented in pure numpy. This
deliberately avoids sklearn/scipy here: this module is imported in the same
process as torch, and scipy's compiled spatial backend clashes with the
libstdc++ that torch loads. Pure-numpy keeps pruning robust and dependency-light.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .config import RAConfig


def _standardize(x: np.ndarray) -> np.ndarray:
    """Column z-score so no single metric (e.g. UGBW ~1e8) dominates distance."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (x - mu) / sd


def _kmeans(x: np.ndarray, k: int, seed: int, n_init: int = 10,
            max_iter: int = 100) -> Tuple[np.ndarray, float]:
    """Lloyd's k-means with k-means++-style multi-restart. Returns
    (labels, inertia) for the best (lowest-inertia) restart."""
    n = x.shape[0]
    best_labels, best_inertia = None, np.inf
    for r in range(n_init):
        rng = np.random.default_rng(seed * 1000 + r)
        # k-means++ seeding.
        centers = np.empty((k, x.shape[1]))
        centers[0] = x[rng.integers(n)]
        d2 = ((x - centers[0]) ** 2).sum(axis=1)
        for c in range(1, k):
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            centers[c] = x[rng.choice(n, p=probs)]
            d2 = np.minimum(d2, ((x - centers[c]) ** 2).sum(axis=1))
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                members = x[labels == c]
                if len(members) > 0:
                    centers[c] = members.mean(axis=0)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels, best_inertia


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (pure numpy). Needs >= 2 clusters."""
    n = x.shape[0]
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    # Pairwise Euclidean distances.
    dmat = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    sil = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            sil[i] = 0.0
            continue
        a = dmat[i, same].mean()
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            mask = labels == c
            if mask.sum() > 0:
                b = min(b, dmat[i, mask].mean())
        sil[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(sil.mean())


def choose_n_clusters(features: np.ndarray, cluster_range, seed: int) -> int:
    """Pick the cluster count in ``cluster_range`` maximizing silhouette."""
    lo, hi = int(cluster_range[0]), int(cluster_range[1])
    n = features.shape[0]
    hi = min(hi, n - 1)
    if hi < lo:
        return max(2, min(lo, max(2, n - 1)))
    best_k, best_score = lo, -np.inf
    for k in range(lo, hi + 1):
        labels, _ = _kmeans(features, k, seed)
        if len(np.unique(labels)) < 2:
            continue
        score = _silhouette(features, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def prune_tasks(
    perf: np.ndarray,
    corner_rewards: np.ndarray,
    cfg: RAConfig,
    seed: int = 0,
    include_nominal: Optional[bool] = None,
) -> List[int]:
    """Return the pruned training-task subset (corner ids).

    Args:
      perf           : (K, M) performance matrix (raw metric values).
      corner_rewards : (K,) eq-3 raw reward per corner (lower = worse).
      cfg            : RAConfig (cluster_range, include_nominal).
      seed           : k-means seed for reproducibility.
      include_nominal: override cfg.include_nominal.

    Returns a sorted list of unique corner ids: the worst-reward corner in
    each cluster, plus the nominal corner (id 0) if requested.
    """
    perf = np.asarray(perf, dtype=float)
    corner_rewards = np.asarray(corner_rewards, dtype=float)
    K = perf.shape[0]
    inc_nominal = cfg.include_nominal if include_nominal is None else include_nominal

    # Replace non-finite metric rows with column means so clustering is robust.
    finite = np.isfinite(perf)
    if not finite.all():
        col_mean = np.nanmean(np.where(finite, perf, np.nan), axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        perf = np.where(finite, perf, col_mean[None, :])

    feats = _standardize(perf)

    if K <= 2:
        selected = list(range(K))
    else:
        n_clusters = choose_n_clusters(feats, cfg.cluster_range, seed)
        labels, _ = _kmeans(feats, n_clusters, seed)
        selected = []
        for c in np.unique(labels):
            members = np.where(labels == c)[0]
            worst = members[int(np.argmin(corner_rewards[members]))]
            selected.append(int(worst))

    if inc_nominal and 0 not in selected:
        selected.append(0)

    return sorted(set(int(i) for i in selected))
