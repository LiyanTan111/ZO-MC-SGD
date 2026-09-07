"""Thompson-sampling acquisition for TuRBO (paper §2).

Generates candidate points by perturbing the TR center within the TR box
(the paper's perturbation strategy), draws ONE joint GP posterior realization
over the candidates, and returns the candidate with the minimum sampled value
(we minimize the surrogate loss). Thompson sampling — NOT EI/UCB — is the
paper's choice and is what keeps the acquisition scalable to large candidate
sets.
"""
from __future__ import annotations

import numpy as np

from .config import TuRBOConfig
from .gp_model import GPModel


def generate_candidates(center: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                        n_cand: int, rng: np.random.Generator) -> np.ndarray:
    """Perturbation candidates around `center`, clipped to the TR box [lo,hi].

    Mirrors the paper's strategy: perturb a random subset of coordinates of the
    center (prob = min(1, 20/d) per coord), filling the rest from the center,
    with the perturbation drawn uniformly within the per-dim TR box.
    """
    d = center.shape[0]
    prob = min(1.0, 20.0 / d)
    # Uniform draw within the TR box.
    u = rng.uniform(lo[None, :], hi[None, :], size=(n_cand, d))
    # Perturbation mask: each candidate perturbs a random coord subset.
    mask = rng.random((n_cand, d)) < prob
    # Ensure at least one perturbed coordinate per candidate.
    empty = ~mask.any(axis=1)
    if empty.any():
        j = rng.integers(0, d, size=int(empty.sum()))
        mask[np.where(empty)[0], j] = True
    cand = np.where(mask, u, center[None, :])
    return np.clip(cand, lo[None, :], hi[None, :])


def thompson_select(gp: GPModel, candidates: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """Draw a joint TS realization over candidates; return the argmin candidate."""
    f = gp.thompson_sample(candidates, rng)
    return candidates[int(np.argmin(f))]


def propose(gp: GPModel, center: np.ndarray, lo: np.ndarray, hi: np.ndarray,
            cfg: TuRBOConfig, rng: np.random.Generator) -> np.ndarray:
    """Full acquisition step: generate candidates + Thompson-select one point."""
    n_cand = cfg.n_cand(center.shape[0])
    cand = generate_candidates(center, lo, hi, n_cand, rng)
    return thompson_select(gp, cand, rng)
