"""Projection operators onto common feasible sets."""
from __future__ import annotations

from typing import Optional

import numpy as np


def project_box(
    x: np.ndarray,
    lo: Optional[np.ndarray] = None,
    hi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project x onto the box [lo, hi] (element-wise clip). None disables that side."""
    out = x.copy()
    if lo is not None:
        out = np.maximum(out, lo)
    if hi is not None:
        out = np.minimum(out, hi)
    return out


def project_simplex(x: np.ndarray, z: float = 1.0) -> np.ndarray:
    """Euclidean projection of x onto { y >= 0 : sum y = z }. (Duchi et al. 2008.)"""
    if z <= 0:
        raise ValueError("simplex sum target z must be positive")
    n = x.shape[0]
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - z
    rho = np.nonzero(u - cssv / (np.arange(n) + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(x - theta, 0.0)
