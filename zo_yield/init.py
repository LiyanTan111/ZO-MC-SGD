"""sub-optimal X_init generators.

Used by the M1.10.2–M1.10.4 RS-necessity sweeps to start optimization
from a deliberately bad operating point — yield(X_init) ∈ [0.10, 0.30]
at the calibrated σ_scale and α — so that directed methods
have somewhere to go and random methods don't trivially luck-find the
basin.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def perturb_x_init(
    x_nominal: np.ndarray,
    seed: int,
    scale_lo: float = 0.7,
    scale_hi: float = 1.4,
    x_lo: Optional[np.ndarray] = None,
    x_hi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-design-var multiplicative perturbation of X_NOMINAL.

    Args:
      x_nominal: validated nominal design point (length n).
      seed:      reproducibility seed.
      scale_lo:  per-var multiplicative lower bound (default 0.7 = −30 %).
      scale_hi:  per-var multiplicative upper bound (default 1.4 = +40 %).
      x_lo, x_hi: optional design-box bounds; if provided, the returned
                  point is clipped to ``[x_lo, x_hi]`` (any clipped var is
                  warning-printed so the caller can see how aggressive the
                  perturbation was relative to the box).

    Returns:
      x_init: perturbed design point with the same shape as ``x_nominal``.

    Notes:
      Multiplicative (not additive) preserves device-size order-of-magnitude
      across W/L/I_bias values that span 5 decades together. Per-var
      independent draws break the carefully-tuned bias loop without
      requiring per-stage decisions.
    """
    rng = np.random.default_rng(seed)
    factors = rng.uniform(scale_lo, scale_hi, size=len(x_nominal))
    x = x_nominal * factors
    if x_lo is not None or x_hi is not None:
        if x_lo is None:
            x_lo = -np.inf * np.ones_like(x)
        if x_hi is None:
            x_hi = +np.inf * np.ones_like(x)
        x_clipped = np.clip(x, x_lo, x_hi)
        n_clipped = int(np.sum(x != x_clipped))
        if n_clipped > 0:
            print(f"[perturb_x_init] seed={seed}: {n_clipped} of {len(x)} "
                  f"vars clipped to [x_lo, x_hi]")
        x = x_clipped
    return x
