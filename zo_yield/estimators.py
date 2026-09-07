"""Two-point zeroth-order gradient estimators.

Following Liu et al. 2020 (IEEE SPM) and Berahas et al. 2022 (FoCM) notation:

    g_hat(x) = phi(n) / eps * [f(x + eps * v) - f(x)] * v       (forward)
    g_hat(x) = phi(n) / (2*eps) * [f(x + eps*v) - f(x - eps*v)] * v   (central)

where v is a random direction. The scaling phi(n) depends on the v distribution:

  * gaussian (v ~ N(0, I_n)):     phi(n) = 1
  * sphere   (v ~ Unif(S^{n-1})): phi(n) = n
  * coordinate (v = e_i):          phi(n) = 1 (treated as plain finite difference)

All functions accept a callable f: R^n -> R.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np


VDist = Literal["gaussian", "sphere", "coordinate"]
Mode = Literal["forward", "central"]


# --------------------------------------------------------------------------- #
# direction sampling                                                          #
# --------------------------------------------------------------------------- #
def sample_direction(n: int, v_dist: VDist, rng: np.random.Generator) -> np.ndarray:
    """Sample a single direction v in R^n; returns an (n,) array."""
    if v_dist == "gaussian":
        return rng.standard_normal(n)
    if v_dist == "sphere":
        v = rng.standard_normal(n)
        return v / np.linalg.norm(v)
    if v_dist == "coordinate":
        i = rng.integers(0, n)
        v = np.zeros(n)
        v[i] = 1.0
        return v
    raise ValueError(f"unknown v_dist={v_dist}")


def phi_factor(n: int, v_dist: VDist) -> float:
    """Smoothing-factor scaling so that E[g_hat] -> grad f as eps -> 0."""
    if v_dist == "gaussian":
        return 1.0
    if v_dist == "sphere":
        return float(n)
    if v_dist == "coordinate":
        return float(n)
    raise ValueError(f"unknown v_dist={v_dist}")


# --------------------------------------------------------------------------- #
# single-shot estimators                                                      #
# --------------------------------------------------------------------------- #
def two_point_forward(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    epsilon: float,
    v: np.ndarray,
    f_x: Optional[float] = None,
    v_dist: VDist = "gaussian",
) -> tuple[np.ndarray, int]:
    """Forward two-point estimator. Returns (gradient_estimate, n_function_calls)."""
    n = x.shape[0]
    phi = phi_factor(n, v_dist)
    if f_x is None:
        f_x = float(f(x))
        n_calls = 2
    else:
        n_calls = 1
    f_p = float(f(x + epsilon * v))
    g = (phi / epsilon) * (f_p - f_x) * v
    return g, n_calls


def two_point_central(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    epsilon: float,
    v: np.ndarray,
    v_dist: VDist = "gaussian",
) -> tuple[np.ndarray, int]:
    """Central two-point estimator. Returns (gradient_estimate, n_function_calls=2)."""
    n = x.shape[0]
    phi = phi_factor(n, v_dist)
    f_p = float(f(x + epsilon * v))
    f_m = float(f(x - epsilon * v))
    g = (phi / (2.0 * epsilon)) * (f_p - f_m) * v
    return g, 2


# --------------------------------------------------------------------------- #
# mini-batch estimator                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class EstimatorConfig:
    epsilon: float = 1e-3
    n_samples: int = 1
    v_dist: VDist = "gaussian"
    mode: Mode = "central"
    rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def get_rng(self, seed: Optional[int] = None) -> np.random.Generator:
        if self.rng is not None:
            return self.rng
        return np.random.default_rng(seed)


def mini_batch_estimate(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    epsilon: float = 1e-3,
    n_samples: int = 1,
    v_dist: VDist = "gaussian",
    mode: Mode = "central",
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, int]:
    """Average n_samples two-point estimates of grad f at x.

    Returns (gradient_estimate, total_function_evaluations).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = x.shape[0]

    grads = np.zeros(n)
    total_calls = 0

    if mode == "forward":
        f_x = float(f(x))
        total_calls += 1
        for _ in range(n_samples):
            v = sample_direction(n, v_dist, rng)
            g, c = two_point_forward(f, x, epsilon, v, f_x=f_x, v_dist=v_dist)
            grads += g
            total_calls += c
    elif mode == "central":
        for _ in range(n_samples):
            v = sample_direction(n, v_dist, rng)
            g, c = two_point_central(f, x, epsilon, v, v_dist=v_dist)
            grads += g
            total_calls += c
    else:
        raise ValueError(f"unknown mode={mode}")

    return grads / n_samples, total_calls
