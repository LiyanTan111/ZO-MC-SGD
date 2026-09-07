""", — variance-reduction primitives.

Single ZO gradient estimate with optional Common Random Numbers (CRN)
and optional antithetic-ξ sampling. Standalone helpers that don't
modify the production ZO-MC-SGD class — validates these
on synthetic before wires them into the optimizers.

API:
  zo_grad_with_vr(simulator, sampler, x, eps, n_mc, v, xi_seed,
                   crn=False, antithetic=False) -> np.ndarray

  draw_xi_batch(sampler, n_mc, seed, antithetic=False) -> np.ndarray
"""
from __future__ import annotations

import numpy as np

from .estimators import phi_factor


def draw_xi_batch(sampler, n_mc: int, seed: int,
                   antithetic: bool = False) -> np.ndarray:
    """Draw an MC batch of ξ from the sampler, optionally antithetically.

    Antithetic mode pairs each draw with its reflection through the
    sampler's mean: for each ξ in the first half, the second half
    contains 2·μ − ξ. For zero-mean samplers this reduces to the simple
    negation `−ξ`. Cuts the variance of any function symmetric (or
    monotonic) in ξ around the mean.

    Requires even ``n_mc`` when ``antithetic=True``.
    """
    if antithetic:
        if n_mc % 2 != 0:
            raise ValueError(f"antithetic requires even n_mc, got {n_mc}")
        # Use the sampler to draw the first half so all per-circuit σ /
        # mean rescaling lives inside ``sampler.sample`` (we don't have
        # to know the sampler's internal parameterization).
        rng = np.random.default_rng(seed)
        half = sampler.sample(n_mc // 2, rng=rng)
        # Mirror through the sampler's mean if it has one; otherwise
        # negate (S1/S2 use zero-mean Gaussians).
        mu = getattr(sampler, "mean", np.zeros_like(half[0]))
        mirrored = 2.0 * mu[None, :] - half
        return np.concatenate([half, mirrored], axis=0)
    rng = np.random.default_rng(seed)
    return sampler.sample(n_mc, rng=rng)


def zo_grad_with_vr(
    simulator,
    sampler,
    x: np.ndarray,
    eps: float,
    n_mc: int,
    v: np.ndarray,
    xi_seed: int,
    crn: bool = False,
    antithetic: bool = False,
) -> np.ndarray:
    """One ZO central-difference gradient with optional CRN + antithetic ξ.

    Args:
      simulator: object with ``.evaluate(x, xi) -> float``.
      sampler:   object with ``.sample(n_mc, rng=...) -> (n_mc, d_xi)`` and
                 (optionally) ``.mean``. For antithetic mode, the sample
                 is mirrored through ``sampler.mean`` (or 0).
      x:         design point (length n_x).
      eps:       FD step (in the *normalized* design space if x is
                 normalized; the caller is responsible for scaling).
      n_mc:      ξ batch size for averaging f at each FD perturbation.
      v:         FIXED ZO direction for this estimate (length n_x). The
                 caller draws v separately so the variance ablation can
                 control which random source contributes which noise.
      xi_seed:   seed for the ξ batch RNG.
      crn:       True — share ξ batch between f(x+εv) and f(x−εv).
                 False — draw independent batches for each FD side.
      antithetic: True — pair (ξ, mirrored-ξ) within each batch.

    Returns:
      ĝ: ZO gradient estimate (length n_x), already with the ``φ``
      factor applied (matches ``zo_yield.estimators.mini_batch_estimate``
      conventions).
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    n = x.shape[0]

    if crn:
        xi_batch = draw_xi_batch(sampler, n_mc, seed=xi_seed,
                                   antithetic=antithetic)
        f_plus = float(np.mean([simulator.evaluate(x + eps * v, xi)
                                 for xi in xi_batch]))
        f_minus = float(np.mean([simulator.evaluate(x - eps * v, xi)
                                  for xi in xi_batch]))
    else:
        xi_plus = draw_xi_batch(sampler, n_mc, seed=xi_seed,
                                 antithetic=antithetic)
        xi_minus = draw_xi_batch(sampler, n_mc, seed=xi_seed + 1,
                                  antithetic=antithetic)
        f_plus = float(np.mean([simulator.evaluate(x + eps * v, xi)
                                 for xi in xi_plus]))
        f_minus = float(np.mean([simulator.evaluate(x - eps * v, xi)
                                  for xi in xi_minus]))

    phi = phi_factor(n, "gaussian")
    return (phi / (2.0 * eps)) * (f_plus - f_minus) * v


def yield_estimate_with_vr(
    yield_predicate,
    sampler,
    n_mc: int,
    seed: int,
    antithetic: bool = False,
) -> float:
    """Monte Carlo yield estimate with optional antithetic ξ.

    ``yield_predicate(xi) -> bool`` returns True iff the design passes
    all specs at this ξ. Used by to compare yield-estimate
    variance under antithetic vs naive sampling — note CRN does NOT
    affect single-x yield estimates (no FD difference here), so we only
    expose ``antithetic``.
    """
    xi_batch = draw_xi_batch(sampler, n_mc, seed=seed, antithetic=antithetic)
    return float(np.mean([1.0 if yield_predicate(xi) else 0.0
                            for xi in xi_batch]))
