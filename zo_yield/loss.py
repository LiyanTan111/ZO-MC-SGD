"""Smooth-shortfall loss module.

Each :class:`Penalty` describes one constraint. ``_smooth_shortfall`` evaluates
``softplus(α·δ)/α`` (or ``max(0, δ)`` when ``alpha is None``).

Unit conventions:

* ``delta`` is in the metric's natural unit (gain in dB, log10(UGBW) in
  decade-of-Hz, PM in degrees, power in µW). ``Penalty.extract`` is responsible
  for putting the metric into its natural unit.
* ``alpha`` is in 1 / (metric natural unit). E.g. ``alpha_gain = 0.5 /dB``
  gives a soft-boundary half-width of ~2 dB.
* ``weight`` maps the smoothed shortfall from metric units to loss units.
  Pick weights so that one natural-unit violation contributes loss ≈ 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence


@dataclass
class Penalty:
    """One scalar constraint on a circuit metric.

    Attributes:
      name    : human-readable identifier.
      extract : pulls the relevant metric value from a sim-output dict, in the
                natural unit (e.g. dB, log10(Hz), degrees, µW).
      kind    : 'ge' for ``value >= target`` constraints; 'le' for
                ``value <= target``.
      target  : threshold in the same unit as ``extract``'s output.
      weight  : multiplied into the smoothed shortfall.
      alpha   : softplus sharpness (1 / natural unit). ``None`` = hard ReLU.
    """
    name: str
    extract: Callable[[dict], float]
    kind: Literal["ge", "le"]
    target: float
    weight: float
    alpha: Optional[float] = None


def _smooth_shortfall(delta: float, alpha: Optional[float]) -> float:
    """Smoothed shortfall.

    delta = (target - value) for 'ge' constraints, (value - target) for 'le'.
    delta > 0 means the constraint is violated by that amount.

    Hard mode (``alpha is None``): returns ``max(0, delta)`` exactly.

    Smooth mode: returns ``softplus(alpha * delta) / alpha`` with:
      * as alpha -> inf, approaches max(0, delta);
      * at delta = 0, value is ``ln(2) / alpha`` (small residual);
      * derivative w.r.t. delta is ``sigmoid(alpha * delta)`` everywhere.
    """
    if alpha is None:
        return max(0.0, delta)
    z = alpha * delta
    # numerically stable upper tail (softplus(z) ≈ z when z >> 0)
    if z > 20.0:
        return delta
    # softplus(z) is fine in lower tail too (log1p(exp(z)) ≈ exp(z) when z << 0)
    return math.log1p(math.exp(z)) / alpha


def evaluate_penalty(p: Penalty, sim: dict) -> float:
    """Apply Penalty p to a sim-output dict; returns ``weight * smoothed shortfall``."""
    v = p.extract(sim)
    if p.kind == "ge":
        delta = p.target - v
    elif p.kind == "le":
        delta = v - p.target
    else:
        raise ValueError(f"unknown kind={p.kind!r}; expected 'ge' or 'le'")
    return p.weight * _smooth_shortfall(delta, p.alpha)


def combined_loss(
    penalties: Sequence[Penalty],
    sim: dict,
    objective: float = 0.0,
) -> float:
    """Combined loss = ``objective + Σ_k weight_k · _smooth_shortfall_k``.

    ``objective`` is a free term to minimize directly (e.g. raw power in W,
    or ``-gain_dB`` for a gain-maximization bonus). It bypasses the smoothing
    machinery and adds linearly into the loss.
    """
    return objective + sum(evaluate_penalty(p, sim) for p in penalties)
