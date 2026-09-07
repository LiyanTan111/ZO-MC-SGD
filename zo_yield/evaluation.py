"""Yield + spec-margin evaluation under Monte Carlo over rho(xi).

A `Spec` declares one performance constraint with three callbacks: how to
extract the value from the simulator's metric dict, whether a value satisfies
the constraint, and (signed) margin. Margin convention: **positive = violated
by that amount, negative = met with that much slack** -- so `satisfies(value)
== margin(value) <= 0`.

`evaluate_design(...)` runs N_mc simulations, applies all specs, and returns:

  * 'yield'              : P[all specs simultaneously satisfied]
  * 'per_spec_passrate'  : dict[spec_name -> P[that spec satisfied]]
  * 'per_spec_margin'    : dict[spec_name -> array(N_mc,)] of margins
  * 'E_loss'             : mean of training loss across MC
  * 'loss_samples'       : array(N_mc,)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class Spec:
    """One scalar performance constraint.

    Args:
      name      : human-readable identifier (used as dict key in outputs).
      extract   : pulls the relevant value from a dict of simulator metrics.
      satisfies : returns True iff the constraint holds.
      margin    : signed margin. Positive = violated by that amount, negative =
                  met with that much slack. Conventionally the units are
                  whatever the spec measures (dB, Hz, deg, W, ...).
    """

    name: str
    extract: Callable[[dict], float]
    satisfies: Callable[[float], bool]
    margin: Callable[[float], float]


def evaluate_design(
    simulator,
    x: np.ndarray,
    sampler,
    specs: Sequence[Spec],
    n_mc: int = 5000,
    rng: np.random.Generator | None = None,
    metrics_fn=None,
) -> dict:
    """Monte Carlo yield + per-spec evaluation of design `x`.

    Args:
      simulator  : object with `evaluate_with_metrics(x, xi) -> (loss, metrics)`.
                   Falls back to `evaluate(x, xi)` if metrics extraction is
                   unavailable (then per-spec margins are NaN).
      x          : design vector.
      sampler    : ρ(ξ) sampler with `.sample(n_mc, rng) -> ndarray(n_mc, d)`.
      specs      : list of Spec objects.
      n_mc       : number of MC samples (default 5000).
      rng        : numpy generator (use a fixed seed to compare designs fairly).
      metrics_fn : optional override -- callable (sim, x, xi) -> (loss, metrics)
                   for simulators that don't expose `evaluate_with_metrics`.
    """
    rng = rng or np.random.default_rng()
    xis = sampler.sample(n_mc, rng=rng)

    losses = np.empty(n_mc)
    margins = {s.name: np.empty(n_mc) for s in specs}
    passes = {s.name: np.empty(n_mc, dtype=bool) for s in specs}
    all_pass = np.empty(n_mc, dtype=bool)

    has_emm = hasattr(simulator, "evaluate_with_metrics")
    for j, xi in enumerate(xis):
        try:
            if metrics_fn is not None:
                loss, metrics = metrics_fn(simulator, x, xi)
            elif has_emm:
                loss, metrics = simulator.evaluate_with_metrics(x, xi)
            else:
                loss = simulator.evaluate(x, xi)
                metrics = {}
        except Exception:
            # treat as a yield miss (failed simulation -> all specs violated)
            loss = float("inf")
            metrics = {}

        losses[j] = loss
        ap = True
        for s in specs:
            try:
                v = s.extract(metrics)
                m = float(s.margin(v))
                ok = bool(s.satisfies(v))
            except Exception:
                m = float("inf")  # missing metric -> violated
                ok = False
            margins[s.name][j] = m
            passes[s.name][j] = ok
            ap = ap and ok
        all_pass[j] = ap

    return dict(
        yield_=float(all_pass.mean()),
        per_spec_passrate={n: float(passes[n].mean()) for n in passes},
        per_spec_margin={n: margins[n] for n in margins},
        E_loss=float(losses.mean()),
        loss_samples=losses,
        n_mc=int(n_mc),
    )


def to_serializable(eval_result: dict) -> dict:
    """Convert the dict returned by evaluate_design to a JSON-serializable form."""
    out = dict(eval_result)
    out["per_spec_margin"] = {k: v.tolist() for k, v in out["per_spec_margin"].items()}
    out["loss_samples"] = out["loss_samples"].tolist()
    out["yield"] = out.pop("yield_")
    return out
