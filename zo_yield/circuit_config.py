"""Per-benchmark calibration inputs, version-controlled under ``configs/``.

Each ``configs/<circuit>.json`` holds the three calibrated quantities every
experiment needs, as reported in Table I of the paper:

* ``sigma_scale`` -- the process-variation scale chosen by the benchmark
  admission gate, so that yield at ``X_NOMINAL`` falls in the target band;
* ``alphas``      -- the per-specification softplus sharpness chosen to
  maximize the loss--yield Spearman rank correlation;
* ``x_init``      -- the shared hard starting design (real coordinates), which
  every method is started from.

These are *inputs*, not results: they are fixed before optimization and shared
by all methods, which is what makes the comparison fair. They are committed so
a clone reproduces the reported numbers exactly; ``scripts/sigma_scale_
calibration.py``, ``scripts/loss_yield_calibration.py`` and
``scripts/calibrate_x_init_hard.py`` regenerate them from scratch.
"""
from __future__ import annotations

import json
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")

#: Benchmark circuits, in the order used by the paper's tables.
CIRCUITS = (
    "cs_amp",
    "cs_amp_3stage",
    "cs_amp_5stage",
    "cs_se_miller",
    "cs_se_miller_3stage",
)


def load_circuit_config(circuit: str) -> dict:
    """Return the calibration inputs for ``circuit``.

    Args:
      circuit: one of :data:`CIRCUITS`.

    Returns:
      dict with keys ``sigma_scale`` (float), ``alphas`` (dict), ``x_init``
      (``np.ndarray`` in real design coordinates), ``spec_module`` (str) and
      the bookkeeping fields stored alongside them.
    """
    path = os.path.join(CONFIG_DIR, f"{circuit}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing calibration config {path}. Known circuits: "
            f"{', '.join(CIRCUITS)}. Regenerate with "
            f"scripts/sigma_scale_calibration.py, scripts/loss_yield_calibration.py "
            f"and scripts/calibrate_x_init_hard.py."
        )
    with open(path) as f:
        cfg = json.load(f)
    cfg["x_init"] = np.asarray(cfg["x_init"], dtype=float)
    cfg.setdefault("alphas", {})
    return cfg


def load_spec_module(circuit: str):
    """Import and return the benchmark spec module for ``circuit``."""
    cfg = load_circuit_config(circuit)
    return __import__(cfg["spec_module"], fromlist=["spec"])
