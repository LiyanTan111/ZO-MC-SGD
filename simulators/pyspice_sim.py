"""ngspice-shared-library wrapper.

We bypass PySpice's high-level netlist API (which mangles ``.param``) and instead
hand a fully-substituted raw netlist string straight to NgSpiceShared. This keeps
parameterization fully in Python: the user supplies a Python ``str.format``-style
template and we render in numerical values for design vars and process variations.

Failures (parser errors, ngspice convergence failures, non-finite results) are
caught and turned into ``LOSS_PENALTY``. Optimizers can therefore continue without
exception handling on their side.
"""
from __future__ import annotations

import math
import os
import threading
from typing import Callable, Optional, Sequence

import numpy as np

from .base import BlackBoxSimulator


# A single NgSpiceShared instance per process is the recommended pattern; reusing
# it across many simulations is much faster than instantiating one per call.
_NG_LOCK = threading.Lock()
_NG = None


def _ng() -> "NgSpiceShared":
    global _NG
    if _NG is None:
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared

        _NG = NgSpiceShared.new_instance()
    return _NG


class PySpiceSimulator(BlackBoxSimulator):
    """Black-box wrapper that runs raw netlists via NgSpiceShared.

    Args:
        netlist_template: ``str.format``-style template; placeholders for both
            design and variation variables. The first non-empty line must be the
            netlist title (ngspice convention). Use *Python* numerical placeholders
            like ``W={w1}`` -- avoid ngspice ``.param`` indirection.
        design_var_names / variation_var_names: ordered lists; x[i] is substituted
            into ``{design_var_names[i]}`` and similarly for xi.
        analysis_command: an ngspice control command (e.g. ``ac dec 20 1 1g``).
        extractor: callable ``extractor(ngspice_shared, x, xi) -> float``. Receives
            the live NgSpiceShared instance after ``analysis_command`` completes;
            should return the scalar loss. Use ``ngspice_shared.plot('ac1').nodes``
            etc. to read results.
    """

    def __init__(
        self,
        netlist_template: str,
        design_var_names: Sequence[str],
        variation_var_names: Sequence[str],
        analysis_command: str,
        extractor: Callable[["NgSpiceShared", np.ndarray, np.ndarray], float],
        title: str = "ZO_yield_circuit",
    ):
        super().__init__()
        self.netlist_template = netlist_template
        self.design_var_names = list(design_var_names)
        self.variation_var_names = list(variation_var_names)
        self.analysis_command = analysis_command
        self.extractor = extractor
        self.title = title

        try:
            _ = _ng()
            self._ngspice_ok = True
        except Exception:
            self._ngspice_ok = False

    # ----------------------------------------------------------------------- #
    def _render(self, x: np.ndarray, xi: np.ndarray) -> str:
        if x.shape[0] != len(self.design_var_names):
            raise ValueError(
                f"x has {x.shape[0]} entries but design_var_names has {len(self.design_var_names)}"
            )
        if xi.shape[0] != len(self.variation_var_names):
            raise ValueError(
                f"xi has {xi.shape[0]} entries but variation_var_names has {len(self.variation_var_names)}"
            )
        kwargs = {n: float(v) for n, v in zip(self.design_var_names, x)}
        kwargs.update({n: float(v) for n, v in zip(self.variation_var_names, xi)})
        return self.netlist_template.format(**kwargs)

    # ----------------------------------------------------------------------- #
    def evaluate(self, x: np.ndarray, xi: np.ndarray) -> float:
        self.n_calls += 1
        if not self._ngspice_ok:
            self.n_failures += 1
            return self.LOSS_PENALTY
        try:
            netlist_str = self._render(x, xi)
        except Exception:
            self.n_failures += 1
            return self.LOSS_PENALTY

        with _NG_LOCK:
            try:
                ng = _ng()
                # remove any prior circuit
                try:
                    ng.remove_circuit()
                except Exception:
                    pass
                ng.load_circuit(netlist_str)
                ng.exec_command(self.analysis_command)
                loss = float(self.extractor(ng, x, xi))
                if not math.isfinite(loss):
                    self.n_failures += 1
                    return self.LOSS_PENALTY
                return loss
            except Exception:
                self.n_failures += 1
                return self.LOSS_PENALTY
