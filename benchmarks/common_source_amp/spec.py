"""Common-source amplifier benchmark spec.

Design vars (n = 6):
  w1, l1   -- input NMOS dimensions    [m]
  w2, l2   -- load PMOS dimensions     [m]
  ibias    -- mirror reference current [A]
  cload    -- output load capacitor    [F]

Process variations (d = 10):
  dvth_n, dkp_n, dl_n, dw_n    -- NMOS-local
  dvth_p, dkp_p, dl_p, dw_p    -- PMOS-local
  dvth_g, dkp_g                -- global Vth/Kp shifts

Loss:
  L(x, xi) = -gain_dB
             + lambda_ugbw * max(0, ugbw_target - ugbw) / ugbw_target
             + lambda_p    * power_mW
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

import math as _math

from simulators.base import BlackBoxSimulator
from simulators.pyspice_sim import PySpiceSimulator
from zo_yield.loss import Penalty, combined_loss
from zo_yield.samplers import IndependentGaussianSampler


# Smooth-loss penalties. Defaults are lenient because the
# CS amp is an easy benchmark; the regression check just requires
# yield_final >= yield_init.
DEFAULT_ALPHAS = dict(
    alpha_gain=0.5,         # per dB
    alpha_ugbw_log=1.0,     # per decade of log10(Hz)
    alpha_power_mW=0.5,     # per mW
)


def make_penalties(targets=None, alphas=None):
    t = targets or DEFAULT_TARGETS
    a = dict(DEFAULT_ALPHAS)
    if alphas:
        a.update(alphas)
    return [
        Penalty(
            name="gain_dB",
            extract=lambda s: s["gain_db"],
            kind="ge",
            target=20.0,                  # CS amp gain spec
            weight=1.0,
            alpha=a["alpha_gain"],
        ),
        Penalty(
            name="ugbw_log",
            extract=lambda s: _math.log10(max(s["ugbw_hz"], 1.0)),
            kind="ge",
            target=_math.log10(max(t["ugbw_target"], 1.0)),
            weight=2.0,
            alpha=a["alpha_ugbw_log"],
        ),
        Penalty(
            name="power_mW",
            extract=lambda s: s["power_w"] * 1e3,
            kind="le",
            target=0.5,                   # 0.5 mW max
            weight=1.0,
            alpha=a["alpha_power_mW"],
        ),
    ]


HERE = os.path.dirname(os.path.abspath(__file__))


DEFAULT_TARGETS = dict(
    ugbw_target=1.0e6,
    lambda_ugbw=20.0,
    lambda_p=10.0,
    vdd=1.0,
    rfb=10e6,        # 10 MOhm feedback for self-bias
    cin=1e-9,        # 1 nF AC coupling cap (>> CL)
)


DESIGN_VARS = ["w1", "l1", "w2", "l2", "ibias", "cload"]
VARIATION_VARS = [
    "dvth_n", "dkp_n", "dl_n", "dw_n",
    "dvth_p", "dkp_p", "dl_p", "dw_p",
    "dvth_g", "dkp_g",
]

X_NOMINAL = np.array([20e-6, 0.18e-6, 40e-6, 0.5e-6, 50e-6, 1e-12])
X_LO = np.array([2e-6, 0.045e-6, 4e-6, 0.045e-6, 5e-6, 0.1e-12])
X_HI = np.array([200e-6, 2e-6, 400e-6, 4e-6, 500e-6, 100e-12])


def make_sampler(scale: float = 1.0) -> IndependentGaussianSampler:
    sigmas = np.array([
        0.020 * scale,
        0.05 * scale,
        5e-9 * scale,
        0.02e-6 * scale,
        0.020 * scale,
        0.05 * scale,
        5e-9 * scale,
        0.02e-6 * scale,
        0.010 * scale,
        0.02 * scale,
    ])
    return IndependentGaussianSampler(mean=np.zeros_like(sigmas), std=sigmas)


# --------------------------------------------------------------------------- #
# netlist template helper                                                     #
# --------------------------------------------------------------------------- #
def _read_template() -> str:
    with open(os.path.join(HERE, "netlist.cir"), "r") as f:
        return f.read()


def _render_kwargs(x: np.ndarray, xi: np.ndarray, targets: dict) -> dict:
    """Build the Python-side kwargs that fully specify the netlist."""
    w1, l1, w2, l2, ibias, cload = [float(v) for v in x]
    (dvth_n, dkp_n, dl_n, dw_n,
     dvth_p, dkp_p, dl_p, dw_p,
     dvth_g, dkp_g) = [float(v) for v in xi]

    # effective model params (baseline values are synthetic 45 nm-ish)
    vto_n = 0.4 + dvth_n + dvth_g
    vto_p = -0.4 + dvth_p + dvth_g
    kp_n = 300e-6 * (1.0 + dkp_n + dkp_g)
    kp_p = 120e-6 * (1.0 + dkp_p + dkp_g)
    wn_eff = max(w1 + dw_n, 1e-9)
    ln_eff = max(l1 + dl_n, 1e-9)
    wp_eff = max(w2 + dw_p, 1e-9)
    lp_eff = max(l2 + dl_p, 1e-9)

    return dict(
        vto_n=vto_n, vto_p=vto_p, kp_n=kp_n, kp_p=kp_p,
        wn_eff=wn_eff, ln_eff=ln_eff, wp_eff=wp_eff, lp_eff=lp_eff,
        ibias=ibias, cload=cload,
        vdd=targets["vdd"],
        rfb=targets["rfb"],
        cin=targets["cin"],
    )


# --------------------------------------------------------------------------- #
# simulator class                                                             #
# --------------------------------------------------------------------------- #
class CSAmpSimulator(BlackBoxSimulator):  # noqa: E302 (CV: above is fine)
    """CS amp simulator -- subprocess-isolated ngspice.

    discovered that PySpice's in-process ``NgSpiceShared`` caches
    ``.model`` cards across calls — variations had no effect. This benchmark
    now uses ``simulators.ngspice_subprocess.run_ngspice_subprocess``: one
    fresh ngspice subprocess per simulation (~250 ms each).
    """

    def __init__(self, targets: Optional[dict] = None,
                 alphas: Optional[dict] = None):
        super().__init__()
        self.targets = targets or DEFAULT_TARGETS
        self.alphas = alphas or {}
        self.template = _read_template()
        # Probe whether subprocess invocation works.
        try:
            from simulators.ngspice_subprocess import run_ngspice_subprocess  # noqa
            self._ok = True
        except Exception:
            self._ok = False

    # ----------------------------------------------------------------------- #
    def _render(self, x: np.ndarray, xi: np.ndarray) -> str:
        kw = _render_kwargs(x, xi, self.targets)
        return self.template.format(**kw)

    # ----------------------------------------------------------------------- #
    def _run_and_extract(self, netlist_str: str, x: np.ndarray) -> dict:
        """Run one subprocess sim and return metrics dict."""
        from simulators.ngspice_subprocess import run_ngspice_subprocess
        res = run_ngspice_subprocess(
            netlist_str,
            analysis_cmd="ac dec 20 1 1g",
            plot_name="ac1",
            nodes=["out", "frequency"],
        )
        vout = res["out"]
        freqs = np.real(res["frequency"])
        mag = np.abs(vout)
        gain_db_arr = 20.0 * np.log10(np.maximum(mag, 1e-30))

        # mid-band gain = max over the response (peak away from input high-pass)
        idx_peak = int(np.argmax(gain_db_arr))
        gain_db = float(gain_db_arr[idx_peak])

        # UGBW = first 0-dB crossing *after* the peak
        ugbw = 0.0
        if gain_db > 0:
            tail = gain_db_arr[idx_peak:]
            below = np.where(tail < 0)[0]
            if below.size == 0:
                ugbw = float(freqs[-1])
            else:
                i = idx_peak + below[0]
                if i == 0:
                    ugbw = float(freqs[0])
                else:
                    f0, f1 = float(freqs[i - 1]), float(freqs[i])
                    g0, g1 = float(gain_db_arr[i - 1]), float(gain_db_arr[i])
                    if g1 == g0:
                        ugbw = f0
                    else:
                        logf = np.log10(f0) + (np.log10(f1) - np.log10(f0)) * (0 - g0) / (g1 - g0)
                        ugbw = 10.0**logf

        phase = np.unwrap(np.angle(vout)) * 180.0 / np.pi
        if ugbw > 0:
            ph_at_ugbw = float(np.interp(np.log10(ugbw), np.log10(np.maximum(freqs, 1e-30)), phase))
        else:
            ph_at_ugbw = -180.0
        pm = 180.0 + ph_at_ugbw

        ibias = x[4]
        power = self.targets["vdd"] * 2.0 * ibias
        return dict(gain_db=gain_db, ugbw_hz=ugbw, phase_margin_deg=pm, power_w=power)

    # ----------------------------------------------------------------------- #
    def _loss(self, metrics: dict) -> float:
        """Smooth-shortfall loss plus a small gain bonus."""
        pens = make_penalties(targets=self.targets, alphas=self.alphas)
        # objective term: a small bonus for additional gain beyond target keeps
        # the optimizer pushing past the target rather than parking at the edge.
        objective = -0.1 * metrics["gain_db"]
        return float(combined_loss(pens, dict(metrics), objective=objective))

    # ----------------------------------------------------------------------- #
    def evaluate(self, x: np.ndarray, xi: np.ndarray) -> float:
        self.n_calls += 1
        if not self._ok:
            self.n_failures += 1
            return self.LOSS_PENALTY
        try:
            netlist = self._render(x, xi)
        except Exception:
            self.n_failures += 1
            return self.LOSS_PENALTY
        try:
            metrics = self._run_and_extract(netlist, x)
            loss = self._loss(metrics)
            import math
            if not math.isfinite(loss):
                self.n_failures += 1
                return self.LOSS_PENALTY
            return loss
        except Exception:
            self.n_failures += 1
            return self.LOSS_PENALTY

    # ----------------------------------------------------------------------- #
    def evaluate_with_metrics(self, x: np.ndarray, xi: np.ndarray) -> tuple[float, dict]:
        """Full diagnostic evaluation -- loss + metric dict. No penalty masking."""
        netlist = self._render(x, xi)
        metrics = self._run_and_extract(netlist, x)
        return self._loss(metrics), metrics


def build_simulator(targets: Optional[dict] = None,
                    alphas: Optional[dict] = None) -> CSAmpSimulator:
    return CSAmpSimulator(targets=targets, alphas=alphas)


# --------------------------------------------------------------------------- #
# Yield specs                                                 #
# --------------------------------------------------------------------------- #
from zo_yield.evaluation import Spec  # noqa: E402


def build_specs(targets: Optional[dict] = None) -> list:
    t = targets or DEFAULT_TARGETS
    return [
        Spec(
            name="gain_db",
            extract=lambda m: m["gain_db"],
            satisfies=lambda v: v >= 20.0,            # CS amp target: 20 dB min
            margin=lambda v: 20.0 - v,
        ),
        Spec(
            name="ugbw_hz",
            extract=lambda m: m["ugbw_hz"],
            satisfies=lambda v: v >= t["ugbw_target"],
            margin=lambda v: t["ugbw_target"] - v,
        ),
        Spec(
            name="power_w",
            extract=lambda m: m["power_w"],
            satisfies=lambda v: v <= 0.5e-3,          # 0.5 mW max
            margin=lambda v: v - 0.5e-3,
        ),
    ]


SPECS = build_specs()

